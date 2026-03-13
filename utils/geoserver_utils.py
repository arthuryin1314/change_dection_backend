import httpx
from pathlib import Path
import logging
import rasterio
from rasterio.warp import transform_bounds
logger = logging.getLogger(__name__)

# ── 配置（建议放到 .env） ──────────────────────────────────────────────────
GEOSERVER_URL       = "http://localhost:8080/geoserver"
GEOSERVER_USER      = "admin"
GEOSERVER_PASSWORD  = "geoserver"
GEOSERVER_WORKSPACE = "change-detection"
# ─────────────────────────────────────────────────────────────────────────

_AUTH     = (GEOSERVER_USER, GEOSERVER_PASSWORD)
_XML_HDR  = {"Content-Type": "application/xml"}


async def _req(method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.request(method, url, auth=_AUTH, **kwargs)


async def _ensure_workspace() -> None:
    resp = await _req("GET", f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WORKSPACE}")
    if resp.status_code == 404:
        r = await _req(
            "POST",
            f"{GEOSERVER_URL}/rest/workspaces",
            content=f"<workspace><name>{GEOSERVER_WORKSPACE}</name></workspace>",
            headers=_XML_HDR,
        )
        r.raise_for_status()
        logger.info(f"[GeoServer] workspace '{GEOSERVER_WORKSPACE}' 已创建")


def _to_geoserver_path(tif_path: Path) -> str:
    """
    把 Windows 绝对路径转成 GeoServer file: URL。
    Path.as_posix() 会把反斜杠转为正斜杠：
      E:\\change_detection\\...\\xxx.tif
      → file:///E:/change_detection/.../xxx.tif
    """
    posix = tif_path.absolute().as_posix()   # E:/change_detection/.../xxx.tif
    return f"file:///{posix}"                # file:///E:/change_detection/.../xxx.tif


async def publish_geotiff_layer(tif_path: Path, layer_name: str) -> str:
    """
    同机 Windows：让 GeoServer 直接读取本地 TIF 文件路径，发布为图层。
    返回图层的 WMS GetMap URL 前缀。
    """
    await _ensure_workspace()

    file_url = _to_geoserver_path(tif_path)
    logger.info(f"[GeoServer] 发布路径: {file_url}")

    # ── Step 1：创建 CoverageStore ─────────────────────────────────────
    store_xml = f"""<coverageStore>
  <name>{layer_name}</name>
  <workspace>{GEOSERVER_WORKSPACE}</workspace>
  <enabled>true</enabled>
  <type>GeoTIFF</type>
  <url>{file_url}</url>
</coverageStore>"""

    resp = await _req(
        "POST",
        f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WORKSPACE}/coveragestores",
        content=store_xml,
        headers=_XML_HDR,
    )
    if resp.status_code == 409:
        logger.warning(f"[GeoServer] CoverageStore '{layer_name}' 已存在，跳过创建")
    elif resp.status_code not in (200, 201):
        raise RuntimeError(f"创建 CoverageStore 失败 [{resp.status_code}]: {resp.text}")

    # ── Step 2：发布 Coverage（图层） ──────────────────────────────────
    # GeoTIFF 的原生 coverage 名通常来自文件名（不含扩展名），
    # 不能直接用业务图层名，否则可能报 coverageName not supported。
    native_name = tif_path.stem
    coverage_xml = f"""<coverage>
  <name>{layer_name}</name>
  <title>{layer_name}</title>
  <enabled>true</enabled>
  <nativeName>{native_name}</nativeName>
</coverage>"""

    resp = await _req(
        "POST",
        f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WORKSPACE}"
        f"/coveragestores/{layer_name}/coverages",
        content=coverage_xml,
        headers=_XML_HDR,
    )
    if resp.status_code == 409:
        logger.warning(f"[GeoServer] Coverage '{layer_name}' 已存在，跳过创建")
    elif resp.status_code not in (200, 201):
        # 兼容部分 GeoServer 场景：不显式传 nativeName 再试一次
        if "not supported" in resp.text.lower() or resp.status_code == 500:
            fallback_xml = f"""<coverage>
  <name>{layer_name}</name>
  <title>{layer_name}</title>
  <enabled>true</enabled>
</coverage>"""
            retry = await _req(
                "POST",
                f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WORKSPACE}"
                f"/coveragestores/{layer_name}/coverages",
                content=fallback_xml,
                headers=_XML_HDR,
            )
            if retry.status_code in (200, 201, 409):
                logger.warning(
                    f"[GeoServer] Coverage 首次发布失败，已用兼容模式成功发布: "
                    f"layer={layer_name}, native={native_name}"
                )
            else:
                raise RuntimeError(
                    f"发布 Coverage 失败 [{retry.status_code}]: {retry.text}"
                )
        else:
            raise RuntimeError(f"发布 Coverage 失败 [{resp.status_code}]: {resp.text}")

    wms_url = (
        f"{GEOSERVER_URL}/{GEOSERVER_WORKSPACE}/wms"
        f"?service=WMS&version=1.1.0&request=GetMap"
        f"&layers={GEOSERVER_WORKSPACE}:{layer_name}"
        f"&format=image/png"
    )
    logger.info(f"[GeoServer] 图层 '{GEOSERVER_WORKSPACE}:{layer_name}' 发布成功")
    return wms_url

def get_tif_bbox_wgs84(tif_path: Path) -> list[float]:
    """
    用 rasterio 读取 GeoTIFF 空间范围，统一转换为 WGS84。
    返回: [minLon, minLat, maxLon, maxLat]
    """
    with rasterio.open(str(tif_path)) as ds:
        bounds = ds.bounds
        crs    = ds.crs

        if crs is None:
            raise RuntimeError(f"TIF 文件缺少坐标系信息: {tif_path}")

        if crs.to_epsg() == 4326:
            return [
                round(bounds.left,   6),
                round(bounds.bottom, 6),
                round(bounds.right,  6),
                round(bounds.top,    6),
            ]

        lon_min, lat_min, lon_max, lat_max = transform_bounds(
            crs, "EPSG:4326",
            bounds.left, bounds.bottom, bounds.right, bounds.top
        )
        return [
            round(lon_min, 6),
            round(lat_min, 6),
            round(lon_max, 6),
            round(lat_max, 6),
        ]