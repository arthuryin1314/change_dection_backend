import asyncio
import httpx
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds


GEOSERVER_URL =os.environ.get('GEOSERVER_URL')
GEOSERVER_USER =os.environ.get('GEOSERVER_USER')
GEOSERVER_PASSWORD =os.environ.get('GEOSERVER_PASSWORD')
GEOSERVER_WORKSPACE =os.environ.get('GEOSERVER_WORKSPACE')

# 统一校验所有必填环境变量
_required_env_vars = {
    "GEOSERVER_URL": GEOSERVER_URL,
    "GEOSERVER_USER": GEOSERVER_USER,
    "GEOSERVER_PASSWORD": GEOSERVER_PASSWORD,
    "GEOSERVER_WORKSPACE": GEOSERVER_WORKSPACE,
}
missing_env_vars = [name for name, value in _required_env_vars.items() if not value]
if missing_env_vars:
    raise RuntimeError(
        "以下 GeoServer 环境变量必须通过环境变量设置且不能为空: "
        + ", ".join(missing_env_vars)
    )

_AUTH     = (GEOSERVER_USER, GEOSERVER_PASSWORD)
_XML_HDR  = {"Content-Type": "application/xml"}
_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_RETRY_DELAY_SECONDS = 0.5


async def _req(method: str, url: str, **kwargs) -> httpx.Response:
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.request(method, url, auth=_AUTH, **kwargs)

            # 对暂时性服务异常做有限重试，避免瞬时抖动直接失败。
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BASE_RETRY_DELAY_SECONDS * (2 ** attempt))
                continue

            return response
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            last_error = exc
            if attempt >= _MAX_RETRIES - 1:
                break
            await asyncio.sleep(_BASE_RETRY_DELAY_SECONDS * (2 ** attempt))

    raise RuntimeError(
        f"GeoServer 请求失败，已重试 {_MAX_RETRIES} 次: {method} {url}"
    ) from last_error


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


def _to_geoserver_path(tif_path: Path) -> str:
    """
    把 Windows 绝对路径转成 GeoServer file: URL。
    Path.as_posix() 会把反斜杠转为正斜杠：
      E:\\change_detection\\...\\xxx.tif
      → file:///E:/change_detection/.../xxx.tif
    """
    posix = tif_path.absolute().as_posix()   # E:/change_detection/.../xxx.tif
    return f"file:///{posix}"                # file:///E:/change_detection/.../xxx.tif


def _build_coverage_store_xml(layer_name: str, workspace: str, file_url: str) -> str:
    root = ET.Element("coverageStore")
    ET.SubElement(root, "name").text = layer_name
    ET.SubElement(root, "workspace").text = workspace
    ET.SubElement(root, "enabled").text = "true"
    ET.SubElement(root, "type").text = "GeoTIFF"
    ET.SubElement(root, "url").text = file_url
    return ET.tostring(root, encoding="unicode")


def _build_coverage_xml(layer_name: str, native_name: str | None = None) -> str:
    root = ET.Element("coverage")
    ET.SubElement(root, "name").text = layer_name
    ET.SubElement(root, "title").text = layer_name
    ET.SubElement(root, "enabled").text = "true"
    if native_name is not None:
        ET.SubElement(root, "nativeName").text = native_name
    return ET.tostring(root, encoding="unicode")


async def publish_geotiff_layer(tif_path: Path, layer_name: str) -> str:
    """
    同机 Windows：让 GeoServer 直接读取本地 TIF 文件路径，发布为图层。
    返回图层的 WMS GetMap URL 前缀。
    """
    await _ensure_workspace()

    file_url = _to_geoserver_path(tif_path)

    # ── Step 1：创建 CoverageStore ─────────────────────────────────────
    store_xml = _build_coverage_store_xml(
        layer_name=layer_name,
        workspace=GEOSERVER_WORKSPACE,
        file_url=file_url,
    )

    resp = await _req(
        "POST",
        f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WORKSPACE}/coveragestores",
        content=store_xml,
        headers=_XML_HDR,
    )
    if resp.status_code == 409:
        pass
    elif resp.status_code not in (200, 201):
        raise RuntimeError(f"创建 CoverageStore 失败 [{resp.status_code}]: {resp.text}")

    # ── Step 2：发布 Coverage（图层） ──────────────────────────────────
    # GeoTIFF 的原生 coverage 名通常来自文件名（不含扩展名），
    # 不能直接用业务图层名，否则可能报 coverageName not supported。
    native_name = tif_path.stem
    coverage_xml = _build_coverage_xml(layer_name=layer_name, native_name=native_name)

    resp = await _req(
        "POST",
        f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WORKSPACE}"
        f"/coveragestores/{layer_name}/coverages",
        content=coverage_xml,
        headers=_XML_HDR,
    )
    if resp.status_code == 409:
        pass
    elif resp.status_code not in (200, 201):
        # 兼容部分 GeoServer 场景：不显式传 nativeName 再试一次
        if "not supported" in resp.text.lower() or resp.status_code == 500:
            fallback_xml = _build_coverage_xml(layer_name=layer_name)
            retry = await _req(
                "POST",
                f"{GEOSERVER_URL}/rest/workspaces/{GEOSERVER_WORKSPACE}"
                f"/coveragestores/{layer_name}/coverages",
                content=fallback_xml,
                headers=_XML_HDR,
            )
            if retry.status_code in (200, 201, 409):
                pass
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