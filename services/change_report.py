import asyncio
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from crud import change_results as change_crud
from crud import classification_results as classification_crud
from crud import images as image_crud
from utils.change_report_images import read_period_images
from utils.change_report_pdf import build_pdf, format_crs_label, format_report_datetime
from utils.classification_area import validate_class_area_m2
from utils.change_artifacts import MAX_VISUAL_EDGE


class ReportError(Exception):
    def __init__(self, status, message, missing_items=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.missing_items = missing_items or []


def _valid_area(values):
    try:
        validate_class_area_m2(values)
    except (ValueError, TypeError):
        return False
    return True


def _resolve_area(snapshot, classification):
    snapshot_area = snapshot.get('class_area_m2')
    if _valid_area(snapshot_area):
        return snapshot_area
    classification_area = getattr(classification, 'class_area_m2', None)
    if _valid_area(classification_area):
        return classification_area
    return None


def _validate_image_identity(snapshot, classification, period):
    snapshot_hash = snapshot.get('image_content_sha256')
    classification_hash = getattr(classification, 'image_content_sha256', None)
    if snapshot_hash is not None and classification_hash is not None and snapshot_hash != classification_hash:
        return {"period": period, "resource": "image"}
    return None


def _artifact_path(row, artifact_name):
    metadata = getattr(row, "artifact_metadata", None)
    if not isinstance(metadata, dict):
        return None
    item = metadata.get(artifact_name)
    if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
        return None
    root = Path(os.environ.get("CHANGE_RESULT_ARTIFACT_DIR", "uploads/change_result_artifacts")).resolve()
    result_directory = (root / str(row.id)).resolve()
    path = (result_directory / item["filename"]).resolve()
    if path.parent != result_directory or not path.is_file():
        return None
    return path


def _read_artifact_image(path):
    with Image.open(path) as source:
        image = source.copy()
        image.thumbnail((MAX_VISUAL_EDGE, MAX_VISUAL_EDGE), Image.Resampling.NEAREST)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _load_transition_artifact(row):
    metadata = getattr(row, "artifact_metadata", None)
    if metadata is None or metadata == {}:
        return None
    if not isinstance(metadata, dict):
        raise ReportError(409, "报告所需资源缺失", [{"resource": "transition_artifact"}])
    image_path = _artifact_path(row, "visualization_png")
    legend_path = _artifact_path(row, "legend")
    snapshot = getattr(row, "transition_snapshot", None)
    if image_path is None or legend_path is None or not isinstance(snapshot, dict):
        raise ReportError(409, "报告所需资源缺失", [{"resource": "transition_artifact"}])
    try:
        legend = json.loads(legend_path.read_text(encoding="utf-8"))
        if not isinstance(legend, dict):
            raise ValueError("legend 必须是对象")
        image = _read_artifact_image(image_path)
    except (OSError, ValueError, json.JSONDecodeError):
        raise ReportError(409, "报告所需资源缺失", [{"resource": "transition_artifact"}])
    return image, legend, snapshot


async def _resolve_report_period(db, name, row, user_id):
    snapshot = getattr(row, f"{name}_snapshot") or {}
    source = snapshot.get("source") or {}
    image_ref = source.get("image") or {}
    image_id = snapshot.get("source_image_id") or image_ref.get("id")
    classification_id = snapshot.get("result_id")
    classification = await classification_crud.get_result_by_id(db, classification_id, user_id) if classification_id else None
    image = await image_crud.get_image_by_id(db, image_id, user_id) if image_id else None
    if classification is None or classification.status != "SUCCEEDED":
        return None, {"period": name, "resource": "classification"}
    identity_error = _validate_image_identity(snapshot, classification, name)
    if identity_error is not None:
        return None, identity_error
    if image is None or image.img_path is None or not Path(image.img_path).is_file():
        return None, {"period": name, "resource": "original"}
    if not classification.classes_path or not classification.valid_mask_path:
        return None, {"period": name, "resource": "mask"}
    area = _resolve_area(snapshot, classification)
    if area is None:
        return None, {"period": name, "resource": "area"}
    return (snapshot, classification, image, area), None


async def build_change_report(db, result_id, user_id, unit):
    row = await change_crud.get_succeeded_history_by_id(db, result_id, user_id)
    if row is None:
        raise ReportError(404, "历史记录不存在或无权访问")
    if row.status != "SUCCEEDED" or row.matrix_m2 is None or row.calculated_at is None:
        raise ReportError(409, "历史结果数据不完整")
    if unit not in ("m2", "ha", "km2"):
        raise ReportError(422, "面积单位无效")
    periods = {}
    missing = []
    for name in ("before", "after"):
        period, error = await _resolve_report_period(db, name, row, user_id)
        if error is not None:
            missing.append(error)
        else:
            periods[name] = period
    if missing:
        if all(item["resource"] == "area" for item in missing):
            raise ReportError(409, "单期面积未保存或不完整", missing)
        raise ReportError(409, "报告所需资源缺失", missing)

    def render():
        transition_artifact = _load_transition_artifact(row)
        report = {
            "title": "变化检测结果报告",
            "result_id": row.id,
            "detection_time": format_report_datetime(row.calculated_at),
            "generated_at": format_report_datetime(datetime.now(timezone.utc)),
            "unit": unit,
            "model": None,
            "matrix_m2": row.matrix_m2,
            "common_valid_area_m2": row.common_valid_area_m2,
            "classes": [{"id": i, "name": name} for i, name in enumerate(("其他／背景", "水系", "林地", "道路", "种植土地", "房屋建筑"))],
            "transition_png": None,
            "transition_legend": None,
            "transition_snapshot": None,
            "legacy_transition_note": "未保存多值转移结果；以下仅显示历史矩阵。",
        }
        if transition_artifact is not None:
            report["transition_png"], report["transition_legend"], report["transition_snapshot"] = transition_artifact
            classes = report["transition_legend"].get("classes")
            if isinstance(classes, list) and len(classes) == len(report["classes"]):
                report["classes"] = classes
            report["legacy_transition_note"] = None
        for name, (snapshot, classification, image, _) in periods.items():
            classification.image_path = image.img_path
            original, classified = read_period_images(classification)
            resolution = None if image.resolution is None else f"{format(image.resolution, 'f').rstrip('0').rstrip('.')} m"
            report[name] = {
                "original_png": original,
                "classification_png": classified,
                "name": image.image_name,
                "capture_date": image.capture_date.isoformat() if image.capture_date is not None else None,
                "satellite": image.satellite,
                "resolution": resolution,
                "crs": format_crs_label(classification.crs),
                "size": f"{classification.raster_width} × {classification.raster_height}",
            }
        report["before_area"] = periods["before"][3]
        report["after_area"] = periods["after"][3]
        report["model"] = ((periods["before"][0].get("source") or {}).get("model") or {}).get("name")
        return build_pdf(report)

    return await asyncio.to_thread(render)
