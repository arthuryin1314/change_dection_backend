import asyncio
from datetime import datetime, timezone
from pathlib import Path

from crud import change_results as change_crud
from crud import classification_results as classification_crud
from crud import images as image_crud
from utils.change_report_images import read_period_images
from utils.change_report_pdf import build_pdf

class ReportError(Exception):
    def __init__(self, status, message, missing_items=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.missing_items = missing_items or []

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
        snapshot = getattr(row, f"{name}_snapshot") or {}
        source = snapshot.get("source") or {}
        image_ref = source.get("image") or {}
        image_id = snapshot.get("source_image_id") or image_ref.get("id")
        classification_id = snapshot.get("result_id")
        classification = await classification_crud.get_result_by_id(db, classification_id, user_id) if classification_id else None
        image = await image_crud.get_image_by_id(db, image_id, user_id) if image_id else None
        if classification is None or classification.status != "SUCCEEDED":
            missing.append({"period": name, "resource": "classification"})
            continue
        if image is None or image.img_path is None:
            missing.append({"period": name, "resource": "original"})
            continue
        if not classification.classes_path or not classification.valid_mask_path:
            missing.append({"period": name, "resource": "mask"})
            continue
        periods[name] = (snapshot, classification, image)
    if missing:
        raise ReportError(409, "报告所需资源缺失", missing)
    def render():
        report = {"title":"变化检测结果报告","result_id":row.id,"detection_time":row.calculated_at.isoformat(),"generated_at":datetime.now(timezone.utc).isoformat(),"unit":unit,"model":None,"matrix_m2":row.matrix_m2,"common_valid_area_m2":row.common_valid_area_m2,"classes":[{"id":i,"name":name} for i,name in enumerate(("其他／背景","水系","林地","道路","种植土地","房屋建筑"))]}
        for name, (snapshot, classification, image) in periods.items():
            classification.image_path = image.img_path
            original, classified = read_period_images(classification)
            source = snapshot.get("source") or {}
            image_meta = source.get("image") or {}
            report[name] = {"original_png":original,"classification_png":classified,"name":image_meta.get("name"),"capture_date":image_meta.get("capture_date"),"satellite":image_meta.get("satellite"),"resolution":image_meta.get("resolution"),"crs":classification.crs,"size":f"{classification.raster_width} × {classification.raster_height}"}
        report["before_area"] = periods["before"][1].class_area_m2
        report["after_area"] = periods["after"][1].class_area_m2
        report["model"] = ((periods["before"][0].get("source") or {}).get("model") or {}).get("name")
        return build_pdf(report)
    return await asyncio.to_thread(render)

