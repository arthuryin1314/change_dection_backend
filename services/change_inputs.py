from dataclasses import dataclass
from datetime import datetime

from utils.result_source import ResultSourceSnapshot


@dataclass(frozen=True)
class ResolvedPeriod:
    id: str
    identity_sha256: str
    source_image_id: int
    source_model_id: int
    source_snapshot: ResultSourceSnapshot | None
    image_content_sha256: str
    weight_content_sha256: str
    inference_parameters: dict
    classification_scheme_version: str
    pipeline_version: str
    grid_policy_version: str
    completed_at: datetime
    classes_path: str
    valid_mask_path: str
    area_status: str
    class_area_m2: list | None
    area_completed_at: datetime | None
    area_failure_detail: str | None

    @classmethod
    def from_row(cls, row, source_snapshot: ResultSourceSnapshot | None):
        return cls(
            id=row.id,
            identity_sha256=row.identity_sha256,
            source_image_id=row.source_image_id,
            source_model_id=row.source_model_id,
            source_snapshot=source_snapshot,
            image_content_sha256=row.image_content_sha256,
            weight_content_sha256=row.weight_content_sha256,
            inference_parameters=dict(row.inference_parameters),
            classification_scheme_version=row.classification_scheme_version,
            pipeline_version=row.pipeline_version,
            grid_policy_version=row.grid_policy_version,
            completed_at=row.completed_at,
            classes_path=row.classes_path,
            valid_mask_path=row.valid_mask_path,
            area_status=row.area_status,
            class_area_m2=row.class_area_m2,
            area_completed_at=row.area_completed_at,
            area_failure_detail=row.area_failure_detail,
        )

    @property
    def snapshot(self) -> dict:
        return {
            "result_id": self.id,
            "identity_sha256": self.identity_sha256,
            "source_image_id": self.source_image_id,
            "source_model_id": self.source_model_id,
            "source": self.source_snapshot,
            "image_content_sha256": self.image_content_sha256,
            "weight_content_sha256": self.weight_content_sha256,
            "inference_parameters": self.inference_parameters,
            "classification_scheme_version": self.classification_scheme_version,
            "pipeline_version": self.pipeline_version,
            "grid_policy_version": self.grid_policy_version,
            "completed_at": self.completed_at.isoformat(),
            "area_status": self.area_status,
            "class_area_m2": self.class_area_m2,
            "area_completed_at": (
                self.area_completed_at.isoformat()
                if self.area_completed_at is not None
                else None
            ),
            "area_failure_detail": self.area_failure_detail,
        }


@dataclass(frozen=True)
class ResolvedChangeInputs:
    before: ResolvedPeriod
    after: ResolvedPeriod

    @property
    def before_snapshot(self) -> dict:
        return self.before.snapshot

    @property
    def after_snapshot(self) -> dict:
        return self.after.snapshot
