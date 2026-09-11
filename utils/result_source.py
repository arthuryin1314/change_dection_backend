from typing import TypedDict


class SourceReference(TypedDict):
    id: int
    name: str | None


class ResultSourceSnapshot(TypedDict):
    image: SourceReference
    model: SourceReference


def result_source_snapshot(
    image_id: int,
    image_name: str | None,
    model_id: int,
    model_name: str | None,
) -> ResultSourceSnapshot:
    return {
        "image": {"id": image_id, "name": image_name},
        "model": {"id": model_id, "name": model_name},
    }
