CLASSIFICATION_SCHEME_VERSION = "land-cover-6/v1"
PIPELINE_VERSION = "deeplab-native/v1"
GRID_POLICY_VERSION = "native-v1"
INFERENCE_TILE_SIZE = 512
INFERENCE_OVERLAP = 128
CLASS_NAMES = (
    "其他／背景",
    "水系",
    "林地",
    "道路",
    "种植土地",
    "房屋建筑",
)


def inference_parameters() -> dict[str, int]:
    return {
        "tile_size": INFERENCE_TILE_SIZE,
        "overlap": INFERENCE_OVERLAP,
    }


def ordered_class_definitions() -> list[dict[str, int | str]]:
    return [
        {"id": class_id, "name": name}
        for class_id, name in enumerate(CLASS_NAMES)
    ]
