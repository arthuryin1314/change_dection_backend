import argparse
import os
from pathlib import Path
import sys
import time
from io import BytesIO
from types import SimpleNamespace

from dotenv import load_dotenv


load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from config.db_config import get_db
from router import identification_results
from utils.get_user_by_token import get_current_user


def verify(user_id: int, image_id: int, model_id: int, timeout: float) -> None:
    app = FastAPI()
    app.include_router(identification_results.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)

    started = time.perf_counter()
    with TestClient(app) as client:
        created = client.post(
            "/api/identification-results",
            json={"image_id": image_id, "model_id": model_id},
        )
        created.raise_for_status()
        result_id = created.json()["data"]["result_id"]
        post_seconds = time.perf_counter() - started

        deadline = time.monotonic() + timeout
        while True:
            fetched = client.get(f"/api/identification-results/{result_id}")
            fetched.raise_for_status()
            data = fetched.json()["data"]
            if data["status"] == "SUCCEEDED":
                break
            if data["status"] == "FAILED":
                raise RuntimeError(data["failure_detail"])
            if time.monotonic() >= deadline:
                raise TimeoutError(f"result {result_id} did not finish in {timeout}s")
            time.sleep(0.25)

        grid = data["grid"]
        rendered = client.get(
            f"/api/identification-results/{result_id}/render",
            params={
                "bbox": ",".join(str(value) for value in grid["bounds"]),
                "width": 256,
                "height": 256,
                "srs": grid["crs"],
            },
        )
        rendered.raise_for_status()
        overlay = Image.open(BytesIO(rendered.content))
        overlay.load()

    print(
        f"verified result_id={result_id} post_seconds={post_seconds:.3f} "
        f"total_seconds={time.perf_counter() - started:.3f} "
        f"render={overlay.mode}:{overlay.width}x{overlay.height}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--model-id", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    verify(args.user_id, args.image_id, args.model_id, args.timeout)
