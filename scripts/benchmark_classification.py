import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from services.classification_generation import (
    GenerationRequest,
    generate_classification_files,
)
from utils.content_hash import resolve_content_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-id", required=True)
    args = parser.parse_args()

    weight_sha256 = resolve_content_sha256(args.weight).sha256
    request = GenerationRequest(
        result_id=args.result_id,
        image_path=args.source,
        weight_file_path=str(args.weight),
        weight_sha256=weight_sha256,
        storage_root=args.output_root,
    )
    started = perf_counter()

    def report(metrics) -> None:
        print(
            json.dumps(
                {
                    "event": "progress",
                    "elapsed_seconds": perf_counter() - started,
                    "total_tiles": metrics.total_tiles,
                    "effective_tiles": metrics.effective_tiles,
                    "skipped_tiles": metrics.skipped_tiles,
                    "source_read_seconds": metrics.source_read_seconds,
                    "inference_seconds": metrics.inference_seconds,
                    "compressed_write_seconds": metrics.compressed_write_seconds,
                }
            ),
            flush=True,
        )

    outcome = generate_classification_files(request, progress_callback=report)
    print(
        json.dumps(
            {
                "event": "complete",
                "stored": {
                    "classes_path": str(outcome.stored.classes_path),
                    "valid_mask_path": str(outcome.stored.valid_mask_path),
                },
                "metadata": asdict(outcome.metadata),
                "metrics": asdict(outcome.metrics),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
