"""Convert a DiffSynth multi-annotation config to Wan-Trainer I2V parquet.

The input config is a dict whose values contain ``annotation`` and optional
``root`` fields. Each annotation may be JSON or JSONL. Output rows use the
standard I2V columns consumed by ``I2VDataset``:

    video: absolute local path or s3:// URI
    prompt: text prompt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.remote_io import resolve_media_path


_VIDEO_KEYS = ("clip_path", "video_path", "video", "path")
_PROMPT_KEYS = ("text_annot", "prompt", "caption")
_SCHEMA = pa.schema(
    [
        ("video", pa.string()),
        ("prompt", pa.string()),
        ("task_name", pa.string()),
        ("source_annotation", pa.string()),
        ("source_index", pa.int64()),
        ("width", pa.int64()),
        ("height", pa.int64()),
        ("fps", pa.int64()),
        ("length", pa.int64()),
        ("id", pa.int64()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Copied DiffSynth mix JSON")
    parser.add_argument("--output_parquet", required=True, help="Output parquet path")
    parser.add_argument("--output_json", required=True, help="Output I2V dataset JSON path")
    parser.add_argument("--batch_size", type=int, default=10000, help="Rows per parquet writer batch")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for smoke datasets")
    return parser.parse_args()


def iter_annotation(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        yield item
        return

    data = json.loads(path.read_text())
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield data


def pick_first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def flush(writer: pq.ParquetWriter | None, rows: list[dict[str, Any]], output_parquet: Path) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    if writer is None:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_parquet, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_parquet = Path(args.output_parquet)
    output_json = Path(args.output_json)

    raw = json.loads(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level dict in {config_path}")

    rows: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    total = 0
    skipped_empty = 0
    skipped_missing_fields = 0

    for task_name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        annotation_path = Path(entry["annotation"])
        root = entry.get("root", "") or ""
        source_index = 0
        any_rows = False
        for record in iter_annotation(annotation_path):
            any_rows = True
            video = pick_first(record, _VIDEO_KEYS)
            prompt = pick_first(record, _PROMPT_KEYS)
            if video is None or prompt is None:
                skipped_missing_fields += 1
                source_index += 1
                continue
            rows.append(
                {
                    "video": resolve_media_path(str(video), root),
                    "prompt": str(prompt),
                    "task_name": str(task_name),
                    "source_annotation": str(annotation_path),
                    "source_index": source_index,
                    "width": record.get("width"),
                    "height": record.get("height"),
                    "fps": record.get("fps"),
                    "length": record.get("length"),
                    "id": record.get("id"),
                }
            )
            total += 1
            source_index += 1
            if len(rows) >= args.batch_size:
                writer = flush(writer, rows, output_parquet)
            if args.limit is not None and total >= args.limit:
                break
        if not any_rows:
            skipped_empty += 1
        if args.limit is not None and total >= args.limit:
            break

    if rows:
        writer = flush(writer, rows, output_parquet)
    if writer is not None:
        writer.close()
    else:
        raise RuntimeError("No rows were written")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    dataset_config = [
        {
            "data_path": str(output_parquet.resolve()),
            "root": "",
            "num_frames": 81,
            "height": 384,
            "width": 384,
            "fps": 16,
        }
    ]
    output_json.write_text(json.dumps(dataset_config, indent=2) + "\n")

    print(
        json.dumps(
            {
                "rows": total,
                "skipped_empty_annotations": skipped_empty,
                "skipped_missing_fields": skipped_missing_fields,
                "output_parquet": str(output_parquet),
                "output_json": str(output_json),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
