"""Extract native front-camera 2D labels from downloaded Waymo v2 Parquet components."""

import io
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .data import SPLITS, validate_frames, write_jsonl

KEYS = ["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name"]
IMAGE_COLUMN = "[CameraImageComponent].image"
BOX_PREFIX = "[CameraBoxComponent]"
BOX_COLUMNS = [f"{BOX_PREFIX}.type"] + [
    f"{BOX_PREFIX}.box.{part}.{axis}" for part in ("center", "size") for axis in ("x", "y")
]
WAYMO_LABELS = {1: "vehicle", 2: "pedestrian", 4: "cyclist"}


def extract(images_dir, boxes_dir, output, split="train", stride=10, max_frames=5000):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install the Waymo adapter: pip install -e '.[waymo]'") from exc
    if split not in SPLITS or stride < 1 or max_frames < 1:
        raise ValueError("Choose a valid split and positive stride/max_frames")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Waymo extraction output must be empty")
    image_files = sorted(Path(images_dir).glob("*.parquet"))
    if not image_files:
        raise ValueError("No camera_image Parquet files found")
    (root / "images").mkdir(parents=True, exist_ok=True)
    frames, seen, counters = [], set(), defaultdict(int)
    for image_file in image_files:
        box_file = Path(boxes_dir) / image_file.name
        if not box_file.exists():
            raise ValueError(f"Missing paired camera_box component: {box_file}")
        annotations = defaultdict(list)
        for batch in pq.ParquetFile(box_file).iter_batches(
            columns=KEYS + BOX_COLUMNS, batch_size=4096
        ):
            for row in batch.to_pylist():
                if row[KEYS[2]] != 1 or row[BOX_COLUMNS[0]] not in WAYMO_LABELS:
                    continue
                key = (row[KEYS[0]], row[KEYS[1]], row[KEYS[2]])
                cx, cy, width, height = (float(row[c]) for c in BOX_COLUMNS[1:])
                annotations[key].append(
                    {
                        "label": WAYMO_LABELS[row[BOX_COLUMNS[0]]],
                        "bbox": [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
                    }
                )
        # Small image batches avoid materializing all cameras for an entire segment.
        for batch in pq.ParquetFile(image_file).iter_batches(
            columns=KEYS + [IMAGE_COLUMN], batch_size=8
        ):
            for row in batch.to_pylist():
                if row[KEYS[2]] != 1:
                    continue
                sequence, timestamp = str(row[KEYS[0]]), int(row[KEYS[1]])
                counters[sequence] += 1
                if (counters[sequence] - 1) % stride:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", sequence):
                    raise ValueError("Unexpected segment identifier")
                key = (row[KEYS[0]], row[KEYS[1]], row[KEYS[2]])
                image_id = f"{sequence}_{timestamp}_FRONT"
                if image_id in seen:
                    raise ValueError(f"Duplicate camera frame: {image_id}")
                seen.add(image_id)
                raw = row[IMAGE_COLUMN]
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                clipped = []
                for ann in annotations[key]:
                    x1, y1, x2, y2 = ann["bbox"]
                    box = [
                        max(0, min(width, x1)),
                        max(0, min(height, y1)),
                        max(0, min(width, x2)),
                        max(0, min(height, y2)),
                    ]
                    if box[2] > box[0] and box[3] > box[1]:
                        clipped.append({"label": ann["label"], "bbox": box})
                file_name = f"images/{image_id}.jpg"
                (root / file_name).write_bytes(raw)
                frames.append(
                    {
                        "image_id": image_id,
                        "sequence_id": sequence,
                        "split": split,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                        "annotations": clipped,
                        "tags": {"camera": "FRONT", "source": "waymo_v2"},
                    }
                )
                if len(frames) >= max_frames:
                    break
            if len(frames) >= max_frames:
                break
        if len(frames) >= max_frames:
            break
    validate_frames(frames)
    write_jsonl(root / "manifest.jsonl", frames)
    return {
        "frames": len(frames),
        "sequences": len({f["sequence_id"] for f in frames}),
        "split": split,
        "manifest": str(root / "manifest.jsonl"),
    }
