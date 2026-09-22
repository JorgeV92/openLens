"""The dataset boundary: JSONL records, explicit boxes, and sequence-level splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

LABELS = ("vehicle", "pedestrian", "cyclist")
SPLITS = ("train", "val", "test")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def valid_box(box):
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(
            isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
            for v in box
        )
        and box[2] > box[0]
        and box[3] > box[1]
    )


def validate_frames(frames):
    if not frames:
        raise ValueError("The manifest is empty")
    seen, sequence_splits = set(), {}
    for frame in frames:
        for key in ("image_id", "sequence_id", "file_name", "split"):
            if not isinstance(frame.get(key), str) or not frame[key]:
                raise ValueError(f"Frame requires a nonempty {key}")
        image_id = frame["image_id"]
        if image_id in seen:
            raise ValueError(f"Duplicate image_id: {image_id}")
        seen.add(image_id)
        if frame["split"] not in SPLITS:
            raise ValueError(f"Invalid split for {image_id}")
        previous = sequence_splits.setdefault(frame["sequence_id"], frame["split"])
        if previous != frame["split"]:
            raise ValueError(f"Sequence leakage: {frame['sequence_id']} crosses splits")
        for key in ("width", "height"):
            if type(frame.get(key)) is not int or frame[key] <= 0:
                raise ValueError(f"Invalid {key} for {image_id}")
        if not isinstance(frame.get("annotations"), list):
            raise ValueError(f"Missing annotations for {image_id}; use [] for an empty frame")
        tags = frame.get("tags", {})
        if not isinstance(tags, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in tags.items()
        ):
            raise ValueError(f"tags must map strings to strings: {image_id}")
        for ann in frame["annotations"]:
            if (
                not isinstance(ann, dict)
                or ann.get("label") not in LABELS
                or not valid_box(ann.get("bbox"))
            ):
                raise ValueError(f"Invalid annotation for {image_id}")
            x1, y1, x2, y2 = ann["bbox"]
            if x1 < 0 or y1 < 0 or x2 > frame["width"] or y2 > frame["height"]:
                raise ValueError(f"Annotation outside image: {image_id}")
    return frames


def load_frames(path, split=None):
    frames = validate_frames(read_jsonl(path))
    if split is not None:
        frames = [f for f in frames if f["split"] == split]
        if not frames:
            raise ValueError(f"No frames in split {split!r}")
    return frames


def image_path(manifest, frame):
    return (Path(manifest).resolve().parent / frame["file_name"]).resolve()


def load_predictions(path, frames):
    expected = {f["image_id"] for f in frames}
    result = {}
    for row in read_jsonl(path):
        key = row.get("image_id")
        if not isinstance(key, str):
            raise ValueError("Prediction image_id must be a string")
        if key in result:
            raise ValueError(f"Duplicate prediction record: {key}")
        if key not in expected:
            raise ValueError(f"Prediction references an unknown image: {key}")
        if not isinstance(row.get("detections"), list):
            raise ValueError(f"Missing detections for {key}; use [] for no detections")
        for det in row["detections"]:
            if not isinstance(det, dict):
                raise ValueError(f"Invalid detection for {key}")
            score = det.get("score")
            if det.get("label") not in LABELS or not valid_box(det.get("bbox")):
                raise ValueError(f"Invalid detection for {key}")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError(f"Invalid detection score for {key}")
        result[key] = row["detections"]
    missing = expected - result.keys()
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} images (including empty frames)")
    return result


def split_for_sequence(sequence_id, seed=42, train_fraction=0.8, val_fraction=0.1):
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Train, validation, and test fractions must all be positive")
    digest = hashlib.sha256(f"{seed}:{sequence_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < train_fraction:
        return "train"
    return "val" if fraction < train_fraction + val_fraction else "test"


def split_manifest(source, destination, seed=42):
    frames = load_frames(source)
    for frame in frames:
        frame["split"] = split_for_sequence(frame["sequence_id"], seed)
        frame["file_name"] = os.path.relpath(
            image_path(source, frame), Path(destination).resolve().parent
        )
    write_jsonl(destination, frames)
    return {s: sum(f["split"] == s for f in frames) for s in SPLITS}
