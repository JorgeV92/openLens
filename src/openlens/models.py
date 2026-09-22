"""Hugging Face RT-DETR inference and a minimal supervised fine-tuning loop."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import LABELS, image_path, load_frames, write_jsonl

DEFAULT_CHECKPOINT = "PekingU/rtdetr_r18vd"
# A COCO person maps to a pedestrian; vehicles merge several COCO categories.
# COCO bicycle/motorcycle boxes do NOT match Waymo's rider+cycle definition.
COCO_TO_OPENLENS = {"person": "pedestrian", "car": "vehicle", "truck": "vehicle", "bus": "vehicle"}


def ml_imports():
    try:
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    except ImportError as exc:
        raise RuntimeError("Install model dependencies: pip install -e '.[ml]'") from exc
    return torch, RTDetrForObjectDetection, RTDetrImageProcessor


def resolve_device(torch, requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def synchronize(torch, device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def mapped_label(name):
    name = name.lower()
    return name if name in LABELS else COCO_TO_OPENLENS.get(name)


def predict(manifest, output, checkpoint=DEFAULT_CHECKPOINT, split=None, device="auto", min_score=0.01):
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")
    torch, Model, Processor = ml_imports()
    frames = load_frames(manifest, split)
    device = resolve_device(torch, device)
    processor = Processor.from_pretrained(checkpoint)
    model = Model.from_pretrained(checkpoint).to(device).eval()
    rows, times = [], []
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            synchronize(torch, device)
            start = time.perf_counter()
            with Image.open(image_path(manifest, frame)) as source:
                image = source.convert("RGB")
            if image.size != (frame["width"], frame["height"]):
                raise ValueError(f"Image dimensions disagree with manifest: {frame['image_id']}")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            decoded = processor.post_process_object_detection(
                outputs, target_sizes=[(frame["height"], frame["width"])], threshold=min_score
            )[0]
            detections = []
            for score, label, box in zip(decoded["scores"], decoded["labels"], decoded["boxes"]):
                name = mapped_label(model.config.id2label[int(label)])
                if name is None:
                    continue
                x1, y1, x2, y2 = box.detach().cpu().tolist()
                x1, x2 = [min(frame["width"], max(0, x)) for x in (x1, x2)]
                y1, y2 = [min(frame["height"], max(0, y)) for y in (y1, y2)]
                if x2 > x1 and y2 > y1:
                    detections.append(
                        {"label": name, "score": float(score), "bbox": [x1, y1, x2, y2]}
                    )
            synchronize(torch, device)
            elapsed = (time.perf_counter() - start) * 1000
            rows.append({"image_id": frame["image_id"], "detections": detections})
            if index > 0:
                times.append(elapsed)
    write_jsonl(output, rows)
    info = {
        "checkpoint": str(checkpoint),
        "device": device,
        "frames": len(rows),
        "min_score": min_score,
        "timing_scope": "decode + preprocess + inference + postprocess; first frame excluded",
        "timed_frames": len(times),
        "p50_ms": float(np.median(times)) if times else None,
        "p95_ms": float(np.percentile(times, 95)) if times else None,
        "label_mapping": "vehicle/pedestrian/cyclist checkpoint labels are preserved; COCO car/truck/bus->vehicle, person->pedestrian; cycles excluded",
    }
    Path(str(output) + ".meta.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def train(
    manifest,
    output,
    checkpoint=DEFAULT_CHECKPOINT,
    epochs=1,
    batch_size=2,
    learning_rate=1e-5,
    device="auto",
    seed=42,
    max_steps=None,
):
    if (
        epochs < 1
        or batch_size < 1
        or learning_rate <= 0
        or (max_steps is not None and max_steps < 1)
    ):
        raise ValueError("Epochs, batch size, learning rate, and max_steps must be positive")
    frames = load_frames(manifest, "train")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Training output directory must be empty; choose a new run directory")
    torch, Model, Processor = ml_imports()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = resolve_device(torch, device)
    processor = Processor.from_pretrained(checkpoint)
    id2label = dict(enumerate(LABELS))
    model = Model.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id={name: i for i, name in id2label.items()},
        ignore_mismatched_sizes=True,
    ).to(device)

    def collate(indices):
        images, targets = [], []
        for index in indices:
            frame = frames[index]
            with Image.open(image_path(manifest, frame)) as source:
                image = source.convert("RGB")
            if image.size != (frame["width"], frame["height"]):
                raise ValueError(f"Image dimensions disagree with manifest: {frame['image_id']}")
            images.append(image)
            annotations = []
            for ann in frame["annotations"]:
                x1, y1, x2, y2 = ann["bbox"]
                annotations.append(
                    {
                        "category_id": LABELS.index(ann["label"]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
            targets.append({"image_id": index, "annotations": annotations})
        return processor(images=images, annotations=targets, return_tensors="pt")

    loader = torch.utils.data.DataLoader(
        list(range(len(frames))),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    log, step = [], 0
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = [
                {k: v.to(device) if hasattr(v, "to") else v for k, v in label.items()}
                for label in batch["labels"]
            ]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError("Non-finite loss; no checkpoint saved")
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            record = {"epoch": epoch + 1, "step": step, "loss": float(outputs.loss.detach().cpu())}
            log.append(record)
            print(json.dumps(record), flush=True)
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(root)
    processor.save_pretrained(root)
    write_jsonl(root / "training_log.jsonl", log)
    config = {
        "base_checkpoint": str(checkpoint),
        "seed": seed,
        "device": device,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "steps_completed": step,
        "training_frames": len(frames),
        "validation": "Not performed here. Run predict/evaluate on a disjoint validation split.",
    }
    (root / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    return config
