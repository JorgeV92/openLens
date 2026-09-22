from __future__ import annotations

import math

from .data import LABELS


def iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0

def match_frame(annotations, detections, threshold=0.5, iou_threshold=0.5):
    used, matches, false_positives = set(), [], []
    order = sorted(range(len(detections)), key=lambda i: (-detections[i]["score"], i))
    for index in order:
        det = detections[index]
        if det["score"] < threshold:
            continue
        candidates = [
            i for i, ann in enumerate(annotations) if i not in used and ann["label"] == det["label"]
        ]
        best = max(candidates, key=lambda i: iou(det["bbox"], annotations[i]["bbox"]), default=None)
        if best is not None and iou(det["bbox"], annotations[best]["bbox"]) >= iou_threshold:
            used.add(best)
            matches.append((index, best))
        else:
            false_positives.append(index)
    return matches, false_positives, [i for i in range(len(annotations)) if i not in used]

def ap50(frames, predictions, label):
    ground_truth = {
        f["image_id"]: [a for a in f["annotations"] if a["label"] == label] for f in frames
    }
    total = sum(map(len, ground_truth.values()))
    if total == 0:
        return None
    ranked = []
    for frame in frames:
        key = frame["image_id"]
        for index, det in enumerate(predictions[key]):
            if det["label"] == label:
                ranked.append((key, index, det))
    ranked.sort(key=lambda p: (-p[2]["score"], p[0], p[1]))
    used = {key: set() for key in ground_truth}
    tp, precision, recall = 0, [], []
    for count, (key, _, det) in enumerate(ranked, 1):
        candidates = [i for i in range(len(ground_truth[key])) if i not in used[key]]
        best = max(
            candidates, key=lambda i: iou(det["bbox"], ground_truth[key][i]["bbox"]), default=None
        )
        if best is not None and iou(det["bbox"], ground_truth[key][best]["bbox"]) >= 0.5:
            tp += 1
            used[key].add(best)
        precision.append(tp / count)
        recall.append(tp / total)
    return (sum(max((p for p, r in zip(precision, recall) if r >= t / 100), default=0.0) for t in range(101))/ 101)


def summarize(frames, predictions, threshold=0.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Score threshold must be between 0 and 1")
    tp = fp = fn = 0
    for frame in frames:
        matched, extra, missed = match_frame(
            frame["annotations"], predictions[frame["image_id"]], threshold
        )
        tp += len(matched)
        fp += len(extra)
        fn += len(missed)
    ap = {label: ap50(frames, predictions, label) for label in LABELS}
    defined = [value for value in ap.values() if value is not None]
    return {
        "frames": len(frames),
        "ground_truth_objects": tp + fn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positives_per_image": fp / len(frames) if frames else None,
        "ap50_by_class": ap,
        "mean_ap50": sum(defined) / len(defined) if defined else None,
    }


def evaluate(frames, predictions, threshold=0.5):
    slices = {}
    failures = []
    for frame in frames:
        for key, value in frame.get("tags", {}).items():
            slices.setdefault(f"{key}={value}", []).append(frame)
        matched, extra, missed = match_frame(
            frame["annotations"], predictions[frame["image_id"]], threshold
        )
        if extra or missed:
            failures.append(
                {
                    "image_id": frame["image_id"],
                    "false_positives": len(extra),
                    "false_negatives": len(missed),
                    "tags": frame.get("tags", {}),
                }
            )
    return {
        "metric": "openLens diagnostic AP50 (101 recall points); not official Waymo/COCO mAP",
        "score_threshold_for_counts": threshold,
        "iou_threshold": 0.5,
        "ap_uses_all_supplied_detections": True,
        "overall": summarize(frames, predictions, threshold),
        "slices": {
            key: summarize(value, predictions, threshold) for key, value in sorted(slices.items())
        },
        "failures": sorted(failures, key=lambda f: -(f["false_positives"] + f["false_negatives"])),
    }