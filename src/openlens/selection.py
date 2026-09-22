import json
import random
from pathlib import Path

from .data import load_frames, load_predictions


def select(manifest, output, budget, method="random", predictions=None, tag=None, seed=42):
    frames = load_frames(manifest)
    if any(f["split"] != "train" for f in frames):
        raise ValueError(
            "Selection accepts a training-only pool; validation/test frames are forbidden"
        )
    if not 1 <= budget <= len(frames):
        raise ValueError("Budget must be between 1 and the number of training frames")
    candidates = sorted(frames, key=lambda f: f["image_id"])
    random.Random(seed).shuffle(candidates)
    if method == "uncertainty":
        if predictions is None:
            raise ValueError("Uncertainty selection needs --predictions")
        decoded = load_predictions(predictions, frames)

        def uncertainty(frame):
            detections = decoded[frame["image_id"]]
            return max((1 - abs(2 * d["score"] - 1) for d in detections), default=1.0)

        candidates.sort(key=uncertainty, reverse=True)
    elif method == "tag":
        if not tag or "=" not in tag:
            raise ValueError("Tag selection needs --tag KEY=VALUE")
        key, value = tag.split("=", 1)
        candidates.sort(key=lambda f: f.get("tags", {}).get(key) == value, reverse=True)
    elif method != "random":
        raise ValueError(f"Unknown selection method: {method}")
    result = {
        "method": method,
        "seed": seed,
        "budget": budget,
        "tag": tag,
        "image_ids": [f["image_id"] for f in candidates[:budget]],
        "note": "Frame-level baseline; nearby frames can be redundant. Tags may be human or VLM proposed.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result
