"""Raw tensor parity isolates model export from preprocessing and label mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .models import DEFAULT_CHECKPOINT, ml_imports


def compare_arrays(reference, actual, atol=1e-4, rtol=1e-4):
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError("Tolerances must be finite and nonnegative")
    if reference.shape != actual.shape:
        return {
            "passed": False,
            "reason": "shape mismatch",
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        return {"passed": False, "reason": "non-finite tensor"}
    difference = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {
        "passed": bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
    }


def save_bundle(root, input_tensor, logits, boxes, extra=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {"input": input_tensor, "reference_logits": logits, "reference_boxes": boxes}
    for name, tensor in arrays.items():
        np.asarray(tensor, dtype="<f4").tofile(root / f"{name}.f32")
    metadata = {
        "format": "openlens-tensors-v1",
        "dtype": "little-endian float32",
        "input_shape": list(input_tensor.shape),
        "logits_shape": list(logits.shape),
        "boxes_shape": list(boxes.shape),
        **(extra or {}),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parity(bundle, actual_prefix, atol=1e-4, rtol=1e-4):
    root = Path(bundle)
    metadata = json.loads((root / "metadata.json").read_text())
    result = {}
    for name in ("logits", "boxes"):
        shape = metadata[f"{name}_shape"]
        reference = np.fromfile(root / f"reference_{name}.f32", dtype="<f4").reshape(shape)
        actual_path = Path(f"{actual_prefix}_{name}.f32")
        actual = np.fromfile(actual_path, dtype="<f4")
        if actual.size == reference.size:
            actual = actual.reshape(shape)
        result[name] = compare_arrays(reference, actual, atol, rtol)
    return {
        "passed": all(v["passed"] for v in result.values()),
        "atol": atol,
        "rtol": rtol,
        "outputs": result,
    }


def onnx_predict(bundle, output_prefix):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install deployment dependencies: pip install -e '.[deploy]'") from exc
    root = Path(bundle)
    metadata = json.loads((root / "metadata.json").read_text())
    values = np.fromfile(root / "input.f32", dtype="<f4").reshape(metadata["input_shape"])
    session = ort.InferenceSession(str(root / "model.onnx"), providers=["CPUExecutionProvider"])
    logits, boxes = session.run(["logits", "pred_boxes"], {"pixel_values": values})
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    np.asarray(logits, dtype="<f4").tofile(f"{output_prefix}_logits.f32")
    np.asarray(boxes, dtype="<f4").tofile(f"{output_prefix}_boxes.f32")


def export_model(image, output, checkpoint=DEFAULT_CHECKPOINT):
    torch, Model, Processor = ml_imports()
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install deployment dependencies: pip install -e '.[deploy]'") from exc
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Export directory must be empty")
    processor = Processor.from_pretrained(checkpoint)
    model = Model.from_pretrained(checkpoint).cpu().eval()
    model.config.disable_custom_kernels = True
    with Image.open(image) as source:
        picture = source.convert("RGB")
    inputs = processor(images=picture, return_tensors="pt")["pixel_values"]

    class Wrapper(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, pixel_values):
            output = self.detector(pixel_values=pixel_values)
            return output.logits, output.pred_boxes

    wrapped = Wrapper(model).eval()
    with torch.inference_mode():
        logits, boxes = wrapped(inputs)
        metadata = save_bundle(
            root,
            inputs.numpy(),
            logits.numpy(),
            boxes.numpy(),
            {
                "checkpoint": str(checkpoint),
                "image_size": list(picture.size),
                "id2label": model.config.id2label,
                "scope": "Fixed batch=1 and processor-selected input size; raw output parity",
            },
        )
        torch.onnx.export(
            wrapped,
            (inputs,),
            str(root / "model.onnx"),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(str(root / "model.onnx"))
    processor.save_pretrained(root / "processor")
    onnx_predict(root, root / "onnx")
    comparison = parity(root, root / "onnx", atol=1e-3, rtol=1e-3)
    (root / "export_parity.json").write_text(json.dumps(comparison, indent=2) + "\n")
    if not comparison["passed"]:
        raise RuntimeError("ONNX export did not pass raw tensor parity; inspect export_parity.json")
    return metadata


def create_onnx_fixture(output):
    """A tiny input-dependent ONNX graph to test the actual C++ runtime without weights."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    root = Path(output)
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2) / 12
    logits = values.reshape(1, 4, 3)
    boxes = np.array(
        [[[0.5, 0.5, 0.2, 0.4], [0.3, 0.4, 0.1, 0.1], [0.7, 0.6, 0.2, 0.3], [0.2, 0.2, 0.1, 0.2]]],
        dtype=np.float32,
    )
    save_bundle(
        root, values, logits, boxes, {"scope": "synthetic runtime fixture; no learned model"}
    )
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["pixel_values", "new_shape"], ["logits"]),
            helper.make_node("Identity", ["constant_boxes"], ["pred_boxes"]),
        ],
        "openLens runtime fixture",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 2, 2])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, 3]),
            helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, [1, 4, 4]),
        ],
        [
            numpy_helper.from_array(np.array([1, 4, 3], dtype=np.int64), "new_shape"),
            numpy_helper.from_array(boxes, "constant_boxes"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, root / "model.onnx")
    return {"bundle": str(root), "scope": "synthetic runtime fixture"}
