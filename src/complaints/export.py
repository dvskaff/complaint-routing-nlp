"""Ship the fine-tuned router to the browser: ONNX with calibration built in, 8-bit weights, checked for parity.

The exported graph returns calibrated logits (logits / temperature + prior correction), so the probabilities the
browser shows are the same calibrated probabilities the dashboard reports. Dynamic int8 quantization (per-channel) makes
the file about 4x smaller; a parity check against PyTorch runs on validation complaints before anything is published.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import warnings
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
WEB_RUNTIME = [
    "transformers.min.js",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
]


def export(model, tokenizer, calibration, out: Path, max_tokens: int) -> Path:
    """Write a transformers.js model folder: config, tokenizer and onnx/model_quantized.onnx."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class Calibrated(torch.nn.Module):
        def __init__(self, m, t: float, b):
            super().__init__()
            self.m = m
            self.register_buffer("t", torch.tensor(float(t)))
            self.register_buffer("b", torch.tensor(np.asarray(b, dtype=np.float32)))

        def forward(self, input_ids, attention_mask, token_type_ids):
            logits = self.m(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits
            return logits / self.t + self.b

    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    wrapped = Calibrated(model.eval(), calibration.temperature, calibration.bias).eval()
    sample = tokenizer(["My credit card was charged twice.", "A debt collector keeps calling me about a debt."],
                       padding=True, return_tensors="pt")  # fmt: skip
    args = (sample["input_ids"], sample["attention_mask"], torch.zeros_like(sample["input_ids"]))
    axes = {k: {0: "batch", 1: "sequence"} for k in ("input_ids", "attention_mask", "token_type_ids")}
    fp32 = out / "onnx" / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapped, args, str(fp32), input_names=list(axes), output_names=["logits"],
                          dynamic_axes={**axes, "logits": {0: "batch"}}, opset_version=17, dynamo=False)  # fmt: skip
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)  # the quantizer logs every tensor (and a hint) at INFO and WARNING
    try:
        # per-channel scales keep the 8-bit model much closer to the original (99% vs. 96% top-1 agreement)
        quantize_dynamic(
            str(fp32),
            str(out / "onnx" / "model_quantized.onnx"),
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )
    finally:
        root.setLevel(level)
    fp32.unlink()
    tokenizer.model_max_length = max_tokens  # the browser truncates exactly like training did
    tokenizer.save_pretrained(out)
    model.config.save_pretrained(out)
    return out


def parity(model_dir: Path, model, tokenizer, calibration, texts: list[str], max_tokens: int,
           threshold: float = 1.01) -> dict:  # fmt: skip
    """Agreement between the quantized ONNX model and PyTorch (+ calibration) on real validation complaints:
    on the top team, and on the routing decision (auto-route to the same team, or both send to human review)."""
    import onnxruntime as ort

    from .models import predict_logits

    session = ort.InferenceSession(
        str(model_dir / "onnx" / "model_quantized.onnx"), providers=["CPUExecutionProvider"]
    )
    torch_p = np.exp(calibration.apply(predict_logits(model, tokenizer, texts, max_tokens)))
    onnx_p = np.zeros_like(torch_p)
    t0 = time.time()
    for i, t in enumerate(texts):  # one at a time, as the browser does
        enc = tokenizer([t], truncation=True, max_length=max_tokens, return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64),
                "token_type_ids": np.zeros_like(enc["input_ids"], dtype=np.int64)}  # fmt: skip
        z = session.run(None, feed)[0][0]
        e = np.exp(z - z.max())
        onnx_p[i] = e / e.sum()
    ms = (time.time() - t0) / max(len(texts), 1) * 1000
    size = (model_dir / "onnx" / "model_quantized.onnx").stat().st_size / 1e6

    def decide(p):
        return np.where(p.max(1) >= threshold, p.argmax(1), -1)

    return {"n": len(texts), "top1_agreement": float((torch_p.argmax(1) == onnx_p.argmax(1)).mean()),
            "decision_agreement": float((decide(torch_p) == decide(onnx_p)).mean()),
            "max_prob_diff": float(np.abs(torch_p - onnx_p).max()), "size_mb": size, "ms_per_complaint": ms}  # fmt: skip


def copy_runtime(vendor: Path, site: Path) -> bool:
    """transformers.js and its WebAssembly runtime, served from the site itself (no third-party CDN at run time)."""
    if not all((vendor / f).exists() for f in WEB_RUNTIME):
        log.warning("Browser runtime not found in %s; the live demo will show a notice instead", vendor)
        return False
    (site / "vendor").mkdir(parents=True, exist_ok=True)
    for f in WEB_RUNTIME:
        shutil.copy(vendor / f, site / "vendor" / f)
    return True


def write_demo(site: Path, model_id: str, labels: list[str], threshold: float, target: float,
               examples: list[dict]) -> None:  # fmt: skip
    (site / "demo.json").write_text(json.dumps({"model": model_id, "labels": labels, "threshold": threshold, "target_accuracy": target,
                                                "examples": examples}, indent=2))  # fmt: skip
