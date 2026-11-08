import json

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from complaints import data, export, models, topics


def test_summarize_finds_the_growing_topic_within_its_product(cfg):
    months = [f"2026-{m:02d}" for m in range(1, 7)]
    rows, tp = [], []
    for i, m in enumerate(months):
        scam = 30 if i >= 3 else 5  # of 40 checking complaints a month
        credit = 60 if i < 3 else 20  # the product mix shifts too: this must not create fake trends
        spec = [(1, "impersonation scam bank refund", "Checking")] * scam
        spec += [(2, "overdraft fee charged", "Checking")] * (40 - scam)
        spec += [(0, "credit report dispute bureau", "Credit")] * credit + [
            (-1, "misc words here", "Credit")
        ] * 5
        for t, text, product in spec:
            rows.append({"month": m, "text": text, "product": product, "company": "Acme"})
            tp.append(t)
    docs = pd.DataFrame(rows)
    emb = np.random.default_rng(0).normal(size=(len(docs), 8)).astype(np.float32)
    cfg.recent_months, cfg.topic_min_recent_docs = 3, 10
    s = topics.summarize(cfg, docs, emb, np.array(tp), months)
    top = s["emerging"][0]
    assert top["topic"] == 1 and top["top_product"] == "Checking"
    assert top["recent_share"] == pytest.approx(0.75) and top["base_share"] == pytest.approx(0.125)
    assert [f["topic"] for f in s["fading"]] == [
        2
    ]  # the credit topic is 100% of credit complaints throughout
    assert s["topics"] == 3


def test_keywords_do_not_repeat_words():
    docs = pd.DataFrame(
        {"text": ["identity theft identity theft report"] * 5 + ["mortgage escrow payment"] * 5}
    )
    words = topics.keywords(docs, np.array([0] * 5 + [1] * 5))
    flat = [w for x in words[0] for w in x.split()]
    assert len(flat) == len(set(flat))


def test_bertopic_runs_on_fixture(cfg, samples):
    df, w, _ = samples
    docs = df[df["in_topics"]].reset_index(drop=True)
    emb = models.HashEmbedder().encode(docs["text"].tolist())
    tp = topics.fit(cfg, docs, emb)
    assert len(tp) == len(docs) and len(set(tp.tolist()) - {-1}) >= 3
    s = topics.summarize(cfg, docs, emb, tp, w.topics)
    assert s["emerging"], "the synthetic scam theme grows in the last three months"
    assert any("scam" in t["label"] or "impersonation" in t["label"] or "pretending" in t["label"]
               for t in s["emerging"][:3])  # fmt: skip


def test_export_matches_pytorch_and_is_loadable(cfg, samples, tmp_path):
    df, _, _ = samples
    labels = data.label_set(df, cfg)
    lid = {k: i for i, k in enumerate(labels)}
    ok = df["product"].isin(labels)
    tr, va = df[df["in_train"] & ok], df[df["in_val"] & ok]
    y, yv = tr["product"].map(lid).to_numpy(), va["product"].map(lid).to_numpy()
    models.finetune(cfg, tr["text"].tolist(), y, va["text"].tolist(), yv, labels, tmp_path / "ft")
    tok, m = models.load_finetuned(tmp_path / "ft")
    cal = models.Calibration(1.7, np.linspace(-0.5, 0.5, len(labels)))
    out = export.export(m, tok, cal, tmp_path / "web" / "complaint-router", cfg.max_tokens)
    assert (out / "onnx" / "model_quantized.onnx").exists() and not (out / "onnx" / "model.onnx").exists()
    conf = json.loads((out / "config.json").read_text())
    assert conf["id2label"]["0"] == labels[0] and json.loads((out / "tokenizer_config.json").read_text())
    check = export.parity(out, m, tok, cal, va["text"].tolist()[:60], cfg.max_tokens)
    assert check["top1_agreement"] >= 0.9 and check["max_prob_diff"] < 0.1
    # the calibration is inside the graph: a variable-length batch gives calibrated logits directly
    s = ort.InferenceSession(str(out / "onnx" / "model_quantized.onnx"))
    enc = tok(
        va["text"].tolist()[:3], padding=True, truncation=True, max_length=cfg.max_tokens, return_tensors="np"
    )
    z = s.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                     "token_type_ids": np.zeros_like(enc["input_ids"])})[0]  # fmt: skip
    ref = cal.apply(models.predict_logits(m, tok, va["text"].tolist()[:3], cfg.max_tokens))
    assert np.allclose(np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True),
                       np.exp(ref), atol=0.05)  # fmt: skip


def test_runtime_copy(tmp_path):
    vendor, site = tmp_path / "vendor", tmp_path / "site"
    assert export.copy_runtime(vendor, site) is False
    vendor.mkdir()
    for f in export.WEB_RUNTIME:
        (vendor / f).write_text("x")
    assert export.copy_runtime(vendor, site) is True and (site / "vendor" / export.WEB_RUNTIME[0]).exists()
