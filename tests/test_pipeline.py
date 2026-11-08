import json

import pytest

from complaints import models, pipeline, readme

README = "# x\n\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n"


def test_pipeline_end_to_end_then_cached(cfg, monkeypatch):
    out = pipeline.run(cfg)
    site = cfg.site_dir
    page = (site / "index.html").read_text()
    assert "SYNTHETIC TEST DATA" in page and "Try it: route a complaint" in page
    results = json.loads((site / "results.json").read_text())  # strict JSON: NaN would fail here
    assert results["synthetic"] is True and set(results["scores"]) == {
        "Majority class",
        "TF-IDF + logistic regression",
        "Sentence embeddings + logistic regression",
        pipeline.FT_NAME,
        pipeline.ENS_NAME,
    }
    assert (
        results["run"]["sparse_months"] == {"2026-08": 2}
        and results["run"]["window"]["test"][-1] == "2026-07"
    )
    demo = json.loads((site / "demo.json").read_text())
    assert demo["labels"] == out["run"]["labels"] and len(demo["examples"]) == 6
    assert (site / "models" / demo["model"] / "onnx" / "model_quantized.onnx").exists()
    assert out["scores"]["TF-IDF + logistic regression"]["macro_f1"] > 0.6
    r = out["scores"][pipeline.FT_NAME]["routing"]
    assert 0 <= r["auto_share"] <= 1
    # a second run reuses the sample, embeddings, fine-tuned model and topics
    monkeypatch.setattr(models, "finetune", lambda *a, **k: pytest.fail("fine-tuned again"))
    monkeypatch.setattr(pipeline.data, "build_samples", lambda *a, **k: pytest.fail("sampled again"))
    monkeypatch.setattr(pipeline.topics, "fit", lambda *a, **k: pytest.fail("topics again"))
    again = pipeline.run(cfg)
    assert again["run"]["fingerprint"] == out["run"]["fingerprint"]


def test_quality_gate_blocks_publishing(cfg):
    cfg.min_macro_f1 = 1.01
    with pytest.raises(pipeline.QualityGateError, match="below the floor"):
        pipeline.run(cfg)
    assert not (cfg.site_dir / "index.html").exists()


def test_gate_catches_regression_and_parity(cfg):
    cfg.cache_dir.mkdir(parents=True)
    (cfg.cache_dir / "baseline.json").write_text(json.dumps({"macro_f1": 0.80}))
    scores = {pipeline.FT_NAME: {"macro_f1": 0.70, "accuracy": 0.9}}
    problems = pipeline.gate(cfg, scores, 0.5, {"top1_agreement": 0.5, "decision_agreement": 0.5})
    assert any("fell from 0.800" in p for p in problems) and any("browser model" in p for p in problems)


def test_readme_and_model_card(cfg):
    out = pipeline.run(cfg)
    cfg.readme.write_text(README)
    import numpy as np

    from complaints.evaluate import per_class

    labels = out["run"]["labels"]
    pc = per_class(np.eye(len(labels)), np.arange(len(labels)), labels)
    dups = {"share_in_train": 0.1, "unseen": {"macro_f1": 0.5}}
    readme.update(cfg, out["run"], out["scores"], out["topics"], dups)
    text = cfg.readme.read_text()
    assert (
        "old" not in text
        and "Routing policy" in text
        and "dvskaff.github.io/complaint-routing-nlp/#demo" in text
    )
    readme.model_card(cfg, out["run"], out["scores"], pc, dups)
    card = cfg.model_card.read_text()
    assert "Model card" in card and "Intended use" in card
