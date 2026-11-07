import numpy as np
import pandas as pd
import pytest
from scipy.special import log_softmax

from complaints import data, evaluate, models


def _split(df, cfg):
    labels = data.label_set(df, cfg)
    lid = {k: i for i, k in enumerate(labels)}
    ok = df["product"].isin(labels)
    parts = [df[df[f] & ok].reset_index(drop=True) for f in ("in_train", "in_val", "in_test")]
    return labels, parts, [p["product"].map(lid).to_numpy() for p in parts]


def test_classic_models_beat_majority(cfg, samples):
    df, _, _ = samples
    labels, (tr, va, te), (y, yv, yt) = _split(df, cfg)
    maj = models.majority(yv, len(te), len(labels))
    tf = models.tfidf_lr(tr["text"].tolist(), y, va["text"].tolist(), yv, te["text"].tolist())
    e = models.HashEmbedder()
    emb = models.embed_lr(e.encode(tr["text"].tolist()), y, e.encode(va["text"].tolist()), yv,
                          e.encode(te["text"].tolist()), 0.0)  # fmt: skip
    assert tf.test.shape == (len(te), len(labels)) and emb.val.shape == (len(va), len(labels))
    m = {
        o.name: evaluate.metrics(np.exp(log_softmax(o.test, axis=1)), yt)["macro_f1"] for o in (maj, tf, emb)
    }
    assert m["TF-IDF + logistic regression"] > m["Majority class"] + 0.3
    assert m["Sentence embeddings + logistic regression"] > m["Majority class"] + 0.2


def test_calibration_recovers_temperature_and_prior():
    rng = np.random.default_rng(0)
    n, k = 4000, 3
    y_val = rng.choice(k, size=n, p=[0.7, 0.2, 0.1])
    true = rng.normal(size=(n, k)) + 3.0 * np.eye(k)[y_val]
    # For this data the ideal balanced-prior logits are 3 * true (unit-variance Gaussians 3 apart). A model trained on
    # balanced data that is 2x overconfident outputs 6 * true: calibration should find T ~= 2 and the real prior.
    logits = 6.0 * true
    y_train = np.repeat(np.arange(k), 500)
    cal = models.fit_calibration(logits, y_val, y_train, k)
    assert 1.5 < cal.temperature < 2.6
    assert cal.bias[0] > cal.bias[1] > cal.bias[2]  # common products get a boost, rare ones a penalty
    p = np.exp(cal.apply(logits))
    assert np.allclose(p.sum(1), 1) and evaluate.ece(p, y_val) < evaluate.ece(
        np.exp(log_softmax(logits, 1)), y_val
    )


def test_finetune_learns_and_is_reproducible(cfg, samples, tmp_path):
    df, _, _ = samples
    labels, (tr, va, te), (y, yv, yt) = _split(df, cfg)
    info = models.finetune(cfg, tr["text"].tolist(), y, va["text"].tolist(), yv, labels, tmp_path / "ft")
    assert info["history"][-1]["train_loss"] < info["history"][0]["train_loss"]
    tok, m = models.load_finetuned(tmp_path / "ft")
    assert m.config.id2label[0] == labels[0]
    a = models.predict_logits(m, tok, te["text"].tolist(), cfg.max_tokens)
    # batching sorts by length; outputs must come back in input order
    b = models.predict_logits(m, tok, te["text"].tolist()[::-1], cfg.max_tokens)[::-1]
    assert np.allclose(a, b, atol=1e-4)
    models.finetune(cfg, tr["text"].tolist(), y, va["text"].tolist(), yv, labels, tmp_path / "ft2")
    tok2, m2 = models.load_finetuned(tmp_path / "ft2")
    assert np.allclose(a, models.predict_logits(m2, tok2, te["text"].tolist(), cfg.max_tokens), atol=1e-4)


def test_threshold_policy():
    # confidences 0.9..0.5; the 4th most confident is wrong
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.65, 0.35], [0.6, 0.4], [0.5, 0.5]])
    y = np.array([0, 0, 0, 1, 0, 0])
    assert evaluate.choose_threshold(probs, y, 1.0) == pytest.approx(0.7)
    assert evaluate.choose_threshold(probs, y, 0.8) == pytest.approx(0.5)  # 5/6 >= 0.8 over the whole set
    assert evaluate.choose_threshold(probs, 1 - y, 0.99) == 1.01  # nothing is accurate enough: route nothing
    r = evaluate.routing(probs, y, 0.7)
    assert r["auto_share"] == pytest.approx(0.5) and r["auto_accuracy"] == 1.0
    assert r["review_top2"] == 1.0  # with two classes the right one is always in the top two
    curve = evaluate.coverage_curve(probs, y, points=6)
    assert curve["accuracy"].iloc[-1] == pytest.approx(5 / 6) and curve["auto_share"].is_monotonic_increasing


def test_metrics_and_confusions():
    probs = np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.6, 0.3, 0.1], [0.2, 0.1, 0.7]])
    y = np.array([0, 1, 1, 2])
    m = evaluate.metrics(probs, y)
    assert m["accuracy"] == 0.75 and m["top2_accuracy"] == 1.0 and m["n"] == 4
    cm = evaluate.confusion(probs, y, ["a", "b", "c"])
    top = evaluate.top_confusions(cm)
    assert top[0] == {"actual": "b", "predicted": "a", "n": 1, "share_of_actual": 0.5}
    pc = evaluate.per_class(probs, y, ["a", "b", "c"])
    assert list(pc["support"]) == [1, 2, 1]


def test_duplicates_share():
    train = pd.DataFrame({"dedupe": ["t1", "t2"]})
    test = pd.DataFrame({"dedupe": ["t1", "x", "y", "y"]})
    probs = np.eye(2)[[0, 1, 0, 1]]
    out = evaluate.duplicates(test, train, probs, np.array([0, 1, 0, 0]))
    assert out["share_in_train"] == 0.25 and out["share_repeated_in_test"] == 0.5
    assert out["unseen"]["n"] == 3
