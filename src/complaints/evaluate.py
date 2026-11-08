"""Metrics, the auto-route vs. human-review policy, and error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: how far stated confidence is from actual accuracy (0 = perfectly calibrated)."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    top2 = np.argsort(-probs, axis=1)[:, :2]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "top2_accuracy": float((top2 == y[:, None]).any(1).mean()),
        "ece": ece(probs, y),
        "n": int(len(y)),
    }


def per_class(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y, probs.argmax(1), labels=range(len(labels)), zero_division=0
    )
    return pd.DataFrame({"product": labels, "precision": p, "recall": r, "f1": f, "support": s})


def confusion(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y, probs.argmax(1), labels=range(len(labels)))
    return pd.DataFrame(cm, index=labels, columns=labels)


def top_confusions(cm: pd.DataFrame, k: int = 6) -> list[dict]:
    rows = []
    for actual in cm.index:
        total = cm.loc[actual].sum()
        for predicted in cm.columns:
            if actual != predicted and cm.loc[actual, predicted] > 0:
                rows.append({"actual": actual, "predicted": predicted, "n": int(cm.loc[actual, predicted]),
                             "share_of_actual": float(cm.loc[actual, predicted] / total)})  # fmt: skip
    return sorted(rows, key=lambda r: -r["n"])[:k]


# --------------------------------------------------------------------------- routing policy
def choose_threshold(probs: np.ndarray, y: np.ndarray, target: float) -> float:
    """Lowest confidence threshold at which the auto-routed complaints are still `target` accurate (validation).

    Complaints are sorted by confidence; the threshold is the confidence of the last complaint in the longest
    most-confident prefix whose accuracy is at least the target. Returns 1.01 (route nothing) if none qualifies.
    """
    conf = probs.max(1)
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    correct = (probs.argmax(1) == y)[order]
    acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    group_end = np.r_[c[1:] < c[:-1], True]  # a threshold routes every complaint tied at that confidence
    ok = np.nonzero((acc >= target) & group_end)[0]
    if len(ok) == 0:
        return 1.01
    return float(c[ok[-1]])


def routing(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    conf, correct = probs.max(1), probs.argmax(1) == y
    auto = conf >= threshold
    return {
        "threshold": float(threshold),
        "auto_share": float(auto.mean()),
        "auto_accuracy": float(correct[auto].mean()) if auto.any() else float("nan"),
        "review_share": float(1 - auto.mean()),
        "review_top2": float((np.argsort(-probs[~auto], axis=1)[:, :2] == y[~auto, None]).any(1).mean())
        if (~auto).any()
        else float("nan"),
    }


def coverage_curve(probs: np.ndarray, y: np.ndarray, points: int = 41) -> pd.DataFrame:
    """Accuracy of the auto-routed share, for every share from 5% to 100% (most confident first)."""
    conf = probs.max(1)
    order = np.argsort(-conf, kind="stable")
    correct = (probs.argmax(1) == y)[order]
    cum = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    shares = np.linspace(0.05, 1.0, points)
    idx = np.clip((shares * len(y)).astype(int) - 1, 0, len(y) - 1)
    return pd.DataFrame({"auto_share": shares, "accuracy": cum[idx]})


# --------------------------------------------------------------------------- duplicates (template letters)
def duplicates(test: pd.DataFrame, train: pd.DataFrame, probs: np.ndarray, y: np.ndarray) -> dict:
    """Template letters repeat across months. How many test narratives also appear (word for word) in training,
    and how well does the model do on the ones that don't?"""
    seen = test["dedupe"].isin(set(train["dedupe"])).to_numpy()
    dup_within = test["dedupe"].duplicated(keep=False).to_numpy()
    out = {"share_in_train": float(seen.mean()), "share_repeated_in_test": float(dup_within.mean())}
    if (~seen).any():
        out["unseen"] = metrics(probs[~seen], y[~seen])
    return out
