"""End to end: archive -> samples -> four routing models -> calibration -> routing policy -> emerging issues ->
browser model -> quality gate -> dashboard, README results and model card.

Heavy steps are cached in data/cache (saved between GitHub Actions runs as a release asset), keyed by what they
depend on: the sample, the embeddings, the fine-tuned model and the topics are only recomputed when their inputs
change. The archive is fixed, so most runs reuse everything and only re-check and republish.

    python -m complaints.pipeline             # real data
    python -m complaints.pipeline --fixture   # synthetic data and stand-in models (seconds)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from . import data, evaluate, export, models, topics
from .config import Config

log = logging.getLogger("complaints")
FT_NAME = "Fine-tuned MiniLM transformer"
TF_NAME = "TF-IDF + logistic regression"
ENS_NAME = "Ensemble: fine-tuned + TF-IDF"
SAMPLE_VERSION = 2  # bump when sampling or text cleaning changes, so cached samples are rebuilt


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class QualityGateError(RuntimeError):
    pass


# --------------------------------------------------------------------------- cached steps
def samples_step(cfg: Config, archives) -> tuple[pd.DataFrame, dict]:
    manifest = {"version": SAMPLE_VERSION, "files": [a.name for a in data.candidate_files(archives, cfg)],
                "sampling": [cfg.train_months, cfg.test_months, cfg.topic_months, cfg.train_cap_per_product,
                             cfg.val_size, cfg.test_size_per_month, cfg.topic_size_per_month, cfg.max_chars, cfg.seed]}  # fmt: skip
    path, meta = cfg.cache_dir / "samples.parquet", cfg.cache_dir / "samples.json"
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get("manifest") == manifest:
            log.info("Samples: reusing cache (%s)", ", ".join(saved["manifest"]["files"][-2:]))
            return pd.read_parquet(path), saved["stats"]
    df, stats = data.build_samples(cfg, archives)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta.write_text(json.dumps({"manifest": manifest, "stats": stats}, indent=2))
    return df, stats


def embeddings_step(cfg: Config, df: pd.DataFrame, fp: str) -> tuple[np.ndarray, float]:
    key = _key(fp, cfg.embedder if cfg.backend != "fake" else "hash", cfg.embed_max_tokens)
    path, meta = cfg.cache_dir / "embeddings.npy", cfg.cache_dir / "embeddings.json"
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("key") == key:
        log.info("Embeddings: reusing cache")
        return np.load(path), json.loads(meta.read_text())["ms_per_complaint"]
    embedder = models.load_embedder(cfg)
    t0 = time.time()
    emb = embedder.encode(df["text"].tolist())
    ms = (time.time() - t0) / len(df) * 1000
    log.info("Embeddings: %s complaints in %.0fs", f"{len(df):,}", time.time() - t0)
    np.save(path, emb)
    meta.write_text(json.dumps({"key": key, "model": embedder.name, "ms_per_complaint": ms}))
    return emb, ms


def finetune_step(cfg: Config, train, y, val, y_val, test, labels) -> tuple[models.Output, Path, dict]:
    key = _key(train["complaint_id"].tolist(), val["complaint_id"].tolist(), labels, cfg.base_model, cfg.backend,
               cfg.max_tokens, cfg.epochs, cfg.batch_size, cfg.learning_rate, cfg.weight_decay, cfg.warmup_share,
               cfg.seed)  # fmt: skip
    root = cfg.cache_dir / "finetuned"
    out, meta = root / "model", root / "info.json"
    logits = root / "logits.npz"
    info = json.loads(meta.read_text()) if meta.exists() else {}
    if info.get("key") == key and logits.exists() and (out / "config.json").exists():
        log.info("Fine-tuned model: reusing cache (best epoch %s)", info["best_epoch"])
    else:
        log.info("Fine-tuning %s on %s complaints (%d epochs)", cfg.base_model, f"{len(train):,}", cfg.epochs)
        info = {"key": key, **models.finetune(cfg, train["text"].tolist(), y, val["text"].tolist(), y_val, labels,
                                              out)}  # fmt: skip
        tok, model = models.load_finetuned(out)
        lv = models.predict_logits(model, tok, val["text"].tolist(), cfg.max_tokens)
        t0 = time.time()
        lt = models.predict_logits(model, tok, test["text"].tolist(), cfg.max_tokens)
        info["ms_per_complaint"] = (time.time() - t0) / max(len(test), 1) * 1000
        np.savez(logits, val=lv, test=lt)
        meta.write_text(json.dumps(info, indent=2))
    z = np.load(logits)
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e6
    o = models.Output(FT_NAME, z["val"], z["test"], info["seconds"], info["ms_per_complaint"], size,
                      {k: info[k] for k in ("history", "best_epoch", "train_size")})  # fmt: skip
    return o, out, info


def topics_step(cfg: Config, docs: pd.DataFrame, emb: np.ndarray, emb_key: str) -> np.ndarray:
    key = _key(docs["complaint_id"].tolist(), emb_key, cfg.topic_min_cluster, cfg.seed)
    path, meta = cfg.cache_dir / "topics.npy", cfg.cache_dir / "topics.json"
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("key") == key:
        log.info("Topics: reusing cache")
        return np.load(path)
    t0 = time.time()
    tp = topics.fit(cfg, docs, emb)
    log.info("Topics: %d topics from %s complaints in %.0fs", len(set(tp.tolist()) - {-1}), f"{len(docs):,}",
             time.time() - t0)  # fmt: skip
    np.save(path, tp)
    meta.write_text(json.dumps({"key": key}))
    return tp


def mix_by_month(df: pd.DataFrame, w, top: int = 5) -> dict:
    """Share of each of the largest products among the natural monthly sample."""
    nat = df[df["in_topics"]]
    tab = pd.crosstab(nat["month"], nat["product"], normalize="index").reindex(w.topics).fillna(0.0)
    keep = tab.mean().sort_values(ascending=False).index[:top]
    return {"months": list(tab.index), "shares": {p: [float(x) for x in tab[p]] for p in keep}}


# --------------------------------------------------------------------------- quality gate
def gate(cfg: Config, scores: dict, majority_acc: float, onnx_check: dict) -> list[str]:
    ft = scores[FT_NAME]
    problems = []
    if ft["macro_f1"] < cfg.min_macro_f1:
        problems.append(f"fine-tuned macro-F1 {ft['macro_f1']:.3f} is below the floor {cfg.min_macro_f1}")
    if ft["accuracy"] < majority_acc and not cfg.fixture:
        problems.append(
            f"fine-tuned accuracy {ft['accuracy']:.3f} is below always-majority {majority_acc:.3f}"
        )
    if onnx_check["top1_agreement"] < cfg.min_onnx_agreement:
        problems.append(
            f"browser model agrees with PyTorch on only {onnx_check['top1_agreement']:.1%} of complaints"
        )
    base = cfg.cache_dir / "baseline.json"
    if base.exists():
        prev = json.loads(base.read_text()).get("macro_f1")
        if prev is not None and ft["macro_f1"] < prev - cfg.max_macro_f1_drop:
            problems.append(f"fine-tuned macro-F1 fell from {prev:.3f} to {ft['macro_f1']:.3f}")
    return problems


DEMO_EXAMPLES = [  # written for the demo (not real complaints)
    "I disputed a late payment on my credit report three times. The bureau keeps saying it was verified but the "
    "account was paid on time and I have the bank statements.",
    "A collection agency keeps calling me at work about a medical bill I already paid. They never sent a validation "
    "letter and now they are threatening to sue.",
    "I was charged twice for the same purchase on my credit card. The bank reversed one charge, then put it back and "
    "added interest.",
    "Someone called pretending to be my bank's fraud department and told me to send money through the payment app to "
    "a safe account. The bank refuses to refund me.",
    "My mortgage servicer raised my monthly payment because of an escrow shortage, but my taxes and insurance did "
    "not go up. They will not explain the calculation.",
    "The dealership financed my car and the lender added a warranty I never agreed to. Now my loan balance is higher "
    "and they will not cancel it.",
]


# --------------------------------------------------------------------------- run
def run(cfg: Config) -> dict:
    t_start = time.time()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.fixture:
        from .fixture import make_archive

        make_archive(cfg.data_dir / "fixture")
    archives = data.list_archives(cfg)
    df, sample_stats = samples_step(cfg, archives)
    w = data.Window(**sample_stats["window"])
    fp = data.fingerprint(df)
    labels = data.label_set(df, cfg)
    lid = {k: i for i, k in enumerate(labels)}
    routable = df["product"].isin(labels)
    train = df[df["in_train"] & routable].reset_index(drop=True)
    val = df[df["in_val"] & routable].reset_index(drop=True)
    test = df[df["in_test"] & routable].reset_index(drop=True)
    not_routed = df[df["in_test"] & ~routable]["product"].value_counts()
    y, y_val, y_test = (d["product"].map(lid).to_numpy() for d in (train, val, test))
    log.info("Samples: %s training, %s validation, %s test complaints; %d products", f"{len(train):,}",
             f"{len(val):,}", f"{len(test):,}", len(labels))  # fmt: skip

    outputs = [models.majority(y_val, len(test), len(labels))]
    log.info("TF-IDF + logistic regression")
    outputs.append(models.tfidf_lr(train["text"].tolist(), y, val["text"].tolist(), y_val, test["text"].tolist(),
                                   cfg.seed))  # fmt: skip
    emb, ms_embed = embeddings_step(cfg, df, fp)
    emb_key = _key(fp, cfg.embedder, cfg.backend, cfg.embed_max_tokens)
    row = {cid: i for i, cid in enumerate(df["complaint_id"])}
    pick = lambda d: emb[d["complaint_id"].map(row).to_numpy()]  # noqa: E731
    log.info("Sentence embeddings + logistic regression")
    outputs.append(models.embed_lr(pick(train), y, pick(val), y_val, pick(test), ms_embed, cfg.seed))
    ft, ft_dir, ft_info = finetune_step(cfg, train, y, val, y_val, test, labels)
    outputs.append(ft)

    # ---- calibration, metrics and the routing policy, all fitted on validation and reported on test
    cals = {}
    for o in outputs:
        cals[o.name] = (models.Calibration(1.0, np.zeros(len(labels))) if o.name == "Majority class"
                        else models.fit_calibration(o.val, y_val, y, len(labels)))  # fmt: skip
    # Ensemble: average of the calibrated log-probabilities of the fine-tuned model and TF-IDF (re-tempered only)
    tf_o = next(o for o in outputs if o.name == TF_NAME)
    ens = models.Output(ENS_NAME, 0.5 * cals[TF_NAME].apply(tf_o.val) + 0.5 * cals[FT_NAME].apply(ft.val),
                        0.5 * cals[TF_NAME].apply(tf_o.test) + 0.5 * cals[FT_NAME].apply(ft.test),
                        tf_o.seconds_train + ft.seconds_train, tf_o.ms_per_complaint + ft.ms_per_complaint,
                        tf_o.size_mb + ft.size_mb)  # fmt: skip
    outputs.append(ens)
    cals[ENS_NAME] = models.fit_calibration(ens.val, y_val, y, len(labels), prior=False)
    scores, curves, probs = {}, {}, {}
    for o in outputs:
        cal = cals[o.name]
        pv, pt = np.exp(cal.apply(o.val)), np.exp(cal.apply(o.test))
        raw_t = np.exp(log_softmax(o.test, axis=1))
        thr = evaluate.choose_threshold(pv, y_val, cfg.target_accuracy)
        scores[o.name] = {**evaluate.metrics(pt, y_test), "ece_uncalibrated": evaluate.ece(raw_t, y_test),
                          "routing": evaluate.routing(pt, y_test, thr), "seconds_train": o.seconds_train,
                          "ms_per_complaint": o.ms_per_complaint, "size_mb": o.size_mb,
                          "temperature": cal.temperature, **o.info}  # fmt: skip
        curves[o.name] = evaluate.coverage_curve(pt, y_test)
        probs[o.name] = pt
        r = scores[o.name]["routing"]
        log.info("  %-44s accuracy %.3f  macro-F1 %.3f  auto-routes %.0f%% at %.1f%% accuracy", o.name,
                 scores[o.name]["accuracy"], scores[o.name]["macro_f1"], r["auto_share"] * 100,
                 r["auto_accuracy"] * 100 if r["auto_share"] else float("nan"))  # fmt: skip
    p_ft = probs[FT_NAME]
    per_class = evaluate.per_class(p_ft, y_test, labels)
    cm = evaluate.confusion(p_ft, y_test, labels)
    conf, pred = p_ft.max(1), p_ft.argmax(1)
    wrong = np.nonzero(pred != y_test)[0]
    sure_wrong = wrong[np.argsort(-conf[wrong])][:6]
    errors = [{"text": test["text"].iloc[i][:260] + ("…" if len(test["text"].iloc[i]) > 260 else ""),
               "actual": labels[y_test[i]], "predicted": labels[pred[i]], "confidence": float(conf[i])}
              for i in sure_wrong]  # fmt: skip
    dups = evaluate.duplicates(test, train, p_ft, y_test)

    # ---- emerging issues
    docs = df[df["in_topics"]].reset_index(drop=True)
    tp = topics_step(cfg, docs, pick(docs), emb_key)
    topic_summary = topics.summarize(cfg, docs, pick(docs), tp, w.topics)

    # ---- the browser model, then the gate (nothing is published if it fails)
    tok, model = models.load_finetuned(ft_dir)
    model_id = (
        f"complaint-router-{ft_info['key'][:8]}"  # versioned, so browsers never use a stale cached model
    )
    web_dir = cfg.site_dir / "models" / model_id
    cfg.site_dir.mkdir(parents=True, exist_ok=True)
    export.export(model, tok, cals[FT_NAME], web_dir, cfg.max_tokens)
    sample_texts = val["text"].sample(min(400, len(val)), random_state=cfg.seed).tolist()
    onnx_check = export.parity(web_dir, model, tok, cals[FT_NAME], sample_texts, cfg.max_tokens)
    log.info("Browser model: %.1f MB, agrees with PyTorch on %.1f%% of %d complaints", onnx_check["size_mb"],
             onnx_check["top1_agreement"] * 100, onnx_check["n"])  # fmt: skip
    problems = gate(cfg, scores, scores["Majority class"]["accuracy"], onnx_check)
    if problems:
        raise QualityGateError("Quality gate failed: " + "; ".join(problems))
    runtime = export.copy_runtime(cfg.vendor_dir, cfg.site_dir)
    thr = scores[FT_NAME]["routing"]["threshold"]
    export.write_demo(
        cfg.site_dir, model_id, labels, thr, cfg.target_accuracy, [{"text": t} for t in DEMO_EXAMPLES]
    )

    run_info = {
        "window": asdict(w), "labels": labels, "files": sample_stats["files"], "rows_read": sample_stats["rows_read"],
        "narratives_by_month": sample_stats["narratives_by_month"], "train": len(train), "val": len(val),
        "test": len(test), "not_routed": {k: int(v) for k, v in not_routed.items()},
        "test_by_product": {labels[i]: int(n) for i, n in enumerate(np.bincount(y_test, minlength=len(labels)))},
        "train_by_product": {labels[i]: int(n) for i, n in enumerate(np.bincount(y, minlength=len(labels)))},
        "backend": cfg.backend, "base_model": cfg.base_model, "embedder": cfg.embedder, "fingerprint": fp,
        "finetune": {k: ft_info[k] for k in ("history", "best_epoch", "seconds", "train_size")},
        "onnx": onnx_check, "runtime_bundled": runtime, "model_id": model_id,
        "sparse_months": sample_stats.get("sparse_months", {}), "mix_by_month": mix_by_month(df, w), "target_accuracy": cfg.target_accuracy,
        "built_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "seconds": round(time.time() - t_start),
    }  # fmt: skip
    from . import readme, report

    report.build_site(
        cfg, run_info, scores, curves, per_class, cm, errors, dups, topic_summary, DEMO_EXAMPLES
    )
    if not cfg.fixture:
        readme.update(cfg, run_info, scores, topic_summary, dups)
        readme.model_card(cfg, run_info, scores, per_class, dups)
    (cfg.cache_dir / "baseline.json").write_text(json.dumps({"macro_f1": scores[FT_NAME]["macro_f1"]}))
    log.info("Done in %ss", run_info["seconds"])
    return {"run": run_info, "scores": scores, "topics": topic_summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fixture", action="store_true", help="synthetic data and stand-in models")
    args = parser.parse_args()
    if args.fixture:
        os.environ.setdefault("ROUTER_FIXTURE", "1")
        os.environ.setdefault("ROUTER_BACKEND", "fake")
        os.environ.setdefault("ROUTER_DATA_DIR", "/tmp/router-fixture/data")
        os.environ.setdefault("ROUTER_SITE_DIR", "/tmp/router-fixture/site")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    for noisy in ("httpx", "huggingface_hub", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        run(Config())
    except QualityGateError as exc:
        log.error("%s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
