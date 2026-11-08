"""The routing models, from simple to transformer, plus calibration.

1. Majority class: always route to the most common product (the floor any model must beat).
2. TF-IDF + logistic regression: words and word pairs; a strong, fast classic baseline.
3. Sentence embeddings + logistic regression: a frozen MiniLM sentence encoder, a linear layer on top.
4. Fine-tuned transformer: the same MiniLM encoder with a classification head, trained end to end.

Every model outputs logits for the validation and test sets. Calibration then corrects them for the real product
mix (training is class-balanced, the real stream is not) and for over- or under-confidence (temperature scaling),
both fitted on the validation month only.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


@dataclass
class Output:
    name: str
    val: np.ndarray  # logits, shape (n_val, n_labels)
    test: np.ndarray
    seconds_train: float = 0.0
    ms_per_complaint: float = float("nan")  # CPU inference time on the test set
    size_mb: float = float("nan")
    info: dict = field(default_factory=dict)


def _macro(y, logits) -> float:
    return f1_score(y, logits.argmax(1), average="macro")


# --------------------------------------------------------------------------- 1. majority class
def majority(y_val: np.ndarray, n_test: int, n_labels: int) -> Output:
    """Always predicts the validation month's most common product (with the validation mix as probabilities)."""
    prior = np.bincount(y_val, minlength=n_labels) + 1.0
    lp = np.log(prior / prior.sum())
    return Output("Majority class", np.tile(lp, (len(y_val), 1)), np.tile(lp, (n_test, 1)), ms_per_complaint=0.0,
                  size_mb=0.0)  # fmt: skip


# --------------------------------------------------------------------------- 2. TF-IDF + logistic regression
def _pick_c(fit, grid, y_val) -> tuple[float, object, np.ndarray]:
    best = None
    for c in grid:
        model, val = fit(c)
        score = _macro(y_val, val)
        log.info("    C=%s: validation macro-F1 %.3f", c, score)
        if best is None or score > best[0]:
            best = (score, c, model, val)
    return best[1], best[2], best[3]


def tfidf_lr(train: list[str], y: np.ndarray, val: list[str], y_val: np.ndarray, test: list[str],
             seed: int = 0) -> Output:  # fmt: skip
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=300_000,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9']+\b", strip_accents="unicode")  # fmt: skip
    xtr, xva = vec.fit_transform(train), vec.transform(val)

    def fit(c):
        m = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(xtr, y)
        return m, np.log(np.clip(m.predict_proba(xva), 1e-12, 1))

    c, model, val_logits = _pick_c(fit, [1.0, 4.0, 16.0], y_val)
    t1 = time.time()
    test_logits = np.log(np.clip(model.predict_proba(vec.transform(test)), 1e-12, 1))
    ms = (time.time() - t1) / max(len(test), 1) * 1000
    size = (model.coef_.size * 8 + len(vec.vocabulary_) * 24) / 1e6
    return Output("TF-IDF + logistic regression", val_logits, test_logits, t1 - t0, ms, size,
                  {"C": c, "features": len(vec.vocabulary_)})  # fmt: skip


# --------------------------------------------------------------------------- 3. sentence embeddings + LR
class HashEmbedder:
    """Offline stand-in for tests: hashed bag of words, L2-normalized."""

    name = "hash-embedder (test stand-in)"

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.strip(".,").encode()) % self.dim] += 1.0  # stable across runs
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class SentenceEmbedder:
    def __init__(self, model: str, max_tokens: int):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model, device="cpu")
        self.model.max_seq_length = max_tokens

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True).astype(np.float32)  # fmt: skip


def load_embedder(cfg):
    if cfg.backend == "fake":
        return HashEmbedder()
    return SentenceEmbedder(cfg.embedder, cfg.embed_max_tokens)


def embed_lr(e_train, y, e_val, y_val, e_test, ms_embed: float, seed: int = 0) -> Output:
    t0 = time.time()

    def fit(c):
        m = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(e_train, y)
        return m, np.log(np.clip(m.predict_proba(e_val), 1e-12, 1))

    c, model, val_logits = _pick_c(fit, [0.5, 2.0, 8.0], y_val)
    t1 = time.time()
    test_logits = np.log(np.clip(model.predict_proba(e_test), 1e-12, 1))
    ms = ms_embed + (time.time() - t1) / max(len(e_test), 1) * 1000
    return Output("Sentence embeddings + logistic regression", val_logits, test_logits, t1 - t0, ms,
                  info={"C": c})  # fmt: skip


# --------------------------------------------------------------------------- 4. fine-tuned transformer
def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tiny_base(train_texts: list[str], labels: list[str], max_tokens: int, out: Path):
    """Test stand-in for the pretrained model: a 2-layer BERT and a WordPiece vocabulary learned from the texts.
    Built once per folder, like a downloaded checkpoint, so repeated fine-tuning starts from the same weights."""
    if (out / "config.json").exists():
        return str(out)
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tok.train_from_iterator(train_texts, trainers.WordPieceTrainer(vocab_size=600, special_tokens=specials))
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls), ("[SEP]", sep)],
    )
    tok.decoder = decoders.WordPiece()
    tokenizer = BertTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]", model_max_length=max_tokens)  # fmt: skip
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=48, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=96, max_position_embeddings=max(64, max_tokens + 2),
                        num_labels=len(labels), id2label=dict(enumerate(labels)),
                        label2id={k: i for i, k in enumerate(labels)})  # fmt: skip
    model = BertForSequenceClassification(config)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out)
    model.save_pretrained(out)
    return str(out)


def load_base(cfg, labels: list[str], train_texts: list[str], work: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = cfg.base_model
    if cfg.backend == "fake":
        source = tiny_base(train_texts, labels, cfg.max_tokens, work / "tiny-base")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={k: i for i, k in enumerate(labels)},
    )
    return tokenizer, model


def _batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool):
    """Batches of similar length (less padding = faster on CPU); batch order is shuffled when training."""
    order = np.argsort(lengths, kind="stable")
    if shuffle:  # sort within random chunks so batches differ between epochs
        idx = rng.permutation(len(lengths))
        chunk = batch_size * 50
        order = np.concatenate([idx[i : i + chunk][np.argsort(lengths[idx[i : i + chunk]], kind="stable")]
                                for i in range(0, len(idx), chunk)])  # fmt: skip
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def encode(tokenizer, texts: list[str], max_tokens: int) -> dict:
    return tokenizer(texts, truncation=True, max_length=max_tokens, padding=False)


def _collate(enc: dict, idx, pad_id: int):
    import torch

    ids = [enc["input_ids"][i] for i in idx]
    n = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), n), dtype=torch.long)
    for j, x in enumerate(ids):
        input_ids[j, : len(x)] = torch.tensor(x)
        mask[j, : len(x)] = 1
    return {"input_ids": input_ids, "attention_mask": mask, "token_type_ids": torch.zeros_like(input_ids)}


def predict_logits(model, tokenizer, texts: list[str], max_tokens: int, batch_size: int = 64) -> np.ndarray:
    import torch

    model.eval()
    enc = encode(tokenizer, texts, max_tokens)
    lengths = np.array([len(x) for x in enc["input_ids"]])
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    with torch.inference_mode():
        for idx in _batches(lengths, batch_size, np.random.default_rng(0), shuffle=False):
            out[idx] = model(**_collate(enc, idx, tokenizer.pad_token_id)).logits.float().numpy()
    return out


def finetune(cfg, train: list[str], y: np.ndarray, val: list[str], y_val: np.ndarray, labels: list[str],
             out: Path) -> dict:  # fmt: skip
    """Fine-tune end to end (AdamW, linear warmup and decay); keeps the epoch with the best validation macro-F1."""
    import torch

    torch.set_num_threads(os.cpu_count() or 1)
    _seed_all(cfg.seed)
    tokenizer, model = load_base(cfg, labels, train, out.parent)
    _seed_all(cfg.seed)  # again: dropout and shuffling start from the same state however the base was loaded
    enc = encode(tokenizer, train, cfg.max_tokens)
    lengths = np.array([len(x) for x in enc["input_ids"]])
    rng = np.random.default_rng(cfg.seed)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total * cfg.warmup_share))
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm", "norm"))]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm", "norm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.learning_rate)  # fmt: skip
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup))
    )
    y_t = torch.tensor(y, dtype=torch.long)
    history, best, step, t0 = [], None, 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for idx in _batches(lengths, cfg.batch_size, rng, shuffle=True):
            batch = _collate(enc, idx, tokenizer.pad_token_id)
            loss = torch.nn.functional.cross_entropy(model(**batch).logits, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                rate = step * cfg.batch_size / (time.time() - t0)
                log.info(
                    "    step %d/%d, loss %.3f, %.1f complaints/s", step, total, np.mean(losses[-50:]), rate
                )
        val_logits = predict_logits(model, tokenizer, val, cfg.max_tokens)
        score = _macro(y_val, val_logits)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": float(score)})
        log.info("  epoch %d: train loss %.3f, validation macro-F1 %.3f", epoch, np.mean(losses), score)
        if best is None or score > best[0]:
            best = (score, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return {"history": history, "best_epoch": best[1], "seconds": time.time() - t0, "train_size": len(train)}


def load_finetuned(path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(
        path
    ).eval()


# --------------------------------------------------------------------------- calibration
@dataclass
class Calibration:
    temperature: float
    bias: np.ndarray  # prior_strength * (log(real product mix) - log(training mix))
    prior_strength: float = 1.0

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated log-probabilities."""
        return log_softmax(logits / self.temperature + self.bias, axis=1)

    def to_dict(self) -> dict:
        return {"temperature": float(self.temperature), "prior_strength": float(self.prior_strength),
                "bias": [float(b) for b in self.bias]}  # fmt: skip


def _temperature(val_logits: np.ndarray, y_val: np.ndarray, bias: np.ndarray) -> float:
    def nll(log_t: float) -> float:
        lp = log_softmax(val_logits / math.exp(log_t) + bias, axis=1)
        return float(-lp[np.arange(len(y_val)), y_val].mean())

    return float(math.exp(minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded").x))


def fit_calibration(val_logits: np.ndarray, y_val: np.ndarray, y_train: np.ndarray, n_labels: int,
                    prior: bool = True, strengths=(0.0, 0.25, 0.5, 0.75, 1.0),
                    tolerance: float = 0.005) -> Calibration:  # fmt: skip
    """Prior correction (training is balanced, the real mix is not) + temperature by validation log-loss.

    The full Bayes prior correction maximizes accuracy but can push a tiny product out of the predictions entirely
    (it is never the most likely team). The correction's strength is therefore chosen on the validation month: the
    strength with the best macro-F1 among those within `tolerance` of the best validation accuracy. Each strength
    gets its own temperature. prior=False fits the temperature only (for inputs that are already prior-corrected).
    """
    prior_train = (np.bincount(y_train, minlength=n_labels) + 1.0) / (len(y_train) + n_labels)
    prior_val = (np.bincount(y_val, minlength=n_labels) + 1.0) / (len(y_val) + n_labels)
    shift = np.log(prior_val) - np.log(prior_train)
    options = []
    for a in strengths if prior else (0.0,):
        bias = a * shift
        cal = Calibration(_temperature(val_logits, y_val, bias), bias, a)
        pred = cal.apply(val_logits).argmax(1)
        options.append((cal, float((pred == y_val).mean()), f1_score(y_val, pred, average="macro")))
    best_acc = max(acc for _, acc, _ in options)
    return max((o for o in options if o[1] >= best_acc - tolerance), key=lambda o: (o[2], o[1]))[0]
