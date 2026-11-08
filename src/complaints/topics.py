"""Emerging issues: BERTopic on the sentence embeddings of a natural monthly sample of complaints.

BERTopic reduces the embeddings with UMAP, clusters them with HDBSCAN, and describes each cluster by its most
distinctive words (class-based TF-IDF). A topic is "emerging" when its share of complaints in the recent months is
clearly above its share in the earlier months of the window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

log = logging.getLogger(__name__)
DOMAIN_STOP = """xxxx xx account accounts told called call calls received company companies would could said also get
got back time times one two even still since never without make made please help day days month months year years
asked know going need want information consumer complaint however due stated state states regarding sent letter
letters number did does just like really way able tried trying today currently""".split()
STOP = sorted(ENGLISH_STOP_WORDS.union(DOMAIN_STOP))


def fit(cfg, docs: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
    """Topic id per document (-1 = no clear topic)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    small = cfg.fixture
    umap = UMAP(n_neighbors=10 if small else 15, n_components=5, min_dist=0.0, metric="cosine",
                random_state=cfg.seed)  # fmt: skip
    hdb = HDBSCAN(min_cluster_size=cfg.topic_min_cluster, min_samples=3 if small else 10, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=False)  # fmt: skip
    vec = CountVectorizer(stop_words=STOP, min_df=2 if small else 5, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")  # fmt: skip
    model = BERTopic(umap_model=umap, hdbscan_model=hdb, vectorizer_model=vec, calculate_probabilities=False,
                     verbose=False)  # fmt: skip
    topics, _ = model.fit_transform(docs["text"].tolist(), embeddings=emb)
    return np.asarray(topics)


def keywords(docs: pd.DataFrame, topics: np.ndarray, n: int = 6) -> dict[int, list[str]]:
    """Class-based TF-IDF words per topic, recomputed here so cached topic ids are enough to rebuild the report."""
    vec = CountVectorizer(
        stop_words=STOP, min_df=1, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b"
    )
    ids = sorted(t for t in set(topics.tolist()) if t != -1)
    if not ids:
        return {}
    joined = [" ".join(docs["text"].to_numpy()[topics == t]) for t in ids]
    counts = vec.fit_transform(joined).toarray().astype(float)
    tf = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    avg = counts.sum() / len(ids)
    idf = np.log(1 + avg / np.maximum(counts.sum(0), 1))
    score = tf * idf
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for row, t in enumerate(ids):
        words, used = [], set()
        for w in vocab[np.argsort(-score[row])]:
            parts = set(w.split())
            if parts & used:  # skip "theft" after "identity theft": each word appears once in a label
                continue
            words.append(w)
            used |= parts
            if len(words) == n:
                break
        out[t] = words
    return out


def summarize(cfg, docs: pd.DataFrame, emb: np.ndarray, topics: np.ndarray, months: list[str]) -> dict:
    """Monthly topic shares, emerging and fading topics, and the largest topics.

    A topic's share is measured within its main product (e.g. the share of checking-account complaints about an
    impersonation scam), so a change in the overall product mix doesn't make whole groups of topics look like they
    are growing or fading. Months with too few complaints of that product are left out.
    """
    d = docs.assign(topic=topics)
    recent = months[-cfg.recent_months :]
    base = months[: -cfg.recent_months]
    by_product = (
        d.groupby(["product", "month"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
    )
    min_docs = 3 if cfg.fixture else 20
    words = keywords(d, topics)
    rows = []
    for t in sorted(set(topics.tolist()) - {-1}):
        mask = d["topic"].to_numpy() == t
        sub = d[mask]
        prod = sub["product"].value_counts(normalize=True)
        main = prod.index[0]
        in_main = sub[sub["product"] == main].groupby("month").size().reindex(months, fill_value=0)
        denom = by_product.loc[main].astype(float)
        share = (in_main / denom.where(denom >= min_docs)).astype(float)
        rs = float(np.nanmean(share[recent])) if share[recent].notna().any() else np.nan
        bs = float(np.nanmean(share[base])) if base and share[base].notna().any() else np.nan
        if np.isnan(rs) or np.isnan(bs):
            continue
        centroid = emb[mask].mean(0)
        example = sub["text"].iloc[int(np.argmax(emb[mask] @ centroid))]
        comp = sub["company"].replace("", np.nan).dropna().value_counts(normalize=True)
        rows.append({
            "topic": int(t), "label": " · ".join(words.get(t, [])[:4]), "words": words.get(t, []),
            "docs": int(mask.sum()), "recent_docs": int(in_main[recent].sum()),
            "recent_share": rs, "base_share": bs, "change_pts": rs - bs, "ratio": (rs + 1e-4) / (bs + 1e-4),
            "top_product": main, "top_product_share": float(prod.iloc[0]),
            "top_company": comp.index[0] if len(comp) else "", "top_company_share": float(comp.iloc[0]) if len(comp) else 0.0,
            "monthly_share": [None if np.isnan(x) else float(x) for x in share],
            "example": example[:240] + ("…" if len(example) > 240 else ""),
        })  # fmt: skip
    table = pd.DataFrame(rows)
    out = {"months": months, "recent_months": recent, "docs": int(len(d)), "topics": int(len(set(topics.tolist()) - {-1})),
           "outlier_share": float((topics == -1).mean()), "emerging": [], "fading": [], "largest": []}  # fmt: skip
    if table.empty:
        return out
    eligible = table[table["recent_docs"] >= cfg.topic_min_recent_docs]
    grow = eligible[(eligible["change_pts"] > 0) & (eligible["ratio"] >= 1.25)]
    out["emerging"] = grow.sort_values("change_pts", ascending=False).head(6).to_dict("records")
    fall = table[(table["change_pts"] < 0) & (table["ratio"] <= 0.8)]
    out["fading"] = fall.sort_values("change_pts").head(3).to_dict("records")
    out["largest"] = table.sort_values("docs", ascending=False).head(10).to_dict("records")
    return out
