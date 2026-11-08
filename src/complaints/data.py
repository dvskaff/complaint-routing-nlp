"""The data: the CFPB's archive of published complaint narratives, read straight from the zip files and sampled.

The CFPB stopped publishing complaint narratives in August 2026 and moved the ones it had published into an archive
(21 zip files, December 2011 to August 14, 2026). This module finds the archive files, downloads the ones that cover
the analysis window, reads them in chunks, and keeps a deterministic sample:

* a class-balanced training set from the training months (at most N complaints per product), and
* "natural" samples (the real product mix) of the validation month, the test months and the other topic months.

Sampling ranks complaints by a salted hash of their ID, so every run picks the same complaints no matter the file
order or chunk size.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ARCHIVE_BASE, ARCHIVE_PAGE, KNOWN_ARCHIVES, Config

log = logging.getLogger(__name__)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]  # fmt: skip
NAME_RE = re.compile(
    rf"CCDB_Export_(\d+)_({'|'.join(MONTHS)})_(\d{{4}})(?:_through_({'|'.join(MONTHS)})_(\d{{4}}))?\.zip",
    re.I,
)
# Column names in the archive, normalized (lowercase, non-letters -> "_"), with the names older exports used.
COLUMNS = {
    "complaint_id": ["complaint_id"],
    "date": ["date_received"],
    "product": ["product"],
    "sub_product": ["sub_product"],
    "issue": ["issue"],
    "company": ["company"],
    "state": ["state"],
    "text": ["consumer_complaint_narrative", "complaint_what_happened", "narrative"],
}
REQUIRED = ("date", "product", "text")
# Products the CFPB renamed; complaints keep the newest name so the label set is stable over time.
RENAMES = {
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",
    "Credit reporting": "Credit reporting or other personal consumer reports",
}  # fmt: skip
SPLITS = ("in_train", "in_val", "in_test", "in_topics")


# --------------------------------------------------------------------------- text
_MASK = re.compile(r"\b(?:X{2,}[/-]?)+\d*(?![A-WYZ])|\{\$([\d,.]+)\}")  # not "XXL"
_TAGS = re.compile(
    r"<?\b(?:year|scrub|redacted|date|name)\w*>|<\w+>", re.I
)  # newer redaction marks: "on year>."


def clean_text(text: str, max_chars: int = 3000) -> str:
    """Drop the CFPB's redaction marks (XXXX, XX/XX/XXXX, year>, scrub>), unwrap amounts ({$45.00} -> $45.00),
    squash spaces."""

    def sub(m: re.Match) -> str:
        return f"${m.group(1)}" if m.group(1) else " "

    t = _TAGS.sub(" ", _MASK.sub(sub, str(text)))
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    return t[:max_chars]


def dedupe_key(text: str) -> str:
    """Same key for narratives that differ only in case, spacing or punctuation (template letters)."""
    norm = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- archive files
@dataclass(frozen=True)
class Archive:
    name: str
    url: str
    start: str  # first month covered, "YYYY-MM"
    end: str  # last month covered


def parse_name(name: str) -> tuple[str, str] | None:
    m = NAME_RE.fullmatch(name)
    if not m:
        return None
    month = {n.lower(): i + 1 for i, n in enumerate(MONTHS)}
    start = f"{m.group(3)}-{month[m.group(2).lower()]:02d}"
    end = f"{m.group(5)}-{month[m.group(4).lower()]:02d}" if m.group(4) else start
    return start, end


def session(cfg: Config, retries: int = 6) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=3, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = cfg.user_agent
    return s


def list_archives(cfg: Config, page: str | None = None) -> list[Archive]:
    """Archive files listed on the CFPB's FOIA page; the known list if the page can't be read or has no links."""
    page = page or (str(cfg.data_dir / "fixture" / "archive.html") if cfg.fixture else ARCHIVE_PAGE)
    names, base = [], ARCHIVE_BASE
    try:
        if page.startswith("http"):
            html = session(cfg, retries=2).get(page, timeout=30).text  # optional: don't wait long
        else:
            html = Path(page).read_text()
            base = str(Path(page).parent) + "/"
        names = sorted(set(re.findall(r"CCDB_Export_[A-Za-z0-9_]+\.zip", html)))
    except Exception as exc:  # the page is only a convenience: new files would show up there first
        log.warning("Could not read the archive page (%s); using the known file list", exc)
    if not names:
        names, base = KNOWN_ARCHIVES, ARCHIVE_BASE
        log.info("Archive page had no file links; using the %d known files", len(names))
    out = []
    for n in names:
        months = parse_name(n)
        if months:
            out.append(Archive(n, base + n, *months))
    if not out:
        raise RuntimeError("No archive files found")
    return sorted(out, key=lambda a: (a.start, a.end))


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


@dataclass(frozen=True)
class Window:
    train: list[str]
    val: str
    test: list[str]
    topics: list[str]

    @property
    def first(self) -> str:
        return min(self.train[0], self.topics[0])

    @property
    def last(self) -> str:
        return self.test[-1]


def window_ending(last: str, cfg: Config) -> Window:
    """Newest months for testing, the month before for validation, the months before that for training."""
    p = pd.Period(last, freq="M")
    test = [str(p - i) for i in range(cfg.test_months - 1, -1, -1)]
    val = str(p - cfg.test_months)
    train = [str(p - cfg.test_months - i) for i in range(cfg.train_months, 0, -1)]
    topics = [str(p - i) for i in range(cfg.topic_months - 1, -1, -1)]
    return Window(train, val, test, topics)


def plan_window(archives: list[Archive], cfg: Config) -> Window:
    """Provisional window, ending with the newest month any archive file covers."""
    return window_ending(max(a.end for a in archives), cfg)


def effective_last(counts: dict[str, int], last: str, lookback: int = 6, share: float = 0.2) -> str:
    """The newest month with a normal volume of narratives. The final archive month can be nearly empty (August 2026
    holds a single narrative), and a month like that can't be a test month or a point on a trend line."""
    p = pd.Period(last, freq="M")
    for _ in range(lookback):
        before = [counts.get(str(p - i), 0) for i in range(1, lookback + 1)]
        before = [c for c in before if c > 0]
        if not before or counts.get(str(p), 0) >= share * float(np.median(before)):
            return str(p)
        p -= 1
    return str(p)


def candidate_files(archives: list[Archive], cfg: Config, spare_months: int = 3) -> list[Archive]:
    """Files covering the provisional window plus a few spare months, in case the newest months are nearly empty."""
    first = str(pd.Period(plan_window(archives, cfg).first, freq="M") - spare_months)
    return [a for a in archives if a.end >= first]


def needed(archives: list[Archive], w: Window) -> list[Archive]:
    return [a for a in archives if a.end >= w.first]


def download(cfg: Config, a: Archive, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / a.name
    if out.exists() and zipfile.is_zipfile(out):
        return out
    if not a.url.startswith("http"):  # fixture files
        shutil.copy(a.url, out)
        return out
    tmp = out.with_suffix(".part")
    with session(cfg).get(a.url, stream=True, timeout=120) as r:
        r.raise_for_status()
        size = int(r.headers.get("Content-Length") or 0)
        log.info("Downloading %s (%s MB)", a.name, f"{size / 1e6:,.0f}" if size else "?")
        with tmp.open("wb") as fh:
            for block in r.iter_content(chunk_size=1 << 20):
                fh.write(block)
    if not zipfile.is_zipfile(tmp):
        head = tmp.read_bytes()[:200]
        tmp.unlink()
        raise RuntimeError(f"{a.name} is not a zip file (the server sent: {head!r})")
    tmp.rename(out)
    return out


# --------------------------------------------------------------------------- reading
def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lstrip("﻿").lower()).strip("_")


def _layout(header_line: str) -> tuple[str, dict[str, str]]:
    sep = max([",", "\t", "|", ";"], key=header_line.count)
    names = next(csv.reader([header_line], delimiter=sep))
    by_norm = {_norm(n): n for n in names}
    found = {}
    for key, options in COLUMNS.items():
        for o in options:
            if o in by_norm:
                found[key] = by_norm[o]
                break
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        raise RuntimeError(f"Archive columns {names} are missing {missing}")
    return sep, found


def read_archive(path: Path, chunksize: int = 100_000):
    """Yield standardized chunks (complaint_id, date, product, sub_product, issue, company, state, text)."""
    with zipfile.ZipFile(path) as zf:
        members = [
            m for m in zf.namelist() if m.lower().endswith((".csv", ".txt")) and not m.startswith("__MACOSX")
        ]
        if not members:
            raise RuntimeError(f"{path.name} has no CSV file inside (found {zf.namelist()[:5]})")
        for m in members:
            with zf.open(m) as fh:
                header = io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace").readline()
            sep, cols = _layout(header)
            log.info("Reading %s/%s (columns: %s)", path.name, m, ", ".join(cols.values()))
            with zf.open(m) as fh:
                reader = pd.read_csv(fh, sep=sep, usecols=list(cols.values()), dtype=str, chunksize=chunksize,
                                     encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip",
                                     keep_default_na=False)  # fmt: skip
                for chunk in reader:
                    yield standardize(chunk.rename(columns={v: k for k, v in cols.items()}))


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df = df[list(COLUMNS)].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() >= 20]
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["product"] = df["product"].fillna("").str.strip().replace(RENAMES)
    df = df[df["date"].notna() & (df["product"] != "")]
    missing_id = df["complaint_id"].fillna("").str.strip() == ""
    if missing_id.any():  # no ID: derive a stable one from the content
        df.loc[missing_id, "complaint_id"] = [
            "h" + hashlib.sha1(f"{d}|{t}".encode()).hexdigest()[:15]
            for d, t in zip(df.loc[missing_id, "date"], df.loc[missing_id, "text"], strict=True)
        ]
    df["complaint_id"] = df["complaint_id"].astype(str).str.strip()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


# --------------------------------------------------------------------------- sampling
def _rank(ids: pd.Series, salt: str) -> np.ndarray:
    key = hashlib.sha1(salt.encode()).hexdigest()[:16]
    return pd.util.hash_array(ids.to_numpy(dtype=object), hash_key=key)


class Sampler:
    """Keeps the lowest-hash complaints per group while streaming through the archive."""

    def __init__(self, cfg: Config, w: Window):
        self.cfg, self.w = cfg, w
        self.pools: dict[str, pd.DataFrame] = {}
        self.caps = {"train": cfg.train_cap_per_product * 2}  # 2x: room to drop duplicate templates later
        self.natural_caps = {m: cfg.topic_size_per_month for m in w.topics}
        self.natural_caps[w.val] = cfg.val_size
        for m in w.test:
            self.natural_caps[m] = cfg.test_size_per_month
        self.rows_seen = 0
        self.by_month: dict[str, int] = {}

    def _keep(self, name: str, df: pd.DataFrame, group: str, cap) -> None:
        pool = pd.concat([self.pools[name], df]) if name in self.pools else df
        pool = pool.drop_duplicates("complaint_id").sort_values("h", kind="stable")
        if isinstance(cap, dict):
            pool = pool[pool[group].map(cap).notna()]
            pool = pool[pool.groupby(group).cumcount() < pool[group].map(cap)]
        else:
            pool = pool.groupby(group).head(cap)
        self.pools[name] = pool

    def add(self, df: pd.DataFrame) -> None:
        self.rows_seen += len(df)
        df = df.assign(
            text=df["text"].str.slice(0, 2 * self.cfg.max_chars)
        )  # bound memory; cleaned at the end
        for m, n in df["month"].value_counts().items():
            self.by_month[m] = self.by_month.get(m, 0) + int(n)
        tr = df[df["month"].isin(self.w.train)].assign(
            h=lambda d: _rank(d["complaint_id"], f"train{self.cfg.seed}")
        )
        if len(tr):
            self._keep("train", tr, "product", self.caps["train"])
        nat = df[df["month"].isin(list(self.natural_caps))]
        if len(nat):
            nat = nat.assign(h=_rank(nat["complaint_id"], f"natural{self.cfg.seed}"))
            self._keep("natural", nat, "month", self.natural_caps)

    def result(self) -> pd.DataFrame:
        """One row per complaint with split flags (a complaint can be in training and the topic sample)."""
        cfg, w = self.cfg, self.w
        train = self.pools.get("train", pd.DataFrame(columns=[*COLUMNS, "month", "h"]))
        train = train.assign(text=train["text"].map(lambda t: clean_text(t, cfg.max_chars)))
        train = train.assign(dedupe=train["text"].map(dedupe_key)).drop_duplicates("dedupe")
        train = train.groupby("product").head(cfg.train_cap_per_product)
        nat = self.pools.get("natural", pd.DataFrame(columns=[*COLUMNS, "month", "h"]))
        nat = nat.assign(text=nat["text"].map(lambda t: clean_text(t, cfg.max_chars)))
        nat = nat.assign(dedupe=nat["text"].map(dedupe_key))
        keep = ["complaint_id", "date", "month", "product", "sub_product", "issue", "company", "state", "text",
                "dedupe"]  # fmt: skip
        df = pd.concat([train[keep], nat[keep]]).drop_duplicates("complaint_id").reset_index(drop=True)
        ids_train = set(train["complaint_id"])
        df["in_train"] = df["complaint_id"].isin(ids_train)
        df["in_val"] = df["complaint_id"].isin(nat["complaint_id"]) & (df["month"] == w.val)
        df["in_test"] = df["complaint_id"].isin(nat["complaint_id"]) & df["month"].isin(w.test)
        df["in_topics"] = df["complaint_id"].isin(nat["complaint_id"]) & df["month"].isin(w.topics)
        return df.sort_values(["month", "complaint_id"]).reset_index(drop=True)


def build_samples(cfg: Config, archives: list[Archive]) -> tuple[pd.DataFrame, dict]:
    """Download the candidate archive files, count narratives per month to place the window, then sample."""
    raw = cfg.data_dir / "raw"
    files = candidate_files(archives, cfg)
    paths = [download(cfg, a, raw) for a in files]
    counts: dict[str, int] = {}
    for path in paths:  # pass 1: monthly volumes
        for chunk in read_archive(path):
            for m, n in chunk["month"].value_counts().items():
                counts[m] = counts.get(m, 0) + int(n)
    provisional = max(a.end for a in archives)
    last = effective_last(counts, provisional)
    w = window_ending(last, cfg)
    sparse = [m for m in month_range(last, provisional)[1:]]
    if sparse:
        log.info("Ending the window at %s: %s hold only %s narratives", last, ", ".join(sparse),
                 ", ".join(f"{counts.get(m, 0):,}" for m in sparse))  # fmt: skip
    log.info(
        "Window: train %s to %s, validation %s, test %s", w.train[0], w.train[-1], w.val, ", ".join(w.test)
    )
    sampler = Sampler(cfg, w)
    for path in paths:  # pass 2: sampling
        for chunk in read_archive(path):
            sampler.add(chunk)
        log.info("  %s rows read so far", f"{sampler.rows_seen:,}")
    for path in paths:
        path.unlink()
    df = sampler.result()
    stats = {"files": [a.name for a in files], "rows_read": sampler.rows_seen, "window": asdict(w),
             "sparse_months": {m: counts.get(m, 0) for m in sparse},
             "narratives_by_month": {m: counts.get(m, 0) for m in month_range(w.first, w.last)}}  # fmt: skip
    return df, stats


def fingerprint(df: pd.DataFrame) -> str:
    payload = json.dumps([df["complaint_id"].tolist(), df["product"].tolist(), df["text"].str.len().tolist(),
                          [df[s].tolist() for s in SPLITS]])  # fmt: skip
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def label_set(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Products with enough training complaints to learn; rarer ones are left to human review (not routed)."""
    counts = df[df["in_train"]]["product"].value_counts()
    return sorted(counts[counts >= cfg.min_train_per_product].index)
