import zipfile

import pandas as pd
import pytest

from complaints import data
from complaints.fixture import make_archive


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CCDB_Export_21_August_2026.zip", ("2026-08", "2026-08")),
        ("CCDB_Export_13_September_2025_through_October_2025.zip", ("2025-09", "2025-10")),
        ("CCDB_Export_1_December_2011_through_April_2018.zip", ("2011-12", "2018-04")),
        ("something_else.zip", None),
    ],
)
def test_parse_name(name, expected):
    assert data.parse_name(name) == expected


def test_window_is_time_ordered(cfg):
    archives = data.list_archives(cfg, page=str(make_archive(cfg.data_dir / "fixture")))
    w = data.plan_window(archives, cfg)  # provisional: ends with the newest month any file covers
    assert w.test == ["2026-07", "2026-08"] and w.val == "2026-06"
    assert w.train[-1] < w.val < w.test[0] and len(w.train) == cfg.train_months
    assert w.topics[-1] == "2026-08" and len(w.topics) == cfg.topic_months
    # spare months are fetched too, in case the newest months turn out to be nearly empty
    assert [a.name for a in data.candidate_files(archives, cfg)][0].startswith("CCDB_Export_1_")


def test_nearly_empty_final_month_is_dropped():
    counts = {"2026-02": 900, "2026-03": 1000, "2026-04": 950, "2026-05": 1100, "2026-06": 1000, "2026-07": 800,
              "2026-08": 1}  # fmt: skip
    assert data.effective_last(counts, "2026-08") == "2026-07"
    assert data.effective_last({**counts, "2026-08": 400}, "2026-08") == "2026-08"
    assert data.effective_last({"2026-08": 5}, "2026-08") == "2026-08"  # no history: keep it


def test_window_follows_the_data(samples):
    df, w, stats = samples
    assert w.test == ["2026-06", "2026-07"] and w.val == "2026-05"  # the fixture's August holds 2 complaints
    assert stats["sparse_months"] == {"2026-08": 2} and df["month"].max() == "2026-07"


def test_known_list_when_page_has_no_links(cfg, tmp_path):
    page = tmp_path / "blocked.html"
    page.write_text("<html>Access denied</html>")
    archives = data.list_archives(cfg, page=str(page))
    assert len(archives) == 21 and archives[-1].name == "CCDB_Export_21_August_2026.zip"
    assert archives[-1].url.startswith("https://files.consumerfinance.gov/")


def test_clean_text_and_dedupe_key():
    t = data.clean_text("On XX/XX/XXXX I paid {$1,200.50} to XXXX XXXX bank , then XXXX.")
    assert t == "On I paid $1,200.50 to bank, then."
    assert data.clean_text("Paid on XX/XX/year>. Then <scrub> called on year>2026, again.") == (
        "Paid on. Then called on 2026, again."
    )
    assert data.clean_text("If balance > 0 then pay") == "If balance > 0 then pay"
    assert data.dedupe_key("I dispute THIS item!") == data.dedupe_key("i dispute this item")
    assert data.dedupe_key("a") != data.dedupe_key("b")


def _zip(tmp_path, text: str, name: str = "x.zip"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("export.csv", text)
    return p


def test_read_archive_maps_columns_bom_and_tabs(tmp_path):
    good = ("﻿Date received,Product,Consumer complaint narrative,Complaint ID\n"
            '2026-07-02,Mortgage,"My escrow, as usual,\nwas wrong and nobody helps",101\n'
            "2026-07-03,,no product here so it is dropped from the data,102\n")  # fmt: skip
    df = pd.concat(data.read_archive(_zip(tmp_path, good)))
    assert df["complaint_id"].tolist() == ["101"] and df["month"].tolist() == ["2026-07"]
    assert "escrow" in df["text"].iloc[0]
    tabs = "date_received\tproduct\tcomplaint_what_happened\n07/04/2026\tCredit card\tCharged twice for one purchase\n"
    df2 = pd.concat(data.read_archive(_zip(tmp_path, tabs, "y.zip")))
    assert df2["product"].tolist() == ["Credit card"] and df2["complaint_id"].str.startswith("h").all()
    with pytest.raises(RuntimeError, match="missing"):
        list(data.read_archive(_zip(tmp_path, "Date received,Product\n2026-07-01,Mortgage\n", "z.zip")))


def test_old_product_names_are_renamed():
    df = data.standardize(pd.DataFrame({
        "date": ["2026-01-05"], "text": ["A narrative long enough to keep."],
        "product": ["Credit reporting, credit repair services, or other personal consumer reports"]}))  # fmt: skip
    assert df["product"].iloc[0] == "Credit reporting or other personal consumer reports"


def test_samples_balanced_natural_and_deterministic(cfg, samples):
    df, w, stats = samples
    train = df[df["in_train"]]
    assert train["month"].isin(w.train).all()
    assert train["product"].value_counts().max() <= cfg.train_cap_per_product
    assert not train["dedupe"].duplicated().any()  # template letters appear once in training
    test = df[df["in_test"]]
    assert (
        test["month"].isin(w.test).all() and (test.groupby("month").size() <= cfg.test_size_per_month).all()
    )
    assert not (df["in_train"] & (df["in_val"] | df["in_test"])).any()  # no overlap across the time split
    assert stats["rows_read"] > len(df)
    # same sample whatever the chunk size (hash-based ranking)
    sampler = data.Sampler(cfg, w)
    for a in data.candidate_files(data.list_archives(cfg), cfg):
        path = cfg.data_dir / "raw" / a.name
        data.download(cfg, a, cfg.data_dir / "raw")
        for chunk in data.read_archive(path, chunksize=97):
            sampler.add(chunk)
    again = sampler.result()
    assert data.fingerprint(again) == data.fingerprint(df)


def test_label_set_drops_rare_products(cfg, samples):
    df, _, _ = samples
    labels = data.label_set(df, cfg)
    assert "Prepaid card" not in labels and len(labels) == 6


class _Resp:
    def __init__(self, body: bytes):
        self.body, self.headers = body, {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.body


def test_download_rejects_error_pages(cfg, tmp_path, monkeypatch):
    class Session:
        def get(self, url, **kw):
            return _Resp(b"<html>Access denied</html>")

    monkeypatch.setattr(data, "session", lambda cfg: Session())
    a = data.Archive("CCDB_Export_9_May_2026.zip", "https://example.invalid/x.zip", "2026-05", "2026-05")
    with pytest.raises(RuntimeError, match="not a zip"):
        data.download(cfg, a, tmp_path / "raw")
    assert not list((tmp_path / "raw").iterdir())  # no partial file left behind
