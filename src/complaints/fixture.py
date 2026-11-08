"""SYNTHETIC test data: a fake narrative archive in the CFPB's file layout, small enough for tests to run in seconds.

Nothing here is real: products have their own invented vocabulary, a template letter is repeated (like the credit
repair templates in the real data), and an "impersonation scam" theme grows in the last three months so the
emerging-issue code has something to find.
"""

from __future__ import annotations

import csv
import io
import random
import zipfile
from pathlib import Path

HEADER = ["Date received", "Product", "Sub-product", "Issue", "Sub-issue", "Consumer complaint narrative",
          "Company public response", "Company", "State", "ZIP code", "Tags", "Submitted via",
          "Date sent to company", "Company response to consumer", "Timely response?", "Complaint ID"]  # fmt: skip
CREDIT = "Credit reporting or other personal consumer reports"
PRODUCTS = {
    CREDIT: (0.55, ["credit report", "inaccurate account", "dispute with the bureau", "late payment reported",
                    "hard inquiry", "identity theft on my report", "reinvestigation", "consumer reporting agency"]),
    "Debt collection": (0.13, ["debt collector", "collection agency", "calls about a debt", "validation letter",
                               "owe this debt", "threatened to sue"]),
    "Credit card": (0.09, ["credit card", "charged twice", "statement balance", "interest rate", "chargeback",
                           "annual fee on the card"]),
    "Checking or savings account": (0.09, ["checking account", "overdraft fee", "deposit was held", "branch",
                                           "account was closed", "debit card"]),
    "Mortgage": (0.07, ["mortgage", "escrow account", "loan modification", "foreclosure", "loan servicer",
                        "monthly payment went up"]),
    "Money transfer, virtual currency, or money service": (0.065, ["money transfer", "wire", "sent money",
                                                                   "payment app", "transfer was reversed"]),
    "Prepaid card": (0.002, ["prepaid card", "card balance", "reload"]),
}  # fmt: skip
FILLER = ["I contacted them several times", "nobody would help me", "this has caused me stress",
          "I have documents to prove it", "please look into this", "they did not respond", "on XX/XX/XXXX",
          "I was told to wait", "the amount was {$250.00}", "my name is XXXX XXXX"]  # fmt: skip
SCAM = ["someone pretending to be my bank called me", "impersonation scam", "they asked me to move money to a safe "
        "account", "the fraud department never refunded me"]  # fmt: skip
TEMPLATE = ("I am writing to dispute the following information in my file under 15 USC 1681. The items listed are "
            "inaccurate and must be deleted from my credit report immediately.")  # fmt: skip
ARCHIVES = [("CCDB_Export_1_August_2025_through_December_2025.zip", "2025-08", "2025-12"),
            ("CCDB_Export_2_January_2026_through_April_2026.zip", "2026-01", "2026-04"),
            ("CCDB_Export_3_May_2026.zip", "2026-05", "2026-05"), ("CCDB_Export_4_June_2026.zip", "2026-06", "2026-06"),
            ("CCDB_Export_5_July_2026.zip", "2026-07", "2026-07"),
            ("CCDB_Export_6_August_2026.zip", "2026-08", "2026-08")]  # fmt: skip
COMPANIES = ["Acme Bank", "Blue Harbor Credit", "Cedar Financial", "Delta Lending", "Evergreen Bureau"]


def _narrative(rng: random.Random, product: str, month: str) -> str:
    words = PRODUCTS[product][1]
    if product == CREDIT and rng.random() < 0.15:
        return TEMPLATE
    parts = rng.sample(words, k=rng.choice([1, 2, 3])) + rng.sample(FILLER, k=3)
    if (
        rng.random() < 0.25
    ):  # many complaints mention another product too (overlap and label noise, as in real data)
        other = rng.choice([p for p in PRODUCTS if p != product])
        parts += rng.sample(PRODUCTS[other][1], k=2)
    recent = month >= "2026-06"
    if product in ("Money transfer, virtual currency, or money service", "Checking or savings account"):
        if rng.random() < (0.7 if recent else 0.05):
            parts += rng.sample(SCAM, k=2)
    rng.shuffle(parts)
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."


def make_archive(dest: Path, per_month: int = 420, seed: int = 7, last_month_rows: int = 2) -> Path:
    """Write the fake archive zips and an archive page that links them. Returns the page path.

    Like the real archive, the final month is nearly empty (the CFPB stopped publishing in mid-August 2026)."""
    import pandas as pd

    rng = random.Random(seed)
    dest.mkdir(parents=True, exist_ok=True)
    names, weights = list(PRODUCTS), [v[0] for v in PRODUCTS.values()]
    next_id = 9_000_000
    links = []
    for name, start, end in ARCHIVES:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(HEADER)
        for month in pd.period_range(start, end, freq="M"):
            n = last_month_rows if str(month) == ARCHIVES[-1][2] else per_month
            for _ in range(n):
                product = rng.choices(names, weights)[0]
                day = rng.randint(1, 28)
                next_id += 1
                w.writerow([f"{month}-{day:02d}", product, "", "Issue", "", _narrative(rng, product, str(month)),
                            "", rng.choice(COMPANIES), "CA", "", "", "Web", f"{month}-{day:02d}", "Closed", "Yes",
                            next_id])  # fmt: skip
        with zipfile.ZipFile(dest / name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(name.replace(".zip", ".csv"), "﻿" + buf.getvalue())
        links.append(f'<a href="https://files.consumerfinance.gov/f/documents/{name}">{name}</a>')
    page = dest / "archive.html"
    page.write_text(
        "<html><body>SYNTHETIC archive page<ul>" + "".join(f"<li>{x}</li>" for x in links) + "</ul>"
    )
    return page
