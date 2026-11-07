"""Settings. Every knob has a default that works in GitHub Actions; environment variables override a few for tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_PAGE = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
ARCHIVE_BASE = "https://files.consumerfinance.gov/f/documents/"
# The archive as published on Sept. 14, 2026 (used when the page itself can't be read). The CFPB stopped publishing
# narratives in August 2026; these files are the complete record of narratives it had published.
KNOWN_ARCHIVES = [
    "CCDB_Export_1_December_2011_through_April_2018.zip",
    "CCDB_Export_2_May_2018_through_April_2021.zip",
    "CCDB_Export_3_May_2021_through_October_2022.zip",
    "CCDB_Export_4_November_2022_through_August_2023.zip",
    "CCDB_Export_5_September_2023_through_March_2024.zip",
    "CCDB_Export_6_April_2024_through_July_2024.zip",
    "CCDB_Export_7_August_2024_through_October_2024.zip",
    "CCDB_Export_8_November_2024_through_December_2024.zip",
    "CCDB_Export_9_January_2025_through_February_2025.zip",
    "CCDB_Export_10_March_2025_through_April_2025.zip",
    "CCDB_Export_11_May_2025_through_June_2025.zip",
    "CCDB_Export_12_July_2025_through_August_2025.zip",
    "CCDB_Export_13_September_2025_through_October_2025.zip",
    "CCDB_Export_14_November_2025_through_December_2025.zip",
    "CCDB_Export_15_January_2026_through_February_2026.zip",
    "CCDB_Export_16_March_2026.zip",
    "CCDB_Export_17_April_2026.zip",
    "CCDB_Export_18_May_2026.zip",
    "CCDB_Export_19_June_2026.zip",
    "CCDB_Export_20_July_2026.zip",
    "CCDB_Export_21_August_2026.zip",
]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


@dataclass
class Config:
    # ---- data window: the newest months of the archive, split by time (no future data in training)
    train_months: int = 9  # training months, before the validation month
    test_months: int = 2  # the newest months; the validation month sits just before them
    topic_months: int = 12  # months analysed for emerging issues (natural sample)
    recent_months: int = 3  # "recent" window for emerging issues
    # ---- sampling (hash-based, so the same complaints are picked on every run)
    train_cap_per_product: int = 1500  # balanced training set: at most this many complaints per product
    val_size: int = 6000  # natural (real-mix) sample of the validation month
    test_size_per_month: int = 8000  # natural sample of each test month
    topic_size_per_month: int = 2500  # natural sample of every other month in the topic window
    min_train_per_product: int = 300  # products with fewer training complaints are not routed (too rare)
    max_chars: int = 3000  # narratives are cut to this length when stored
    seed: int = 13
    # ---- models
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # fine-tuned with a classification head
    embed_max_tokens: int = 256
    max_tokens: int = 192
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    warmup_share: float = 0.06
    # ---- routing policy
    target_accuracy: float = (
        0.95  # auto-route only complaints the model is this sure about (accuracy on validation)
    )
    # ---- topics
    topic_min_cluster: int = 100  # smallest topic (complaints); smaller gives hundreds of tiny topics
    topic_min_recent_docs: int = 40
    # ---- quality gate
    min_macro_f1: float = 0.45
    max_macro_f1_drop: float = 0.03
    min_onnx_agreement: float = 0.98
    # ---- paths and modes
    backend: str = field(default_factory=lambda: os.environ.get("ROUTER_BACKEND", "real"))  # "fake" in tests
    fixture: bool = field(default_factory=lambda: bool(os.environ.get("ROUTER_FIXTURE")))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ROUTER_DATA_DIR", ROOT / "data")))
    site_dir: Path = field(default_factory=lambda: Path(os.environ.get("ROUTER_SITE_DIR", ROOT / "site")))
    readme: Path = field(default_factory=lambda: Path(os.environ.get("ROUTER_README", ROOT / "README.md")))
    model_card: Path = field(
        default_factory=lambda: Path(os.environ.get("ROUTER_MODEL_CARD", ROOT / "MODEL_CARD.md"))
    )
    vendor_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("ROUTER_VENDOR_DIR", ROOT / "web" / "vendor"))
    )
    repo_url: str = "https://github.com/dvskaff/complaint-routing-nlp"
    user_agent: str = "complaint-routing-nlp (github.com/dvskaff/complaint-routing-nlp)"

    def __post_init__(self):
        if self.fixture:  # small, fast settings for the synthetic test data
            self.train_cap_per_product, self.val_size, self.test_size_per_month = 60, 120, 120
            self.topic_size_per_month, self.min_train_per_product = 60, 20
            self.epochs, self.batch_size, self.max_tokens, self.embed_max_tokens = 5, 16, 48, 48
            self.learning_rate, self.topic_min_cluster, self.topic_min_recent_docs = 1.5e-3, 10, 3
            self.min_macro_f1, self.min_onnx_agreement = 0.0, 0.9

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"
