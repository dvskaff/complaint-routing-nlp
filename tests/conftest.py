import os

os.environ["ROUTER_BACKEND"] = "fake"  # stand-in models: no downloads in tests
os.environ["ROUTER_FIXTURE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest  # noqa: E402

from complaints.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ROUTER_SITE_DIR", str(tmp_path / "site"))
    monkeypatch.setenv("ROUTER_README", str(tmp_path / "README.md"))
    monkeypatch.setenv("ROUTER_MODEL_CARD", str(tmp_path / "MODEL_CARD.md"))
    monkeypatch.setenv("ROUTER_VENDOR_DIR", str(tmp_path / "vendor"))
    return Config()


@pytest.fixture
def samples(cfg):
    from complaints import data
    from complaints.fixture import make_archive

    make_archive(cfg.data_dir / "fixture")
    archives = data.list_archives(cfg)
    df, stats = data.build_samples(cfg, archives)
    return df, data.Window(**stats["window"]), stats
