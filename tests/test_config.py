from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from rag_med.config import Settings


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run from an empty dir so no real .env / config.yaml leaks in."""
    monkeypatch.chdir(tmp_path)
    for var in ("NCBI_API_KEY", "NCBI_EMAIL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _set_required(monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "fake-ncbi-key")
    monkeypatch.setenv("NCBI_EMAIL", "dev@example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")


def test_defaults(isolated_cwd, monkeypatch):
    _set_required(monkeypatch)
    s = Settings()
    assert s.ncbi_api_key == "fake-ncbi-key"
    assert s.ncbi_email == "dev@example.com"
    assert isinstance(s.anthropic_api_key, SecretStr)
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-fake"
    assert s.monthly_cap_usd == 15.0
    assert s.per_query_ceiling_usd == 0.10
    assert s.max_tokens == 1024
    assert s.rerank_floor == 0.0
    assert s.hf_home == Path("./data/hf_cache")
    assert s.data_dir == Path("./data")


def test_sqlite_path_computed(isolated_cwd, monkeypatch):
    _set_required(monkeypatch)
    s = Settings()
    assert s.sqlite_path == Path("./data") / "sqlite.db"


def test_sqlite_path_follows_data_dir(isolated_cwd, monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("DATA_DIR", "/tmp/rag-med-test")
    s = Settings()
    assert s.sqlite_path == Path("/tmp/rag-med-test/sqlite.db")


def test_missing_required_fails(isolated_cwd, monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    with pytest.raises(ValidationError):
        Settings()


def test_bad_email_rejected(isolated_cwd, monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("NCBI_EMAIL", "not-an-email")
    with pytest.raises(ValidationError):
        Settings()


def test_secret_str_does_not_leak_in_repr(isolated_cwd, monkeypatch):
    _set_required(monkeypatch)
    s = Settings()
    assert "sk-ant-fake" not in repr(s)
    assert "sk-ant-fake" not in str(s)


def test_yaml_overrides_defaults(isolated_cwd, monkeypatch):
    _set_required(monkeypatch)
    (isolated_cwd / "config.yaml").write_text(
        "monthly_cap_usd: 99.0\nmax_tokens: 2048\nrerank_floor: 0.25\n"
    )
    s = Settings()
    assert s.monthly_cap_usd == 99.0
    assert s.max_tokens == 2048
    assert s.rerank_floor == 0.25


def test_env_overrides_yaml(isolated_cwd, monkeypatch):
    _set_required(monkeypatch)
    (isolated_cwd / "config.yaml").write_text("monthly_cap_usd: 99.0\n")
    monkeypatch.setenv("MONTHLY_CAP_USD", "7.5")
    s = Settings()
    assert s.monthly_cap_usd == 7.5
