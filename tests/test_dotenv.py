import os

from agent_eval.cli import _load_dotenv


class TestLoadDotenv:
    def test_loads_key_value_pairs(self, tmp_path, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        p = tmp_path / ".env"
        p.write_text(
            "# comment\n"
            "\n"
            "KIMI_API_KEY=sk-test-123\n"
            'KIMI_API_BASE="https://api.kimi.com/coding/v1"\n'
            "QUOTED='single'\n",
            encoding="utf-8",
        )
        for k in ("KIMI_API_BASE", "QUOTED"):
            monkeypatch.delenv(k, raising=False)

        loaded = _load_dotenv(p)

        assert set(loaded) == {"KIMI_API_KEY", "KIMI_API_BASE", "QUOTED"}
        assert os.environ["KIMI_API_KEY"] == "sk-test-123"
        assert os.environ["KIMI_API_BASE"] == "https://api.kimi.com/coding/v1"
        assert os.environ["QUOTED"] == "single"

    def test_existing_env_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "from-shell")
        p = tmp_path / ".env"
        p.write_text("KIMI_API_KEY=from-dotenv\n", encoding="utf-8")

        assert _load_dotenv(p) == []
        assert os.environ["KIMI_API_KEY"] == "from-shell"

    def test_missing_file_is_noop(self, tmp_path):
        assert _load_dotenv(tmp_path / "nope.env") == []

    def test_malformed_lines_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GOOD", raising=False)
        p = tmp_path / ".env"
        p.write_text("no-equals-here\n=empty-key\nGOOD=1\n", encoding="utf-8")

        assert _load_dotenv(p) == ["GOOD"]
        assert os.environ["GOOD"] == "1"
