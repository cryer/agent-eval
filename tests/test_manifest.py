import json

from agent_eval.manifest import build_manifest, dataset_hash


def test_dataset_hash_includes_fixture_contents(tmp_path):
    source = tmp_path / "cases.yaml"
    source.write_text("cases", encoding="utf-8")
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "code.py").write_text("value = 1\n", encoding="utf-8")
    before = dataset_hash(source, [fixture])
    assert before == dataset_hash(source, [fixture])
    (fixture / "code.py").write_text("value = 2\n", encoding="utf-8")
    assert dataset_hash(source, [fixture]) != before


def test_dataset_hash_normalizes_text_line_endings(tmp_path):
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    for root, newline in ((lf, "\n"), (crlf, "\r\n")):
        root.mkdir()
        (root / "cases.yaml").write_bytes(f"- id: c1{newline}".encode())
        (root / "fixture").mkdir()
        (root / "fixture" / "code.py").write_bytes(f"value = 1{newline}".encode())
    assert dataset_hash(lf / "cases.yaml", [lf / "fixture"]) == dataset_hash(
        crlf / "cases.yaml", [crlf / "fixture"]
    )


def test_manifest_contains_reproducibility_config_but_no_credentials(tmp_path, monkeypatch):
    source = tmp_path / "cases.yaml"
    source.write_text("cases", encoding="utf-8")
    monkeypatch.setenv("KIMI_API_KEY", "secret-that-must-not-be-recorded")
    manifest = build_manifest(
        run_id="r1",
        kind="case",
        source_path=str(source),
        case_ids=["c1"],
        samples=3,
        concurrency=2,
        agent="example:agent",
        judge_backend="litellm",
        judge_model="k3",
    )
    assert manifest["git_commit"]
    assert manifest["samples"] == 3
    assert manifest["judge"]["model"] == "k3"
    assert len(manifest["judge"]["prompt_sha256"]) == 64
    assert "secret-that-must-not-be-recorded" not in json.dumps(manifest)
