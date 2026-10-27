from __future__ import annotations

import hashlib
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Iterable

from . import __version__
from .judge import JUDGE_PROMPT_TEMPLATE
from .simulation import USER_PROMPT_TEMPLATE


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_value(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _file_content(path: Path) -> bytes:
    raw = path.read_bytes()
    if b"\0" in raw:
        return raw
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def dataset_hash(source_path: str | Path, env_paths: Iterable[str | Path] = ()) -> str:
    digest = hashlib.sha256()
    source = Path(source_path)
    digest.update(_file_content(source))
    for index, env_path in enumerate(env_paths):
        root = Path(env_path)
        digest.update(f"\nenv:{index}\n".encode())
        if not root.is_dir():
            raise ValueError(f"env fixture is not a directory: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(_file_content(path))
            digest.update(b"\0")
    return digest.hexdigest()


def judge_config(judge_backend: str | None, judge_model: str | None) -> dict:
    return {
        "backend": judge_backend,
        "model": judge_model if judge_backend else None,
        "prompt_sha256": _sha256_bytes(JUDGE_PROMPT_TEMPLATE.encode())
        if judge_backend
        else None,
    }


def build_manifest(
    *,
    run_id: str,
    kind: str,
    source_path: str,
    case_ids: list[str],
    samples: int,
    concurrency: int,
    agent: str,
    judge_backend: str | None,
    judge_model: str | None,
    user_backend: str | None = None,
    user_model: str | None = None,
    env_paths: Iterable[str | Path] = (),
) -> dict:
    source_name = Path(source_path).as_posix()
    execution_id = uuid.uuid4().hex
    return {
        "execution_id": execution_id,
        "run_id": run_id,
        "source_key": f"{kind}:{source_name}",
        "kind": kind,
        "source_path": source_path,
        "source_hash": dataset_hash(source_path, env_paths),
        "case_ids": case_ids,
        "samples": samples,
        "concurrency": concurrency,
        "agent": agent,
        "agent_eval_version": __version__,
        "python_version": platform.python_version(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": _git_value("status", "--porcelain") not in {"", "unknown"},
        "judge": judge_config(judge_backend, judge_model),
        "simulated_user": {
            "backend": user_backend,
            "model": user_model if user_backend == "litellm" else None,
            "prompt_sha256": _sha256_bytes(USER_PROMPT_TEMPLATE.encode())
            if user_backend == "litellm"
            else None,
        }
        if kind == "scenario"
        else None,
    }
