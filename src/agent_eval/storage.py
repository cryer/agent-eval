from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    trace_id       TEXT NOT NULL,
    span_id        TEXT NOT NULL,
    parent_span_id TEXT,
    name           TEXT NOT NULL,
    start_time     INTEGER NOT NULL,
    end_time       INTEGER NOT NULL,
    attributes     TEXT NOT NULL,
    status         TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    agent       TEXT NOT NULL,
    cases_path  TEXT NOT NULL,
    total       INTEGER NOT NULL,
    passed      INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    case_id     TEXT NOT NULL,
    sample_idx  INTEGER NOT NULL DEFAULT 0,
    passed      INTEGER NOT NULL,
    output      TEXT NOT NULL,
    assertions  TEXT NOT NULL,
    trace_id    TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    cost_usd    REAL NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibrations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    judge       TEXT NOT NULL,
    golden_path TEXT NOT NULL,
    metrics     TEXT NOT NULL,
    details     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        # sample_idx was added after the first schema version
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(case_results)")}
        if "sample_idx" not in cols:
            self._conn.execute(
                "ALTER TABLE case_results ADD COLUMN sample_idx INTEGER NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- spans ----

    def insert_span(self, span) -> None:
        ctx = span.get_span_context()
        parent_id = f"{span.parent.span_id:016x}" if span.parent else None
        attrs = {k: v for k, v in (span.attributes or {}).items()}
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{ctx.trace_id:032x}",
                    f"{ctx.span_id:016x}",
                    parent_id,
                    span.name,
                    span.start_time or 0,
                    span.end_time or 0,
                    json.dumps(attrs, ensure_ascii=False, default=str),
                    span.status.status_code.name,
                ),
            )
            self._conn.commit()

    def get_spans(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (trace_id,),
            ).fetchall()
        return [
            {
                "trace_id": r["trace_id"],
                "span_id": r["span_id"],
                "parent_span_id": r["parent_span_id"],
                "name": r["name"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "attributes": json.loads(r["attributes"]),
                "status": r["status"],
            }
            for r in rows
        ]

    # ---- runs / results ----

    def record_run(self, run_id: str, agent: str, cases_path: str, total: int, passed: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, agent, cases_path, total, passed, _now()),
            )
            self._conn.commit()

    def record_case_result(
        self,
        run_id: str,
        case_id: str,
        passed: bool,
        output: str,
        assertions: list[dict],
        trace_id: str,
        duration_ms: float,
        cost_usd: float,
        sample_idx: int = 0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO case_results"
                " (run_id, case_id, sample_idx, passed, output, assertions, trace_id,"
                " duration_ms, cost_usd, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    case_id,
                    sample_idx,
                    int(passed),
                    output,
                    json.dumps(assertions, ensure_ascii=False),
                    trace_id,
                    duration_ms,
                    cost_usd,
                    _now(),
                ),
            )
            self._conn.commit()

    def delete_run_results(self, run_id: str) -> None:
        """Remove prior results of a run so re-running the same run_id is idempotent."""
        with self._lock:
            self._conn.execute("DELETE FROM case_results WHERE run_id = ?", (run_id,))
            self._conn.commit()

    def get_run_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM case_results WHERE run_id = ? ORDER BY case_id, sample_idx",
                (run_id,),
            ).fetchall()
        return [
            {**dict(r), "assertions": json.loads(r["assertions"])}
            for r in rows
        ]

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- calibrations ----

    def record_calibration(
        self,
        judge: str,
        golden_path: str,
        metrics: dict[str, float],
        details: list[dict],
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO calibrations (judge, golden_path, metrics, details, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    judge,
                    golden_path,
                    json.dumps(metrics),
                    json.dumps(details, ensure_ascii=False),
                    _now(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def list_calibrations(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM calibrations ORDER BY created_at DESC"
            ).fetchall()
        return [
            {**dict(r), "metrics": json.loads(r["metrics"]), "details": json.loads(r["details"])}
            for r in rows
        ]
