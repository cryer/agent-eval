from __future__ import annotations

import argparse
import importlib
import json
import sys

from .runner import load_cases, run_cases
from .storage import Storage
from .tracing import init_tracing


def _load_agent(spec: str):
    """Load 'module:callable', e.g. examples.toy_agent:agent."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"--agent must be 'module:callable', got {spec!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ValueError(f"{spec!r} does not resolve to a callable")
    return fn


def _cmd_run(args) -> int:
    agent_fn = _load_agent(args.agent)
    storage = Storage(args.db)
    try:
        tracer = init_tracing(storage)
        cases = load_cases(args.cases)
        summary = run_cases(agent_fn, cases, storage, tracer)
        storage.record_run(
            run_id=summary.run_id,
            agent=args.agent,
            cases_path=args.cases,
            total=summary.total,
            passed=summary.passed,
        )

        print(f"run {summary.run_id}  agent={args.agent}  cases={args.cases}\n")
        for report in summary.reports:
            status = "PASS" if report.passed else "FAIL"
            n_ok = sum(1 for r in report.results if r.passed)
            print(
                f"{status}  {report.case.id}"
                f"  ({n_ok}/{len(report.results)} assertions,"
                f" {report.duration_ms:.1f}ms, ${report.cost_usd:.6f})"
                f"  trace={report.trace_id}"
            )
            for r in report.results:
                if not r.passed:
                    print(f"      x {r.name}: {r.detail}")
        print(f"\n{summary.passed}/{summary.total} cases passed")
        return 0 if summary.ok else 1
    finally:
        storage.close()


def _cmd_history(args) -> int:
    storage = Storage(args.db)
    try:
        runs = storage.list_runs()
        if not runs:
            print("no runs recorded")
            return 0
        for r in runs:
            print(
                f"{r['created_at']}  run={r['run_id']}  agent={r['agent']}"
                f"  passed={r['passed']}/{r['total']}"
            )
        return 0
    finally:
        storage.close()


def _cmd_trace(args) -> int:
    storage = Storage(args.db)
    try:
        spans = storage.get_spans(args.trace_id)
        if not spans:
            print(f"no spans for trace {args.trace_id}")
            return 1
        for s in spans:
            duration_ms = (s["end_time"] - s["start_time"]) / 1e6
            depth = 0 if s["parent_span_id"] is None else 1
            indent = "  " * depth
            attrs = json.dumps(s["attributes"], ensure_ascii=False)
            print(f"{indent}{s['name']}  [{duration_ms:.1f}ms, {s['status']}]")
            if s["attributes"]:
                print(f"{indent}  {attrs}")
        return 0
    finally:
        storage.close()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description="Regression evaluation harness for LLM agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run cases against an agent and evaluate assertions")
    run_p.add_argument("--cases", required=True, help="path to cases YAML file")
    run_p.add_argument("--agent", required=True, help="agent callable as module:attr")
    run_p.add_argument("--db", default="eval.db", help="SQLite trace DB path")
    run_p.set_defaults(func=_cmd_run)

    hist_p = sub.add_parser("history", help="list recorded runs")
    hist_p.add_argument("--db", default="eval.db")
    hist_p.set_defaults(func=_cmd_history)

    trace_p = sub.add_parser("trace", help="dump spans of one trace")
    trace_p.add_argument("trace_id")
    trace_p.add_argument("--db", default="eval.db")
    trace_p.set_defaults(func=_cmd_trace)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
