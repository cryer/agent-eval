from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from .calibration import load_golden, run_calibration, store_calibration
from .diff import diff_runs
from .judge import Judge, build_backend
from .runner import load_cases, run_cases
from .simulation import load_scenarios, run_scenarios
from .stats import mcnemar_exact, paired_bootstrap
from .storage import Storage
from .tracing import init_tracing


def _load_dotenv(path: str | Path = ".env") -> list[str]:
    """Minimal .env loader so local runs don't need manual `export`.

    KEY=VALUE lines, `#` comments, optional surrounding quotes. Existing
    environment variables win — .env only fills in what is missing.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    loaded = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


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


def _build_judge(args) -> Judge | None:
    backend_name = getattr(args, "judge_backend", None)
    if backend_name is None:
        return None
    backend = build_backend(backend_name, model=getattr(args, "judge_model", "gpt-4o-mini"))
    label = backend_name if backend_name == "stub" else f"litellm:{args.judge_model}"
    return Judge(backend, backend_name=label)


def _add_judge_args(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument(
        "--judge-backend",
        choices=["litellm", "stub"],
        default=None if not required else "litellm",
        required=required,
        help="judge backend: litellm (real model) or stub (offline, deterministic)",
    )
    p.add_argument("--judge-model", default="gpt-4o-mini", help="model for the litellm backend")


def _merge_paths(existing: str | None, new: str) -> str:
    if not existing:
        return new
    return existing if new in existing.split(" + ") else f"{existing} + {new}"


def _print_summary(summary, agent_spec: str, source: str) -> None:
    print(
        f"run {summary.run_id}  agent={agent_spec}  {source}"
        f"  samples={summary.samples_per_case}\n"
    )
    for report in summary.reports:
        status = "PASS" if report.passed else "FAIL"
        n_ok = sum(1 for s in report.samples if s.passed)
        sample_info = f"samples {n_ok}/{len(report.samples)}, pass_rate={report.pass_rate:.2f}"
        print(
            f"{status}  {report.case.id}  ({sample_info},"
            f" {report.duration_ms:.1f}ms, ${report.cost_usd:.6f})"
        )
        for s in report.samples:
            for r in s.results:
                if not r.passed:
                    print(f"      x [sample {s.sample_idx}] {r.name}: {r.detail}")
    print(
        f"\n{summary.passed}/{summary.total} cases passed"
        f"  (mean pass rate {summary.mean_pass_rate:.2f})"
    )


def _cmd_run(args) -> int:
    agent_fn = _load_agent(args.agent)
    judge = _build_judge(args)
    storage = Storage(args.db)
    try:
        tracer = init_tracing(storage)
        cases = load_cases(args.cases, judge=judge)
        if args.run_id:
            storage.delete_run_results(args.run_id, case_ids=[c.id for c in cases])
        summary = run_cases(
            agent_fn, cases, storage, tracer, run_id=args.run_id, samples=args.samples
        )
        existing = storage.get_run(summary.run_id)
        total, passed = storage.run_totals(summary.run_id)
        storage.record_run(
            run_id=summary.run_id,
            agent=args.agent,
            cases_path=_merge_paths(existing["cases_path"] if existing else None, args.cases),
            total=total,
            passed=passed,
        )
        _print_summary(summary, args.agent, f"cases={args.cases}")
        return 0 if summary.ok else 1
    finally:
        storage.close()


def _cmd_simulate(args) -> int:
    agent_fn = _load_agent(args.agent)
    judge = _build_judge(args)
    storage = Storage(args.db)
    try:
        tracer = init_tracing(storage)
        scenarios = load_scenarios(args.scenarios, judge=judge)
        user_backend = (
            None if args.user_backend == "stub" else build_backend("litellm", args.user_model)
        )
        if args.run_id:
            storage.delete_run_results(args.run_id, case_ids=[s.id for s in scenarios])
        summary = run_scenarios(
            agent_fn,
            scenarios,
            storage,
            tracer,
            user_backend=user_backend,
            run_id=args.run_id,
            samples=args.samples,
        )
        existing = storage.get_run(summary.run_id)
        total, passed = storage.run_totals(summary.run_id)
        storage.record_run(
            run_id=summary.run_id,
            agent=args.agent,
            cases_path=_merge_paths(existing["cases_path"] if existing else None, args.scenarios),
            total=total,
            passed=passed,
        )
        _print_summary(summary, args.agent, f"scenarios={args.scenarios}")
        return 0 if summary.ok else 1
    finally:
        storage.close()


def _cmd_calibrate(args) -> int:
    judge = _build_judge(args)
    storage = Storage(args.db)
    try:
        items = load_golden(args.golden)
        report = run_calibration(items, judge)
        cal_id = store_calibration(storage, report, judge.backend_name, args.golden)

        print(f"calibration #{cal_id}  judge={judge.backend_name}  golden={args.golden}\n")
        for r in report.items:
            mark = "ok " if r.agree else "DISAGREE"
            print(
                f"  {mark}  {r.item.id}  human={'pass' if r.item.label else 'fail'}"
                f"  judge={'pass' if r.verdict.passed else 'fail'} (score={r.verdict.score:.2f})"
            )
            if not r.agree:
                print(f"        rationale: {r.verdict.rationale}")
                if r.item.note:
                    print(f"        human note: {r.item.note}")

        m = report.metrics
        print(
            f"\njudge vs human:  accuracy={m['accuracy']:.2f}  kappa={m['kappa']:.2f}"
            f"  fpr={m['fpr']:.2f}  fnr={m['fnr']:.2f}"
            f"  (tp={int(m['tp'])} tn={int(m['tn'])} fp={int(m['fp'])} fn={int(m['fn'])})"
        )
        if m["kappa"] < 0.6:
            print(
                "WARNING: kappa < 0.6 — this judge disagrees with humans too often;"
                " do not trust it as a CI gate until its prompt is improved."
            )
        return 0
    finally:
        storage.close()


def _cmd_compare(args) -> int:
    storage = Storage(args.db)
    try:
        rows_a = storage.get_run_results(args.run_a)
        rows_b = storage.get_run_results(args.run_b)
        if not rows_a or not rows_b:
            missing = args.run_a if not rows_a else args.run_b
            print(f"no case results for run {missing}")
            return 1

        def per_case_pass_rate(rows):
            rates: dict[str, list[bool]] = {}
            for r in rows:
                rates.setdefault(r["case_id"], []).append(bool(r["passed"]))
            return {cid: sum(v) / len(v) for cid, v in rates.items()}

        rates_a = per_case_pass_rate(rows_a)
        rates_b = per_case_pass_rate(rows_b)
        common = sorted(set(rates_a) & set(rates_b))
        if not common:
            print("runs have no cases in common")
            return 1

        va = [rates_a[c] for c in common]
        vb = [rates_b[c] for c in common]
        b_flips = sum(1 for a, b in zip(va, vb) if a == 1.0 and b < 1.0)
        c_flips = sum(1 for a, b in zip(va, vb) if a < 1.0 and b == 1.0)
        p_value = mcnemar_exact(b_flips, c_flips)
        mean_diff, lo, hi = paired_bootstrap(va, vb)

        print(f"compare  A={args.run_a} -> B={args.run_b}  ({len(common)} common cases)\n")
        for cid in common:
            arrow = "" if rates_a[cid] == rates_b[cid] else "  <- changed"
            print(f"  {cid}  {rates_a[cid]:.2f} -> {rates_b[cid]:.2f}{arrow}")
        print(
            f"\nmean pass-rate delta (B-A): {mean_diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
            f"\nMcNemar: {b_flips} regression flips vs {c_flips} improvement flips,"
            f" p={p_value:.4f}"
        )
        significant = p_value < args.alpha or (mean_diff < 0 and hi < 0) or (mean_diff > 0 and lo > 0)
        if mean_diff < 0 and significant:
            print(f"verdict: SIGNIFICANT REGRESSION (alpha={args.alpha}) — gate would BLOCK")
            return 2
        if mean_diff > 0 and significant:
            print(f"verdict: SIGNIFICANT IMPROVEMENT (alpha={args.alpha})")
        else:
            print(f"verdict: no significant change (alpha={args.alpha}) — gate would PASS")
        return 0
    finally:
        storage.close()


def _cmd_diff(args) -> int:
    storage = Storage(args.db)
    try:
        diffs = diff_runs(storage, args.run_a, args.run_b)
        if not diffs:
            print(f"runs {args.run_a} and {args.run_b} have no cases in common")
            return 1
        changed = [d for d in diffs if d.changed]
        print(f"trajectory diff  A={args.run_a} -> B={args.run_b}\n")
        for d in changed:
            print(f"case {d.case_id}  pass_rate {d.pass_rate_a:.2f} -> {d.pass_rate_b:.2f}")
            if d.tool_diff:
                print("  tool call sequence:")
                for line in d.tool_diff:
                    print(f"    {line}")
            if d.output_diff:
                print("  output:")
                for line in d.output_diff:
                    print(f"    {line}")
            print()
        unchanged = len(diffs) - len(changed)
        print(f"{len(changed)} cases changed, {unchanged} unchanged")
        return 0
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
    run_p.add_argument("--samples", type=int, default=1, help="samples per case (pass^k)")
    run_p.add_argument(
        "--run-id",
        default=None,
        help="pin the run id (e.g. baseline/candidate in CI); re-running replaces old results",
    )
    _add_judge_args(run_p, required=False)
    run_p.set_defaults(func=_cmd_run)

    sim_p = sub.add_parser(
        "simulate", help="multi-turn rollout against persona-based simulated users"
    )
    sim_p.add_argument("--scenarios", required=True, help="path to scenarios YAML file")
    sim_p.add_argument("--agent", required=True, help="agent callable as module:attr; takes message history")
    sim_p.add_argument("--db", default="eval.db")
    sim_p.add_argument("--samples", type=int, default=1, help="rollouts per scenario (pass^k)")
    sim_p.add_argument("--run-id", default=None)
    sim_p.add_argument(
        "--user-backend",
        choices=["stub", "litellm"],
        default="stub",
        help="stub = scripted personas only; litellm = LLM plays the user",
    )
    sim_p.add_argument("--user-model", default="gpt-4o-mini")
    _add_judge_args(sim_p, required=False)
    sim_p.set_defaults(func=_cmd_simulate)

    diff_p = sub.add_parser("diff", help="trajectory diff between two runs")
    diff_p.add_argument("run_a", help="baseline run id")
    diff_p.add_argument("run_b", help="candidate run id")
    diff_p.add_argument("--db", default="eval.db")
    diff_p.set_defaults(func=_cmd_diff)

    cal_p = sub.add_parser("calibrate", help="measure judge agreement with human labels")
    cal_p.add_argument("--golden", required=True, help="path to golden set YAML file")
    cal_p.add_argument("--db", default="eval.db")
    _add_judge_args(cal_p, required=True)
    cal_p.set_defaults(func=_cmd_calibrate)

    cmp_p = sub.add_parser("compare", help="statistical comparison of two runs (gate logic)")
    cmp_p.add_argument("run_a", help="baseline run id")
    cmp_p.add_argument("run_b", help="candidate run id")
    cmp_p.add_argument("--alpha", type=float, default=0.05)
    cmp_p.add_argument("--db", default="eval.db")
    cmp_p.set_defaults(func=_cmd_compare)

    hist_p = sub.add_parser("history", help="list recorded runs")
    hist_p.add_argument("--db", default="eval.db")
    hist_p.set_defaults(func=_cmd_history)

    trace_p = sub.add_parser("trace", help="dump spans of one trace")
    trace_p.add_argument("trace_id")
    trace_p.add_argument("--db", default="eval.db")
    trace_p.set_defaults(func=_cmd_trace)

    args = parser.parse_args(argv)
    _load_dotenv()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
