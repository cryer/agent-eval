from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import uuid
from pathlib import Path

from .calibration import (
    calibration_details,
    evaluate_admission,
    load_golden,
    run_calibration,
    store_calibration,
)
from .compare import analyze_run_results, validate_calibrations, validate_run_pair
from .diff import diff_runs
from .distill import distill_cases, parse_since, render_cases_yaml
from .judge import Judge, build_backend
from .manifest import build_manifest, judge_config
from .reporting import summary_report, write_json, write_junit
from .runner import load_cases, run_cases
from .simulation import load_scenarios, run_scenarios
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
        has_error = any(s.status == "error" for s in report.samples)
        status = "ERROR" if has_error else "PASS" if report.passed else "FAIL"
        n_ok = sum(1 for s in report.samples if s.passed)
        sample_info = f"samples {n_ok}/{len(report.samples)}, pass_rate={report.pass_rate:.2f}"
        print(
            f"{status}  {report.case.id}  ({sample_info},"
            f" {report.duration_ms:.1f}ms, ${report.cost_usd:.6f})"
        )
        for s in report.samples:
            if s.status == "error":
                print(f"      ! [sample {s.sample_idx}] execution error: {s.error}")
            for r in s.results:
                if not r.passed:
                    print(f"      x [sample {s.sample_idx}] {r.name}: {r.detail}")
    print(
        f"\n{summary.passed}/{summary.total} cases passed"
        f"  (mean pass rate {summary.mean_pass_rate:.2f},"
        f" execution errors {summary.error_count})"
    )


def _write_summary_reports(args, summary, manifest: dict, source: str, status: str) -> None:
    report = summary_report(summary, manifest, source)
    report["execution_status"] = status
    if args.json_out:
        write_json(args.json_out, report)
    if args.junit_out:
        write_junit(args.junit_out, report)


def _record_observed_agent_metadata(summary, manifest: dict, storage: Storage) -> None:
    models = set()
    prompt_versions = set()
    for report in summary.reports:
        for sample in report.samples:
            for span in storage.get_spans(sample.trace_id):
                model = span["attributes"].get("gen_ai.request.model")
                prompt_version = span["attributes"].get("prompt.version")
                if model:
                    models.add(str(model))
                if prompt_version:
                    prompt_versions.add(str(prompt_version))
    manifest["agent_models"] = sorted(models)
    manifest["agent_prompt_versions"] = sorted(prompt_versions)
    storage.update_execution_metadata(manifest)


def _cmd_run(args) -> int:
    agent_fn = _load_agent(args.agent)
    judge = _build_judge(args)
    storage = Storage(args.db)
    manifest = None
    try:
        tracer = init_tracing(storage)
        cases = load_cases(args.cases, judge=judge)
        run_id = args.run_id or uuid.uuid4().hex[:8]
        storage.delete_run_results(run_id, case_ids=[c.id for c in cases])
        manifest = build_manifest(
            run_id=run_id,
            kind="case",
            source_path=args.cases,
            case_ids=[c.id for c in cases],
            samples=args.samples,
            concurrency=args.concurrency,
            agent=args.agent,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            env_paths=[c.env for c in cases if c.env],
        )
        storage.begin_execution(manifest)
        summary = run_cases(
            agent_fn,
            cases,
            storage,
            tracer,
            run_id=run_id,
            samples=args.samples,
            concurrency=args.concurrency,
        )
        execution_status = "error" if summary.error_count else "completed"
        _record_observed_agent_metadata(summary, manifest, storage)
        storage.finish_execution(manifest["execution_id"], execution_status)
        existing = storage.get_run(summary.run_id)
        total, passed = storage.run_totals(summary.run_id)
        storage.record_run(
            run_id=summary.run_id,
            agent=args.agent,
            cases_path=_merge_paths(existing["cases_path"] if existing else None, args.cases),
            total=total,
            passed=passed,
        )
        _write_summary_reports(args, summary, manifest, f"cases={args.cases}", execution_status)
        _print_summary(summary, args.agent, f"cases={args.cases}")
        return 3 if summary.error_count else 0 if summary.ok else 1
    except Exception as exc:
        if manifest is not None:
            storage.finish_execution(
                manifest["execution_id"], "incomplete", f"{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        storage.close()


def _cmd_simulate(args) -> int:
    agent_fn = _load_agent(args.agent)
    judge = _build_judge(args)
    storage = Storage(args.db)
    manifest = None
    try:
        tracer = init_tracing(storage)
        scenarios = load_scenarios(args.scenarios, judge=judge)
        user_backend = (
            None if args.user_backend == "stub" else build_backend("litellm", args.user_model)
        )
        run_id = args.run_id or uuid.uuid4().hex[:8]
        storage.delete_run_results(run_id, case_ids=[s.id for s in scenarios])
        manifest = build_manifest(
            run_id=run_id,
            kind="scenario",
            source_path=args.scenarios,
            case_ids=[s.id for s in scenarios],
            samples=args.samples,
            concurrency=args.concurrency,
            agent=args.agent,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            user_backend=args.user_backend,
            user_model=args.user_model,
        )
        storage.begin_execution(manifest)
        summary = run_scenarios(
            agent_fn,
            scenarios,
            storage,
            tracer,
            user_backend=user_backend,
            run_id=run_id,
            samples=args.samples,
            concurrency=args.concurrency,
        )
        execution_status = "error" if summary.error_count else "completed"
        _record_observed_agent_metadata(summary, manifest, storage)
        storage.finish_execution(manifest["execution_id"], execution_status)
        existing = storage.get_run(summary.run_id)
        total, passed = storage.run_totals(summary.run_id)
        storage.record_run(
            run_id=summary.run_id,
            agent=args.agent,
            cases_path=_merge_paths(existing["cases_path"] if existing else None, args.scenarios),
            total=total,
            passed=passed,
        )
        _write_summary_reports(
            args, summary, manifest, f"scenarios={args.scenarios}", execution_status
        )
        _print_summary(summary, args.agent, f"scenarios={args.scenarios}")
        return 3 if summary.error_count else 0 if summary.ok else 1
    except Exception as exc:
        if manifest is not None:
            storage.finish_execution(
                manifest["execution_id"], "incomplete", f"{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        storage.close()


def _cmd_calibrate(args) -> int:
    judge = _build_judge(args)
    storage = Storage(args.db)
    try:
        items = load_golden(args.golden)
        report = run_calibration(items, judge)
        admission = evaluate_admission(
            report.metrics,
            min_samples=args.min_samples,
            min_per_class=args.min_per_class,
            min_kappa=args.min_kappa,
            max_fpr=args.max_fpr,
        )
        config = judge_config(args.judge_backend, args.judge_model)
        cal_id = store_calibration(
            storage,
            report,
            judge.backend_name,
            args.golden,
            judge_config=config,
            admission=admission,
        )

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
        accepted = admission["accepted"]
        print(f"admission: {'ADMITTED' if accepted else 'REJECTED'}")
        for reason in admission["reasons"]:
            print(f"  - {reason}")
        if args.json_out:
            stored = storage.list_calibrations()[0]
            write_json(
                args.json_out,
                {
                    "schema_version": 1,
                    "calibration_id": cal_id,
                    "judge": config,
                    "golden_path": args.golden,
                    "golden_hash": stored["golden_hash"],
                    "metrics": report.metrics,
                    "admission": admission,
                    "items": calibration_details(report),
                },
            )
        return 0 if accepted else 4
    finally:
        storage.close()


def _cmd_compare(args) -> int:
    storage = Storage(args.db)
    report = {
        "schema_version": 2,
        "run_a": args.run_a,
        "run_b": args.run_b,
        "alpha": args.alpha,
        "valid": False,
        "verdict": "invalid",
        "validation_errors": [],
    }
    try:
        executions_a = storage.get_run_executions(args.run_a)
        executions_b = storage.get_run_executions(args.run_b)
        validation_errors = validate_run_pair(storage, args.run_a, args.run_b)
        report["calibration"] = {"required": args.require_calibration, "checks": []}
        if not validation_errors and args.require_calibration:
            calibration_errors, checks = validate_calibrations(storage, executions_a)
            validation_errors.extend(calibration_errors)
            report["calibration"]["checks"] = checks
        report["validation_errors"] = validation_errors
        if validation_errors:
            print(f"compare INVALID: {args.run_a} -> {args.run_b}")
            for error in validation_errors:
                print(f"  - {error}")
            print("verdict: INVALID — gate would FAIL; no statistical comparison was performed")
            if args.json_out:
                write_json(args.json_out, report)
            return 1

        analysis = analyze_run_results(
            storage.get_run_results(args.run_a),
            storage.get_run_results(args.run_b),
            min_delta=args.min_delta,
            min_cases=args.min_cases,
            alpha=args.alpha,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        report.update(analysis)
        report["valid"] = True
        report["executions"] = {
            args.run_a: storage.get_run_executions(args.run_a),
            args.run_b: storage.get_run_executions(args.run_b),
        }
        metrics = analysis["metrics"]
        print(f"compare  A={args.run_a} -> B={args.run_b}  ({metrics['common_cases']} common cases)\n")
        for case in analysis["cases"]:
            arrow = "" if case["pass_rate_a"] == case["pass_rate_b"] else "  <- changed"
            print(
                f"  {case['id']}  {case['pass_rate_a']:.2f} ->"
                f" {case['pass_rate_b']:.2f}{arrow}"
            )
        mean = metrics["mean_pass_rate_delta"]
        low, high = metrics["confidence_interval"]
        confidence = metrics["confidence_level"] * 100
        print(
            f"\nmean pass-rate delta (B-A): {mean:+.3f}  {confidence:g}% CI"
            f" [{low:+.3f}, {high:+.3f}]  min_delta={args.min_delta:.3f}"
        )
        if metrics["mcnemar_p_value"] is not None:
            print(
                f"McNemar (diagnostic only): {metrics['mcnemar_regression_flips']} regression"
                f" flips vs {metrics['mcnemar_improvement_flips']} improvement flips,"
                f" p={metrics['mcnemar_p_value']:.4f}"
            )
        else:
            print("McNemar: not applicable to multi-sample rates; cases are bootstrap clusters")
        verdict = analysis["verdict"]
        if verdict == "significant_regression":
            print("verdict: SIGNIFICANT REGRESSION — gate would BLOCK")
        elif verdict == "significant_improvement":
            print("verdict: SIGNIFICANT IMPROVEMENT — gate would PASS")
        elif verdict == "no_meaningful_regression":
            print("verdict: NO MEANINGFUL REGRESSION — gate would PASS")
        else:
            print("verdict: INCONCLUSIVE — gate would BLOCK; collect more evidence")
        for reason in analysis["gate_reasons"]:
            print(f"  - {reason}")
        if args.json_out:
            write_json(args.json_out, report)
        return analysis["return_code"]
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


def _cmd_distill(args) -> int:
    storage = Storage(args.db)
    try:
        since_ns = parse_since(args.since) if args.since else None
        cases, scanned = distill_cases(
            storage,
            since_ns=since_ns,
            limit=args.limit,
            prompt_version=args.prompt_version,
        )
        if not cases:
            print(f"no distillable traces found (scanned {scanned} case traces)")
            return 1
        Path(args.out).write_text(render_cases_yaml(cases), encoding="utf-8")
        print(f"distilled {len(cases)} cases from {scanned} case traces -> {args.out}\n")
        for c in cases:
            tools = ", ".join(c.tools) if c.tools else "(no tool calls)"
            preview = c.input.replace("\n", " ")[:50]
            print(f"  {c.id}  tools: {tools}  input: {preview}")
        print("\ntool_called assertions are derived; add semantic assertions before gating.")
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
        "--concurrency",
        type=int,
        default=1,
        help="parallel workers; bound it to your model plan's rate limit"
        " (429s are retried with backoff, but fewer workers = less throttling)",
    )
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
    sim_p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel workers; bound it to your model plan's rate limit"
        " (429s are retried with backoff, but fewer workers = less throttling)",
    )
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
    cal_p.add_argument("--min-samples", type=int, default=50)
    cal_p.add_argument("--min-per-class", type=int, default=10)
    cal_p.add_argument("--min-kappa", type=float, default=0.6)
    cal_p.add_argument("--max-fpr", type=float, default=0.05)
    cal_p.add_argument("--json-out", help="write calibration metrics and admission as JSON")
    _add_judge_args(cal_p, required=True)
    cal_p.set_defaults(func=_cmd_calibrate)

    cmp_p = sub.add_parser("compare", help="statistical comparison of two runs (gate logic)")
    cmp_p.add_argument("run_a", help="baseline run id")
    cmp_p.add_argument("run_b", help="candidate run id")
    cmp_p.add_argument("--alpha", type=float, default=0.05)
    cmp_p.add_argument(
        "--min-delta",
        type=float,
        default=0.05,
        help="smallest pass-rate regression that matters (non-inferiority margin)",
    )
    cmp_p.add_argument(
        "--min-cases",
        type=int,
        default=50,
        help="minimum paired cases required for a conclusive gate",
    )
    cmp_p.add_argument("--bootstrap", type=int, default=10000, help="paired bootstrap resamples")
    cmp_p.add_argument("--seed", type=int, default=0, help="bootstrap random seed")
    cmp_p.add_argument(
        "--require-calibration",
        action="store_true",
        help="require an admitted calibration for every judge configuration",
    )
    cmp_p.add_argument("--db", default="eval.db")
    cmp_p.set_defaults(func=_cmd_compare)

    hist_p = sub.add_parser("history", help="list recorded runs")
    hist_p.add_argument("--db", default="eval.db")
    hist_p.set_defaults(func=_cmd_history)

    dist_p = sub.add_parser("distill", help="distill recorded traces into new regression cases")
    dist_p.add_argument("--out", default="cases_distilled.yaml", help="output cases YAML path")
    dist_p.add_argument("--db", default="eval.db")
    dist_p.add_argument("--limit", type=int, default=50, help="max cases to distill")
    dist_p.add_argument(
        "--since",
        default=None,
        help="only traces at/after this ISO date/datetime (UTC if naive)",
    )
    dist_p.add_argument(
        "--prompt-version",
        default=None,
        help="only traces containing this prompt.version hash",
    )
    dist_p.set_defaults(func=_cmd_distill)

    trace_p = sub.add_parser("trace", help="dump spans of one trace")
    trace_p.add_argument("trace_id")
    trace_p.add_argument("--db", default="eval.db")
    trace_p.set_defaults(func=_cmd_trace)

    for command_parser in (run_p, sim_p):
        command_parser.add_argument("--json-out", help="write a machine-readable JSON report")
        command_parser.add_argument("--junit-out", help="write a JUnit XML report")
    cmp_p.add_argument("--json-out", help="write comparison verdict and metrics as JSON")

    args = parser.parse_args(argv)
    if args.command == "compare":
        if not 0 < args.alpha < 1:
            parser.error("--alpha must be between 0 and 1")
        if not 0 <= args.min_delta <= 1:
            parser.error("--min-delta must be between 0 and 1")
        if args.min_cases < 1:
            parser.error("--min-cases must be at least 1")
        if args.bootstrap < 1:
            parser.error("--bootstrap must be at least 1")
    if args.command == "calibrate":
        if args.min_samples < 1 or args.min_per_class < 1:
            parser.error("calibration sample thresholds must be at least 1")
        if not -1 <= args.min_kappa <= 1:
            parser.error("--min-kappa must be between -1 and 1")
        if not 0 <= args.max_fpr <= 1:
            parser.error("--max-fpr must be between 0 and 1")
    _load_dotenv()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
