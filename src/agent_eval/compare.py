from __future__ import annotations

from collections import Counter

from .stats import classify_delta, mcnemar_exact, paired_bootstrap
from .storage import Storage


def analyze_run_results(
    rows_a: list[dict],
    rows_b: list[dict],
    *,
    min_delta: float = 0.05,
    min_cases: int = 50,
    alpha: float = 0.05,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    if min_cases < 1:
        raise ValueError("min_cases must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if n_boot < 1:
        raise ValueError("bootstrap must be at least 1")

    def pass_rates(rows: list[dict]) -> dict[str, float]:
        grouped: dict[str, list[bool]] = {}
        for row in rows:
            grouped.setdefault(row["case_id"], []).append(bool(row["passed"]))
        return {case_id: sum(values) / len(values) for case_id, values in grouped.items()}

    rates_a = pass_rates(rows_a)
    rates_b = pass_rates(rows_b)
    common = sorted(set(rates_a) & set(rates_b))
    values_a = [rates_a[case_id] for case_id in common]
    values_b = [rates_b[case_id] for case_id in common]
    mean, low, high = paired_bootstrap(values_a, values_b, n_boot, seed, alpha)
    verdict = classify_delta(mean, low, high, min_delta)
    gate_reasons = []
    if len(common) < min_cases:
        verdict = "inconclusive"
        gate_reasons.append(f"only {len(common)} common cases; require at least {min_cases}")
    sample_counts = [Counter(row["case_id"] for row in rows) for rows in (rows_a, rows_b)]
    single_sample = all(
        counts[case_id] == 1 for counts in sample_counts for case_id in common
    )
    regression_flips = sum(a == 1.0 and b == 0.0 for a, b in zip(values_a, values_b))
    improvement_flips = sum(a == 0.0 and b == 1.0 for a, b in zip(values_a, values_b))
    return {
        "verdict": verdict,
        "return_code": {"significant_regression": 2, "inconclusive": 3}.get(verdict, 0),
        "gate_reasons": gate_reasons,
        "metrics": {
            "common_cases": len(common),
            "mean_pass_rate_delta": mean,
            "confidence_level": 1 - alpha,
            "confidence_interval": [low, high],
            "min_delta": min_delta,
            "min_cases": min_cases,
            "bootstrap_samples": n_boot,
            "bootstrap_seed": seed,
            "inference": "paired_case_bootstrap",
            "mcnemar_regression_flips": regression_flips if single_sample else None,
            "mcnemar_improvement_flips": improvement_flips if single_sample else None,
            "mcnemar_p_value": mcnemar_exact(regression_flips, improvement_flips)
            if single_sample
            else None,
        },
        "cases": [
            {"id": case_id, "pass_rate_a": rates_a[case_id], "pass_rate_b": rates_b[case_id]}
            for case_id in common
        ],
    }


def _execution_map(storage: Storage, run_id: str) -> dict[str, dict]:
    return {e["source_key"]: e for e in storage.get_run_executions(run_id)}


def _validate_execution(execution: dict, rows: list[dict]) -> list[str]:
    source = execution["source_key"]
    errors = []
    if execution["status"] != "completed":
        errors.append(f"{source}: execution status is {execution['status']}, not completed")
    expected_cases = set(execution["case_ids"])
    if len(expected_cases) != len(execution["case_ids"]):
        errors.append(f"{source}: manifest contains duplicate case ids")
    actual_cases = {r["case_id"] for r in rows}
    if actual_cases != expected_cases:
        errors.append(
            f"{source}: case ids differ from manifest"
            f" (missing={sorted(expected_cases - actual_cases)},"
            f" unexpected={sorted(actual_cases - expected_cases)})"
        )
    expected_samples = execution["samples"]
    for case_id in sorted(expected_cases & actual_cases):
        sample_rows = [r for r in rows if r["case_id"] == case_id]
        indexes = Counter(r["sample_idx"] for r in sample_rows)
        expected_indexes = Counter(range(expected_samples))
        if indexes != expected_indexes:
            errors.append(
                f"{source}:{case_id}: expected sample indexes"
                f" {sorted(expected_indexes.elements())}, got {sorted(indexes.elements())}"
            )
    error_rows = [r for r in rows if r["status"] == "error"]
    if error_rows:
        errors.append(f"{source}: {len(error_rows)} sample(s) have execution errors")
    return errors


def validate_run_pair(storage: Storage, run_a: str, run_b: str) -> list[str]:
    errors = []
    executions_a = _execution_map(storage, run_a)
    executions_b = _execution_map(storage, run_b)
    for run_id, executions in ((run_a, executions_a), (run_b, executions_b)):
        if not executions:
            errors.append(
                f"run {run_id}: no execution manifest; rerun it with the current agent-eval"
            )
    if set(executions_a) != set(executions_b):
        errors.append(
            "execution sources differ"
            f" (missing from candidate={sorted(set(executions_a) - set(executions_b))},"
            f" missing from baseline={sorted(set(executions_b) - set(executions_a))})"
        )

    rows_a = storage.get_run_results(run_a)
    rows_b = storage.get_run_results(run_b)
    if not rows_a:
        errors.append(f"run {run_a}: no case results")
    if not rows_b:
        errors.append(f"run {run_b}: no case results")
    for source in sorted(set(executions_a) & set(executions_b)):
        a = executions_a[source]
        b = executions_b[source]
        own_a = [r for r in rows_a if r["case_id"] in set(a["case_ids"])]
        own_b = [r for r in rows_b if r["case_id"] in set(b["case_ids"])]
        errors.extend(_validate_execution(a, own_a))
        errors.extend(_validate_execution(b, own_b))
        for field in ("source_hash", "samples"):
            if a[field] != b[field]:
                errors.append(f"{source}: {field} differs between baseline and candidate")
        for field in ("judge", "simulated_user"):
            if a["metadata"][field] != b["metadata"][field]:
                errors.append(f"{source}: {field} differs between baseline and candidate")
        if set(a["case_ids"]) != set(b["case_ids"]):
            errors.append(f"{source}: case ids differ between baseline and candidate")

    case_ids_a = {r["case_id"] for r in rows_a}
    case_ids_b = {r["case_id"] for r in rows_b}
    if case_ids_a != case_ids_b:
        errors.append(
            "run case sets differ"
            f" (only baseline={sorted(case_ids_a - case_ids_b)},"
            f" only candidate={sorted(case_ids_b - case_ids_a)})"
        )
    return errors


def validate_calibrations(storage: Storage, executions: list[dict]) -> tuple[list[str], list[dict]]:
    errors = []
    checks = []
    seen = set()
    for execution in executions:
        source = execution["source_key"]
        config = execution["metadata"]["judge"]
        if not config.get("backend"):
            checks.append({"source": source, "required": False})
            continue
        key = (config["backend"], config["model"], config["prompt_sha256"])
        if key in seen:
            continue
        seen.add(key)
        calibration = storage.latest_calibration(config)
        check = {"source": source, "required": True, "judge": config}
        if calibration is None:
            errors.append(f"{source}: no calibration for judge {config['backend']}:{config['model']}")
            check["accepted"] = False
        else:
            check.update(
                {
                    "calibration_id": calibration["id"],
                    "golden_hash": calibration["golden_hash"],
                    "accepted": calibration["admission"].get("accepted") is True,
                    "admission": calibration["admission"],
                }
            )
            if not check["accepted"]:
                errors.append(
                    f"{source}: latest matching calibration #{calibration['id']} was not admitted"
                )
        checks.append(check)
    return errors, checks
