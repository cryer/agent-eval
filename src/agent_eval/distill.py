"""Distill recorded traces back into regression cases.

Hand-writing hundreds of cases does not scale; the case layer is fed by
mining the spans table instead. Every `run` already stores one trace per
(case, sample): the root `case.<id>` span carries the input and final output,
and child `tool.*` spans carry the tool-call sequence. This module rebuilds
case entries from what actually happened — tool calls become `tool_called`
assertions, semantic assertions (output_contains / judge) are left for a
human to add on top.

Scope: traces rooted at a `case.*` span (i.e. recorded by `run`). Scenario
rollouts and standalone agent traces are not distilled (yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import yaml

from .storage import Storage


@dataclass
class DistilledCase:
    id: str
    input: str
    output: str
    tools: list[str]
    trace_id: str
    start_time: int  # epoch ns

    def to_yaml_entry(self) -> dict:
        return {
            "id": self.id,
            "input": self.input,
            "assertions": [{"tool_called": t} for t in self.tools],
        }


def parse_since(value: str) -> int:
    """ISO date/datetime -> epoch nanoseconds. Naive values are treated as UTC."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def distill_cases(
    storage: Storage,
    since_ns: int | None = None,
    limit: int = 50,
    prompt_version: str | None = None,
) -> tuple[list[DistilledCase], int]:
    """Mine case traces into DistilledCase entries, newest first.

    Returns (cases, scanned) where scanned counts eligible case traces seen
    (traces whose root lacks `case.input` — recorded before that attribute
    existed — are skipped before scanning). k samples of the same case
    collapse into one entry: dedupe is on (case.id, input).
    """
    cases: list[DistilledCase] = []
    seen_inputs: set[tuple[str, str]] = set()
    id_counts: dict[str, int] = {}
    scanned = 0
    for root in storage.list_trace_roots(since_ns=since_ns):
        if not root["name"].startswith("case."):
            continue
        attrs = root["attributes"]
        case_id = attrs.get("case.id")
        input_text = attrs.get("case.input")
        if case_id is None or input_text is None:
            continue
        scanned += 1
        key = (str(case_id), str(input_text))
        if key in seen_inputs:
            continue
        spans = storage.get_spans(root["trace_id"])
        if prompt_version is not None and not any(
            s["attributes"].get("prompt.version") == prompt_version for s in spans
        ):
            continue
        seen_inputs.add(key)
        tools: list[str] = []
        for s in spans:
            name = s["attributes"].get("tool.name")
            if name and name not in tools:
                tools.append(str(name))
        n = id_counts.get(key[0], 0) + 1
        id_counts[key[0]] = n
        out_id = key[0] if n == 1 else f"{key[0]}_v{n}"
        cases.append(
            DistilledCase(
                id=out_id,
                input=key[1],
                output=str(attrs.get("case.output", "")),
                tools=tools,
                trace_id=root["trace_id"],
                start_time=root["start_time"],
            )
        )
        if len(cases) >= limit:
            break
    return cases, scanned


def render_cases_yaml(cases: list[DistilledCase]) -> str:
    """Render distilled cases as a cases YAML file, one comment per entry
    pointing back at the source trace."""
    lines = [
        "# Distilled from recorded traces by `agent-eval distill`.",
        "# tool_called assertions are derived from the observed tool-call sequence;",
        "# add semantic assertions (output_contains / judge) before gating on these.",
    ]
    for c in cases:
        ts = datetime.fromtimestamp(c.start_time / 1e9, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        lines.append("")
        lines.append(f"# trace {c.trace_id} · {ts}")
        lines.append(
            yaml.safe_dump([c.to_yaml_entry()], allow_unicode=True, sort_keys=False).rstrip("\n")
        )
    return "\n".join(lines) + "\n"
