# agent-eval

**CI for LLM agents** — change the prompt, swap the model, edit a tool,
refactor the loop: replay the eval suite and let statistics decide whether
anything got worse, or whether it's just noise.

English | [中文](README.zh-CN.md)

---

## Why

Agents break silently, and not just when you touch the prompt — swapping a
model version, editing a tool schema, or refactoring the agent loop can shift
behavior just as much. Unit tests don't help: agent behavior is probabilistic,
multi-turn, and tool-driven. `agent-eval` turns "did this change make things
worse?" into a CI gate with a yes/no answer backed by statistical tests —
whatever layer the change came from.

```
any agent change → replay baseline/candidate → calibrated clustered bootstrap gate → merge or block
```

## Features

- **Observability first** — every run is traced with OpenTelemetry spans
  (LLM calls, tool calls, token cost, prompt version hash) and persisted to
  SQLite. Trajectory diffs show *where* behavior changed, not just *that* it did.
- **Two assertion layers** — cheap deterministic rules (output content, tool
  usage, turn/cost/latency budgets, workspace state) plus LLM-as-judge for
  semantic quality.
- **Judge calibration admission** — accuracy / Cohen's kappa / FPR / FNR against
  human labels, with defaults of 50 samples, 10 per class, kappa >= 0.6 and
  FPR <= 0.05. `compare --require-calibration` checks the latest admitted record
  for the exact judge backend, model and prompt hash.
- **Simulated users with hidden goals** — multi-turn rollouts against personas,
  including adversarial ones (e.g. a user who plays nice while fishing for your
  system prompt). Rule assertions backstop the LLM judge so a lenient simulated
  user can't let a leak slip through.
- **Coding agents** — cases can carry a repo-snapshot fixture; each sample runs
  in a fresh copy and assertions check the final workspace state
  (`command_exit`, diff size limits) — the SWE-bench FAIL_TO_PASS pattern.
- **Statistical gate** — bootstrap whole cases, not individual correlated samples.
  A minimum important regression margin and confidence interval yield pass,
  regression or inconclusive; single-sample McNemar is diagnostic only.
- **Trustworthy run manifests** — Git, dataset, model and judge configuration are
  recorded; execution errors stay distinct from assertion failures. `compare`
  rejects incomplete or incompatible runs before doing any statistics.
- **CI reports** — JSON and JUnit XML for runs, JSON for comparisons/calibration,
  plus uploaded trajectory diffs and the SQLite database as workflow artifacts.
- **Parallel + rate-limit tolerant** — `--concurrency N` with exponential
  backoff on 429/5xx, so a rate-limited model plan slows the run instead of
  failing it.
- **GitHub Actions ready** — the included workflow runs the target branch as
  baseline and the PR as candidate, then blocks on regression or inconclusive evidence.

## Quickstart

```bash
pip install -e .          # add [llm] for real-model judge/agents

# run the static regression suite (offline, deterministic toy agent)
python -m agent_eval.cli run --cases examples/cases.yaml \
    --agent examples.toy_agent:agent --judge-backend stub --samples 2

# multi-turn simulation against scripted personas (incl. a prompt-leak attack)
python -m agent_eval.cli simulate --scenarios examples/scenarios.yaml \
    --agent examples.toy_chat_agent:agent --judge-backend stub --samples 2

# coding-agent cases (workspace-state assertions on a repo fixture)
python -m agent_eval.cli run --cases examples/cases_coding.yaml \
    --agent examples.coding_agent:agent

# the offline demo has 12 cases; production keeps the default --min-cases 50
python -m agent_eval.cli compare baseline candidate --min-cases 12
python -m agent_eval.cli diff baseline candidate       # trajectory diff

# production judge admission (defaults: 50+ labels, 10+ per human class)
python -m agent_eval.cli calibrate --golden golden_production.yaml \
    --judge-backend litellm --judge-model k3 --json-out calibration.json

# case supply: distill new cases from recorded traces (tool_called derived
# from the observed tool-call sequence)
python -m agent_eval.cli distill --out cases_distilled.yaml --limit 50
```

Plug in a real model (any LiteLLM-supported provider; Kimi K3 example):

```bash
export KIMI_API_KEY=sk-...
export KIMI_API_BASE=https://api.kimi.com/coding/v1
python -m agent_eval.cli run --cases examples/cases.yaml \
    --agent examples.kimi_agent:agent --judge-backend litellm --judge-model k3 \
    --samples 2 --concurrency 3

# real K3 coding agent: function-calling loop with sandboxed file/shell tools
python -m agent_eval.cli run --cases examples/cases_coding.yaml \
    --agent examples.kimi_coding_agent:agent
```

## Your agent's contract

Any callable works. Instrument it with OTel spans to unlock cost, tool and
trajectory assertions:

- `run`: `fn(str) -> str | {"output", "turns"}` — or `fn(task, env_dir)` for coding agents
- `simulate`: `fn(messages: list[dict]) -> str`
- spans: `tool.name` on tool spans, `gen_ai.usage.cost_usd` on LLM spans
  (`agent_eval.tracing.set_token_usage` does the math), `prompt.version` hash

## How the CI gate works

`.github/workflows/agent-eval.yml`: every PR runs pytest, then one evaluator
version runs the target worktree's agent as `baseline` and the PR agent as
`candidate`. Both share run ids in one SQLite DB. Before statistics, `compare`
checks manifests, case sets, dataset hashes, sample counts, judge/user config
and sample status. Production judge gates should add `--require-calibration`;
a missing or rejected latest calibration is invalid.

For delta `B-A` and `--min-delta` (default 0.05), a CI upper bound below -0.05
is an important regression (2); a lower bound at/above -0.05 is a pass (0);
a crossing interval is inconclusive (3). Fewer than `--min-cases` (default 50)
is also inconclusive. **Inconclusive is neither proof of regression nor evidence
of safety**: add independent cases and inspect changed cases/traces rather than
weakening thresholds just to make CI green.

`run` / `simulate` exit codes: 0 = all passed, 1 = assertion failure,
3 = execution error. `compare`: 0 = pass, 1 = invalid, 2 = regression,
3 = inconclusive. `calibrate`: 0 = admitted, 4 = rejected. JSON reports contain
the full decision inputs. The workflow uploads reports, diffs and eval.db for
14 days. Redact real traces before uploading: artifacts may contain sensitive
inputs and outputs. Legacy runs without a manifest remain readable but must be
rerun before strict `compare`.

## Layout

```
src/agent_eval/   tracing · storage · assertions · judge · retry · calibration
                  stats · simulation · diff · distill · manifest · compare
                  reporting · runner · cli
examples/         toy + real (Kimi K3) agents, cases, scenarios, golden set
tests/            289 offline pytest tests
```

## License

[MIT](LICENSE)
