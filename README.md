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
any agent change → run eval suite (baseline vs candidate) → McNemar + bootstrap → merge or block
```

## Features

- **Observability first** — every run is traced with OpenTelemetry spans
  (LLM calls, tool calls, token cost, prompt version hash) and persisted to
  SQLite. Trajectory diffs show *where* behavior changed, not just *that* it did.
- **Two assertion layers** — cheap deterministic rules (output content, tool
  usage, turn/cost/latency budgets, workspace state) plus LLM-as-judge for
  semantic quality.
- **Judge calibration** — judges are evaluated too: score your judge against a
  human-labeled golden set (accuracy / Cohen's kappa / FPR / FNR) before you
  trust it as a gate.
- **Simulated users with hidden goals** — multi-turn rollouts against personas,
  including adversarial ones (e.g. a user who plays nice while fishing for your
  system prompt). Rule assertions backstop the LLM judge so a lenient simulated
  user can't let a leak slip through.
- **Coding agents** — cases can carry a repo-snapshot fixture; each sample runs
  in a fresh copy and assertions check the final workspace state
  (`command_exit`, diff size limits) — the SWE-bench FAIL_TO_PASS pattern.
- **Statistical gate** — pass^k sampling, McNemar's exact test and paired
  bootstrap CIs. The gate blocks *significant* regressions, not noise.
- **Parallel + rate-limit tolerant** — `--concurrency N` with exponential
  backoff on 429/5xx, so a rate-limited model plan slows the run instead of
  failing it.
- **GitHub Actions ready** — the included workflow runs the target branch as
  baseline and the PR as candidate, then blocks the merge on significant
  regression (exit code 2).

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

# the gate: statistically compare two runs
python -m agent_eval.cli compare baseline candidate    # exit 2 = significant regression
python -m agent_eval.cli diff baseline candidate       # trajectory diff
```

Plug in a real model (any LiteLLM-supported provider; Kimi K3 example):

```bash
export KIMI_API_KEY=sk-...
export KIMI_API_BASE=https://api.kimi.com/coding/v1
python -m agent_eval.cli run --cases examples/cases.yaml \
    --agent examples.kimi_agent:agent --judge-backend litellm --judge-model k3 \
    --samples 2 --concurrency 3
```

## Your agent's contract

Any callable works. Instrument it with OTel spans to unlock cost, tool and
trajectory assertions:

- `run`: `fn(str) -> str | {"output", "turns"}` — or `fn(task, env_dir)` for coding agents
- `simulate`: `fn(messages: list[dict]) -> str`
- spans: `tool.name` on tool spans, `gen_ai.usage.cost_usd` on LLM spans
  (`agent_eval.tracing.set_token_usage` does the math), `prompt.version` hash

## How the CI gate works

`.github/workflows/agent-eval.yml`: on every PR, a worktree of the target
branch runs as `baseline`, your PR runs as `candidate`, both share run ids in
one SQLite DB. `diff` always prints a trajectory report; `compare` exits 2 on
significant regression and blocks the merge. The examples run fully offline
(stub judge), so CI needs no secrets.

## Layout

```
src/agent_eval/   tracing · storage · assertions · judge · retry · calibration
                  stats · simulation · diff · runner · cli
examples/         toy + real (Kimi K3) agents, cases, scenarios, golden set
tests/            217 offline pytest tests
```

## License

[MIT](LICENSE)
