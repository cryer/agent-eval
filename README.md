# agent-eval

**CI for LLM agents** — replay your eval suite on every prompt change, and let
statistics decide whether the change is a regression or just noise.

English | [中文](#中文版本)

---

## Why

Changing one line of a system prompt can silently break your agent. Unit tests
don't help: agent behavior is probabilistic, multi-turn, and tool-driven.
`agent-eval` turns "does this prompt change make things worse?" into a
CI gate with a yes/no answer backed by statistical tests.

```
prompt diff → run eval suite (baseline vs candidate) → McNemar + bootstrap → merge or block
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

---

## 中文版本

**给 LLM agent 用的 CI**——每次改 prompt 都自动重放评测集，用统计检验判断
这次改动是真回归还是随机噪声。

### 为什么需要它

系统提示词改一行，agent 可能悄悄就坏了。单元测试帮不上忙：agent 的行为是
概率性的、多轮的、靠工具驱动的。`agent-eval` 把"这次 prompt 改动有没有让
效果变差"变成一道有统计依据的 CI 门禁，给出明确的拦/放答案。

### 功能

- **观测优先**：每次运行用 OpenTelemetry 埋点（LLM 调用、工具调用、token
  成本、prompt 版本哈希），持久化到 SQLite。轨迹 diff 告诉你行为在*哪一步*
  变了，而不只是*变了*。
- **两层断言**：便宜的确定性规则断言（输出内容、工具调用、轮数/成本/延迟
  预算、工作区状态）+ LLM-as-judge 判语义质量。
- **裁判校准**：裁判本身也要被评测——用人工标注的金标集给 judge 算
  accuracy / Cohen's kappa / FPR / FNR，校准合格才配当门禁。
- **带隐藏目标的模拟用户**：多轮对话仿真，含对抗人格（比如假装配合、
  实际在套系统提示词的用户）。规则断言兜底，模拟用户放水也漏不掉。
- **coding agent 支持**：case 可以挂仓库快照 fixture，每个采样在独立副本里
  跑，断言检查工作区终态（`command_exit`、改动文件数上限）——SWE-bench 的
  FAIL_TO_PASS 模式。
- **统计门禁**：pass^k 采样、McNemar 精确检验、paired bootstrap 置信区间。
  拦的是*显著*回归，不是噪声。
- **并发 + 限流容错**：`--concurrency N` 并行执行，429/5xx 自动指数退避
  重试——套餐限并发时表现为变慢，而不是评测失败。
- **GitHub Actions 就绪**：内置 workflow 用目标分支跑 baseline、PR 跑
  candidate，显著回归直接拦截合并（退出码 2）。

### 快速开始

见上方英文 Quickstart，命令完全一致。核心四条：

```bash
pip install -e .
python -m agent_eval.cli run --cases examples/cases.yaml \
    --agent examples.toy_agent:agent --judge-backend stub --samples 2
python -m agent_eval.cli simulate --scenarios examples/scenarios.yaml \
    --agent examples.toy_chat_agent:agent --judge-backend stub --samples 2
python -m agent_eval.cli compare baseline candidate   # 退出码 2 = 显著回归，拦
```

### 许可证

[MIT](LICENSE)
