# agent-eval

**给 LLM agent 用的 CI**——改 prompt、换模型、动工具、重构循环：任何改动
都自动重放评测集，用统计检验判断是真回归还是随机噪声。

[English](README.md) | 中文

---

## 为什么需要它

agent 会悄悄变坏——而且不只是改 prompt 的时候。换个模型版本、改一下工具
schema、重构 agent 循环代码，同样可能让行为漂移。单元测试帮不上忙：agent
的行为是概率性的、多轮的、靠工具驱动的。`agent-eval` 把"这次改动有没有让
效果变差"变成一道有统计依据的 CI 门禁——不管改动来自哪一层，都给出明确的
拦/放答案。

```
任何 agent 改动 → 重放 baseline/candidate → 校准裁判 + 聚类 bootstrap 门禁 → 合并或拦截
```

## 功能

- **观测优先**：每次运行用 OpenTelemetry 埋点（LLM 调用、工具调用、token
  成本、prompt 版本哈希），持久化到 SQLite。轨迹 diff 告诉你行为在*哪一步*
  变了，而不只是*变了*。
- **两层断言**：便宜的确定性规则断言（输出内容、工具调用、轮数/成本/延迟
  预算、工作区状态）+ LLM-as-judge 判语义质量。
- **裁判校准准入**：用人工金标集计算 accuracy / Cohen's kappa / FPR / FNR，
  默认要求 50 条、每类至少 10 条、kappa ≥ 0.6、FPR ≤ 0.05；不合格退出码 4。
  `compare --require-calibration` 强制检查相同 judge 配置的最新合格记录。
- **带隐藏目标的模拟用户**：多轮对话仿真，含对抗人格（比如假装配合、
  实际在套系统提示词的用户）。规则断言兜底，模拟用户放水也漏不掉。
- **coding agent 支持**：case 可以挂仓库快照 fixture，每个采样在独立副本里
  跑，断言检查工作区终态（`command_exit`、改动文件数上限）——SWE-bench 的
  FAIL_TO_PASS 模式。
- **统计门禁**：多次采样按 case 聚类 bootstrap，不把 k 次采样当独立 case。
  以最小重要回归阈值和置信区间判定放行、回归或证据不足；单采样 McNemar
  只作辅助诊断，不再决定门禁。
- **可信执行清单**：记录 Git、数据集、模型和 judge 配置；agent/judge 异常与
  断言失败分开。`compare` 先拒绝不完整、采样数或数据集不一致的运行，避免假绿。
- **CI 报告**：`run` / `simulate` 支持 JSON 与 JUnit XML，`compare` / `calibrate`
  支持 JSON；workflow 上传报告、轨迹 diff 和 SQLite artifact。
- **并发 + 限流容错**：`--concurrency N` 并行执行，429/5xx 自动指数退避
  重试——套餐限并发时表现为变慢，而不是评测失败。
- **GitHub Actions 就绪**：内置 workflow 用目标分支跑 baseline、PR 跑
  candidate，回归或证据不足都会拦截合并。

## 快速开始

```bash
pip install -e .          # 真实模型裁判/agent 再装 pip install -e .[llm]

# 跑静态回归集（离线、确定性的 toy agent）
python -m agent_eval.cli run --cases examples/cases.yaml \
    --agent examples.toy_agent:agent --judge-backend stub --samples 2

# 多轮仿真（脚本化 persona，含提示词泄露攻击场景）
python -m agent_eval.cli simulate --scenarios examples/scenarios.yaml \
    --agent examples.toy_chat_agent:agent --judge-backend stub --samples 2

# coding agent 用例（对仓库 fixture 断言工作区终态）
python -m agent_eval.cli run --cases examples/cases_coding.yaml \
    --agent examples.coding_agent:agent

# 离线示例只有 12 个 case，显式放宽样本数；生产保持默认 --min-cases 50
python -m agent_eval.cli compare baseline candidate --min-cases 12
python -m agent_eval.cli diff baseline candidate       # 轨迹 diff

# 生产裁判准入：真实金标集默认至少 50 条、每类至少 10 条
python -m agent_eval.cli calibrate --golden golden_production.yaml \
    --judge-backend litellm --judge-model k3 --json-out calibration.json

# case 供给：从历史 trace 蒸馏新 case（tool 调用序列派生 tool_called 断言）
python -m agent_eval.cli distill --out cases_distilled.yaml --limit 50
```

接入真实模型（任何 LiteLLM 支持的 provider；Kimi K3 示例）：

```bash
export KIMI_API_KEY=sk-...
export KIMI_API_BASE=https://api.kimi.com/coding/v1
python -m agent_eval.cli run --cases examples/cases.yaml \
    --agent examples.kimi_agent:agent --judge-backend litellm --judge-model k3 \
    --samples 2 --concurrency 3

# 真实 K3 coding agent：function-calling 循环 + 沙箱化文件/shell 工具
python -m agent_eval.cli run --cases examples/cases_coding.yaml \
    --agent examples.kimi_coding_agent:agent
```

## 你的 agent 要满足的契约

任何可调用对象都可以接入。用 OTel 埋点可以解锁成本、工具调用和轨迹断言：

- `run`：`fn(str) -> str | {"output", "turns"}`——coding agent 用 `fn(task, env_dir)`
- `simulate`：`fn(messages: list[dict]) -> str`
- span 约定：工具 span 带 `tool.name`，LLM span 带 `gen_ai.usage.cost_usd`
  （`agent_eval.tracing.set_token_usage` 帮你算），prompt 哈希放 `prompt.version`

## CI 门禁怎么工作

`.github/workflows/agent-eval.yml`：每个 PR 先跑 pytest，再用同一版评测器分别
执行目标分支 worktree 中的 `baseline` agent 和 PR 中的 `candidate` agent。
两边共用 SQLite run id。`compare` 在统计前检查执行清单、case 集合、数据集
哈希、采样数、judge/模拟用户配置和样本状态。真实 judge 管线还应加
`--require-calibration`；缺失或被拒绝的最新校准会返回 1。

统计判定看平均通过率差 `B-A` 及其置信区间，`--min-delta` 默认 0.05：
区间上界 `< -0.05` = 重要回归（2）；区间下界 `>= -0.05` = 没有重要回归（0）；
跨过该边界 = 证据不足（3）。case 少于 `--min-cases`（默认 50）也为证据不足。
**证据不足不等于发现回归，也不等于安全放行**：应优先补独立 case，再看变差的
case 和轨迹；不要只为变绿而调大阈值或降低样本要求。

`run` / `simulate` 退出码：0 = 全通过，1 = 有断言失败，3 = 有执行异常。
`compare`：0 = 放行，1 = 无效，2 = 重要回归，3 = 证据不足；`calibrate`：
0 = 准入，4 = 拒绝。JSON 报告包含完整判定依据；workflow 上传报告、diff 和
eval.db（保留 14 天）。真实流量的 artifact 可能含敏感输入输出，上传前应先脱敏。
旧版本产生、没有执行清单的 run 仍可查看，但需重跑后才能参与严格 `compare`。

## 目录结构

```
src/agent_eval/   tracing · storage · assertions · judge · retry · calibration
                  stats · simulation · diff · distill · manifest · compare
                  reporting · runner · cli
examples/         toy + 真实（Kimi K3）agent、case、场景、金标集
tests/            289 个离线 pytest 测试
```

## 许可证

[MIT](LICENSE)
