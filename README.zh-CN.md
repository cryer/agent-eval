# agent-eval

**给 LLM agent 用的 CI**——每次改 prompt 都自动重放评测集，用统计检验判断
这次改动是真回归还是随机噪声。

[English](README.md) | 中文

---

## 为什么需要它

系统提示词改一行，agent 可能悄悄就坏了。单元测试帮不上忙：agent 的行为是
概率性的、多轮的、靠工具驱动的。`agent-eval` 把"这次 prompt 改动有没有让
效果变差"变成一道有统计依据的 CI 门禁，给出明确的拦/放答案。

```
prompt 改动 → 重放评测集（baseline vs candidate）→ McNemar + bootstrap → 合并或拦截
```

## 功能

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

# 门禁：统计对比两次运行
python -m agent_eval.cli compare baseline candidate    # 退出码 2 = 显著回归
python -m agent_eval.cli diff baseline candidate       # 轨迹 diff
```

接入真实模型（任何 LiteLLM 支持的 provider；Kimi K3 示例）：

```bash
export KIMI_API_KEY=sk-...
export KIMI_API_BASE=https://api.kimi.com/coding/v1
python -m agent_eval.cli run --cases examples/cases.yaml \
    --agent examples.kimi_agent:agent --judge-backend litellm --judge-model k3 \
    --samples 2 --concurrency 3
```

## 你的 agent 要满足的契约

任何可调用对象都可以接入。用 OTel 埋点可以解锁成本、工具调用和轨迹断言：

- `run`：`fn(str) -> str | {"output", "turns"}`——coding agent 用 `fn(task, env_dir)`
- `simulate`：`fn(messages: list[dict]) -> str`
- span 约定：工具 span 带 `tool.name`，LLM span 带 `gen_ai.usage.cost_usd`
  （`agent_eval.tracing.set_token_usage` 帮你算），prompt 哈希放 `prompt.version`

## CI 门禁怎么工作

`.github/workflows/agent-eval.yml`：每个 PR 触发后，用 git worktree 签出
目标分支跑 `baseline`，PR 代码跑 `candidate`，两边共用同一个 SQLite 库的
run id。`diff` 永远输出轨迹对比报告；`compare` 发现显著回归时以退出码 2
拦截合并。示例完全离线（stub 裁判），CI 不需要配任何密钥。

## 目录结构

```
src/agent_eval/   tracing · storage · assertions · judge · retry · calibration
                  stats · simulation · diff · runner · cli
examples/         toy + 真实（Kimi K3）agent、case、场景、金标集
tests/            217 个离线 pytest 测试
```

## 许可证

[MIT](LICENSE)
