# AgenTracer 复现证据（公开数据上的配对评估）

本文档记录本仓库 AgenTracer 适配层之外的**复现实验**：在公开 TracerTraj 验证集上，
对"多粒度 RL 是否提升失败归因"这一主张做的可复核评估。

> 结论先行：**在公开数据与单卡条件下，没有观察到稳定提升**。12 步 GRPO 的单个 checkpoint
> 相对裸模型 +3.04pp（95% CI [0.63, 5.44]），但三个 checkpoint 的剂量-反应不单调，
> 平均效应 +0.67pp（CI 含 0）。因此本仓库只声称"适配层可用"，不声称"复现了论文指标"。

## 1. 复现边界（为什么不能对标论文）

| 项 | 论文 | 本仓库 |
|---|---|---|
| 基座权重 | AgenTracer-8B | **未公开**，改用 Qwen3-8B + LoRA(r16, α32) |
| 训练数据 | TracerTraj（论文版） | 公开 `data-v1.0.0`（作者自述与论文版不一致） |
| 推理协议 | `<think>…</think><answer>agent \| step</answer>` | 同左；与本仓库适配层使用的 **JSON 协议不同** |
| 方法版本 | v2（NeurIPS 2026）为工具增强多跳 agent | 本仓库适配 v1 的单次直出协议 |

因此本实验只能做"同数据自比"，不能用来验证论文的 Who&When 数字。

## 2. 评估设置

- 数据：`validation_canonical.jsonl`，790 条（agentic 600 / coding 127 / math 63）
- 解码：greedy（temperature 0），prompt ≤24576，生成 ≤16384
- 打分：论文口径的四项指标（parse / agent_exact / step_exact / joint_exact），配对按 `(task_id, source_benchmark)`
- 噪声底：同一策略在同一批 200 条上重跑一次（A/A），用于给出可比噪声

## 3. 主结果（790 条，配对）

| 指标 | 裸 Qwen3-8B | GRPO 12 步 | 差值 | 95% CI（配对 bootstrap） |
|---|---:|---:|---:|---|
| parse | 89.9% | 87.5% | −2.4pp | — |
| agent_exact | 77.8% | 76.8% | −1.0pp | — |
| **step_exact** | **37.3%** | **40.4%** | **+3.04pp** | **[+0.63, +5.44]** |
| **joint_exact** | **34.9%** | **37.7%** | **+2.78pp** | **[+0.38, +5.32]** |
| 截断率 | 4.6% | 6.6% | +2.0pp | — |

分 subset（step_exact）：agentic 42.5% → 46.5%（+4.0pp, p=0.011）；coding 10.2% → 9.4%（−0.8pp, p=1.0）；
math 42.9% → 44.4%（+1.6pp, p=1.0）。

## 4. 剂量-反应（同一批 200 条，逐条配对）

| policy | step_exact | joint_exact | 相对裸模型 | p |
|---|---:|---:|---:|---:|
| 裸模型 | 44.5% | 42.0% | — | — |
| step-5 | 45.5% | 43.0% | +1.0pp | 0.864 |
| step-10 | 43.0% | 40.0% | −1.5pp | 0.728 |
| step-12 | 47.0% | 44.0% | +2.5pp | 0.473 |

三个 checkpoint 的平均效应：step **+0.67pp**（95% CI [−3.67, +5.33]），joint +0.33pp（CI [−4.00, +4.67]）。
checkpoint 间跨度 4.0pp，**不单调**。

## 5. 噪声底（A/A）

同一策略（裸模型）、同一批 200 条、同一 8 分片并行重跑：step 44.5% → 44.5%（+0.0pp），
逐条翻转 13 对 13 完全对称。说明批处理带来的是无偏抖动，测量噪声约 0.5pp，
**不足以解释 checkpoint 间 4pp 的差异**——策略确实在变，但没有朝一个方向走。

## 6. 指标本身的缺陷（重要）

| 现象 | 实测 |
|---|---|
| agentic 子集只有一个 agent 名（`assistant`） | 该子集 600/790，`agent_exact` 恒为 100% |
| 金标 step 高度集中 | 众数占比：agentic 22.5%、coding 28.7%、math 43.0% |
| 常数预测器（永远答众数 step） | **joint 23.8%**；分 subset 为 23.3% / 28.3% / 19.0% |
| 模型 vs 常数预测器 | agentic +19.2pp；**coding −22.9pp；math −6.3pp** |

即：在当前指标下，coding 与 math 两个子集的模型表现**低于"永远答众数 step"**。
任何在该数据上报告的提升，都必须同时给出相对常数基线的增量，否则会被基率吞掉。

## 7. 与适配层相关的三个注意事项

1. **prompt 中含 `Ground Truth` 字段**（`agentracer.build_prompt`）。评测时必须置空，
   否则等于把答案喂给模型，指标失效。
2. **步号基准**：公开 TracerTraj 中 agentic 与 math 的 `mistake_step` 从 1 开始，
   而 `format_trajectory` 按零基枚举，存在整体偏一位的风险。
3. **协议不同**：适配层解析 JSON（`mistake_step/mistake_agent/reason/suggested_fix`），
   本实验的模型输出 `<answer>agent | step</answer>`，两者不能混报。

## 8. 复现方式

```bash
# 1) 用训练侧脚本产出预测（略），得到 predictions/*.jsonl
# 2) 重算配对表
python scripts/reproduction/agentracer_paired_eval.py \
  --records  /path/validation_canonical.jsonl \
  --baseline predictions/baseline.predictions.jsonl \
  --adapter  predictions/trained.predictions.jsonl \
  --extra    s5=predictions/s5.predictions.jsonl \
             s10=predictions/s10.predictions.jsonl \
             s12=predictions/trained.predictions.jsonl

# 3) 重算金标分布与常数基线
python scripts/reproduction/agentracer_base_rates.py --records /path/validation_canonical.jsonl
```

完整产物（逐条预测、重算表格、sha256 清单）保存在实验机
`/data3/xhy/codes/work/AgenTracer/runs/reproduction_bundle/`，
因 `*.jsonl` 被 `.gitignore` 排除，未纳入仓库。

## 9. 结论

- 适配层可用：6 个契约测试通过，归因链路可在框架内正常运行。
- 方法主张未获支持：训练带来的提升无法与 checkpoint 间波动区分。
- 指标需要修正后才能用于任何"复现成功"的声明（见第 6 节）。
