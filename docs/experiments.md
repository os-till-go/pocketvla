# 实验记录：抓取失败诊断与 GRPO 奖励设计

维护者：os-till-go。基于上游 `penpenzh/pocketvla` 的本地复现与扩展；上游基线提交为 `5bdf7911499b4bbf8a0575595568756b45cb72f2`。本报告于 2026-11-12 整理已有文件，未重新训练或重新执行模型评测。

## 研究问题与贡献边界

3.02M VLA 在二维语言引导抓取任务中，失败主要发生在哪个阶段？对“空手时在目标球附近闭爪”增加中间奖励，能否改善成功率？

上游实现模型、仿真器、专家、数据生成、SFT、GRPO 主体与 OOD 场景。本 fork 已有代码提交 `28d15e1` 增加奖励，`063acfb` 增加实验配置；本次公开诊断工具、已有结果及其统计分析，不将上游工作计为个人原创。

## 观察、假设与实现

SFT 的 500 个 ID 场景成功 395 个，失败分类为：从未抓对目标球 87 个、抓错球 1 个、抓对后未放入 17 个。该分类来自保存的评测报告，不能单凭分类区分定位、控制或夹爪时序的因果作用。

已有 [行为快照](results/behavior.json) 比较 SFT 与 GRPO20：SFT 失败的 105 个场景中救回 29 个，同时丢失 19 个原本成功的场景。这说明结果发生变化，尚不能说明模型学会回溯或恢复。

`tools/behavior_diff.py` 记录距离方向反转（5 mm 去抖）、累计远离超过 3 cm 的片段、闭爪到张爪的指令变化、进入抓取窗口次数及最近距离时刻。完整回合包含搬运阶段，球被抓起后会随末端移动，因此这些指标只是描述性代理，不应直接解释为有意重试。过程快照没有历史权重和场景哈希，不能与归档评测视为完全取证一致。

据此提出的探索假设是：给抓取附近的闭爪事件额外信号可能改善抓取。当前实现的事件条件为：

```text
本回合尚未触发 && 空手 && 动作前到目标球距离 <= grasp_thresh && grip > 0
→ 回合奖励加 inwin_bonus（最多一次）
```

它是最终回合标量中的中间事件奖励，没有逐步返回训练奖励。原奖励本身已包含成功、速度、抓对球、最终球筐距离和抓错球项，不能描述为“只有最终成功才有奖励”。

**待核查的实现限制**：环境先更新关节，再以 `< grasp_thresh` 判断最近球能否吸附；奖励用动作前位置和 `<=` 判断目标球。因此额外奖励并不等于实际成功抓取事件，也没有要求夹爪必须从张开切换为闭合。报告保留该语义，不把现有结果当成修正后实现的结果。

## 实验设置

| 项目 | 设置 / 证据 |
| --- | --- |
| 模型 | 上游 3m 配置，开启辅助头 |
| 种子 | 配置 `seed: 42`；未做多训练种子 |
| SFT | 配置为 15 epochs、batch 256；使用保存的本地 SFT checkpoint |
| GRPO | group size 8，每轮 20 场景/更新，学习率 2.5e-5，KL 2.0，辅助权重 0.5 |
| G1 与 G3 | 100 轮，无 early stopping；奖励系数 0 vs 0.3，输出目录各自独立 |
| GRPO20 | 日志只有 20 轮；当前 `grpo_3m.yaml` 是最多 100 轮、early_stop 20，不能当成精确的历史 20 轮运行命令 |
| checkpoint 选择 | GRPO 代码保存训练采样平均回报最高的 checkpoint，并非最后一轮或独立验证集最优模型 |
| 评测 | ID 500 固定场景；OOD 300 固定场景，额外干扰物 |
| 历史硬件记录 | RTX 3060 Laptop 6GB，CUDA 11.8；来自本地复现笔记，原始报告未嵌入硬件信息 |
| 历史软件记录 | Python 3.11.7、torch 2.4.1+cu118、transformers 4.57.6；来自复现笔记，未有完整原始环境锁定文件 |

G1/G3 的可执行配置除 `inwin_bonus` 和输出路径外相同。改变奖励系数也改变 checkpoint 选择标准的数值含义，两个实验的训练回报绝对值不能直接比较。

## 结果与配对检验

逐场景原始结局见 [evaluations.json](results/evaluations.json)，程序生成的成功率、Wilson 区间和完整配对表见 [comparison.md](results/comparison.md)。

| 方法 | ID 成功 / 抓取 | OOD 成功 / 抓取 |
| --- | --- | --- |
| SFT | 79.0% / 82.4% | 31.0% / 58.0% |
| GRPO20 | 81.0% / 82.6% | 30.7% / 58.7% |
| G1-100 | 78.6% / 81.2% | 32.3% / 60.7% |
| G3-100 | 78.0% / 81.6% | 34.3% / 60.7% |

| G3 − G1 | 成功率差 | 救回 / 丢失 | 精确 McNemar p | 配对 bootstrap 95% 区间 |
| --- | --- | --- | --- | --- |
| ID | −0.6 个百分点 | 37 / 40 | 0.8199 | [−4.2, +2.8] 个百分点 |
| OOD | +2.0 个百分点 | 22 / 16 | 0.4177 | [−2.0, +6.0] 个百分点 |

配对 bootstrap 重采样完整场景对 5000 次，统计随机种子 0。McNemar 对成功/失败不一致的场景做双侧精确二项检验。它依赖历史报告确实用了相同场景这一假设；程序拒绝 episode ID 不一致、重复 ID、不同 split 或导出时数据集哈希不一致的配对。

**结论边界**：这批结果没有提供新增奖励改善成功率的统计证据。也没有证据据此宣称“RL 无效”“容量瓶颈”“奖励被刷”或“模型学会恢复”。区间覆盖场景采样不确定性，不包含训练种子方差；多组比较属于探索分析，p 值未做多重比较校正。

## 证据与可复现性

- 归档包含八份报告的逐场景结局、总体抓取率、失败分类、训练日志轮数及最佳回报轮数。
- 每项记录报告 SHA-256，以及**导出时**数据集、配置和本地权重的 SHA-256。它们可追踪当前文件，无法证明历史训练运行用了完全相同的文件。
- 原报告未绑定训练代码提交、运行环境、数据集哈希和权重哈希；当前配置也可能与历史运行不同。配对表是已有报告的重新计算，不是独立重现实验。
- 不上传本地训练权重、全部数据、运行日志或临时探针。上游下载权重有明确来源，但不能替代本地 checkpoint 来验证本报告的精确分数。
- GIF 来自已有 G1/G3 评测产物，#45 和 #249 分别展示改善与退步；完整结果统计覆盖全部场景。

## 复现步骤

### 只复核发布的统计

Python 3.10+，标准库即可，无需 GPU：

```bash
python tools/compare_results.py
python -m unittest discover -s tests -v
```

### 新建训练与评测运行

先创建 Python 环境，安装适合显卡的 CUDA 版 PyTorch，再安装 [requirements.txt](../requirements.txt)。该文件是上游依赖范围，不是历史运行锁定文件；历史使用的 torch 版本低于其中的版本要求。不要将这两种环境描述为同一套已验证环境。

以下命令在仓库根目录运行。已有输出时，先复制 YAML 并修改 `train.output_dir`、`grpo.output_dir`、`grpo.ckpt` 及评测的 `eval.ckpt/output_dir`，保存独立运行；数据集也应冻结并记录哈希。

```bash
python -m pvla collect
python -m pvla train --config configs/sft_3m.yaml
python -m pvla evaluate --config configs/sft_3m.yaml
python -m pvla grpo --config configs/grpo_3m_g1.yaml
python -m pvla grpo --config configs/grpo_3m_g3.yaml
python -m pvla evaluate --config configs/eval_g1_3m.yaml
python -m pvla evaluate --config configs/eval_g3_3m.yaml
python -m pvla.eval_ood --config configs/sft_3m.yaml
python -m pvla.eval_ood --config configs/eval_g1_3m.yaml
python -m pvla.eval_ood --config configs/eval_g3_3m.yaml
```

完整八报告导出还需要 GRPO20 的 ID/OOD 报告。上面的命令用于新实验，不保证复现历史 GRPO20 checkpoint。保留旧归档，用 `python tools/export_results.py --output analysis/new_evaluations.json` 导出另一份快照，再用 `python tools/compare_results.py --snapshot analysis/new_evaluations.json` 比较。

过程诊断另见 [工具说明](../tools/README.md)。建议后续先核对奖励时间语义，再补齐多种子、独立验证集选模，以及运行时配置/代码/权重/场景哈希记录。
