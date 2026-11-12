# 实验分析工具

在仓库根目录执行。原始本地 `analysis/` 继续忽略；这里公开经过整理的工具，临时探针和训练产物不纳入版本控制。

| 工具 | 输入与用途 | 依赖 |
| --- | --- | --- |
| `compare_results.py` | 默认读取发布的快照，输出成功率、Wilson 区间、精确 McNemar、配对 bootstrap | Python 标准库 |
| `export_results.py` | 从固定目录的八份本地评测报告和当前数据/配置/权重导出快照，校验计数并记录 SHA-256 | Python 标准库 |
| `behavior_diff.py` | 在相同场景上回放 SFT 与 GRPO，比较描述性过程指标；由本地分析脚本整理 | 项目训练依赖、两个 checkpoint 和场景文件 |

```bash
python tools/compare_results.py
python tools/compare_results.py --snapshot docs/results/evaluations.json --bootstrap 5000
python tools/export_results.py --output analysis/new_evaluations.json
python tools/behavior_diff.py --config configs/sft_3m.yaml --scenarios dataset/test.npz --sft outputs_sft_3m/train/pocketvla-3m --grpo outputs_grpo_3m/pocketvla-3m-grpo --output analysis/behavior_diff_new.json
```

快照比较按 episode ID 配对，拒绝不同场景集合和重复 ID。配对成立仍依赖历史运行使用相同场景的假设；导出时的文件哈希无法补回运行时证据。bootstrap 使用独立固定随机种子 0，各比较 5000 次，区间仅描述场景采样。

`behavior_diff.py` 中距离反转、远离片段、闭爪到张爪指令变化、抓取窗口重入等是可观察代理。搬运阶段也会计入，不能自动解释成恢复、回溯策略或因果机制。既有过程快照位于 `docs/results/behavior.json`，保留原始文件哈希，未伪补历史运行元数据。

检查统计实现：

```bash
python -m unittest discover -s tests -v
```
