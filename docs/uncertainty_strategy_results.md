# 不确定性标签策略结果

## 当前实验

设置：CheXpert-v1.0-small，frontal 图像，Cardiomegaly，患者级 internal train/val，official valid 固定测试，DenseNet-121，seed=42。

| 指标 | Clean-label | U-Ignore | U-MultiClass |
|---|---:|---:|---:|
| AUROC | 0.8544 | 0.8213 | 0.8430 |
| AUPRC | 0.7486 | 0.7299 | 0.7587 |
| 灵敏度 | 0.9394 | 0.3030 | 0.3030 |
| 特异度 | 0.5662 | 0.9853 | 0.9926 |
| F1 | 0.6631 | 0.4545 | 0.4598 |
| Brier | 0.1700 | 0.1634 | 0.1919 |

阈值分别由 internal validation 选择：Clean-label=0.295，U-Ignore=0.62，U-MultiClass=0.35。official valid 只用于最终评估。

## 解释

- U-MultiClass 的 official valid AUPRC 最高，AUROC 高于 U-Ignore，但略低于当前 clean-label 基线。
- U-Ignore 和 U-MultiClass 的 internal validation 阳性比例分别约为 0.125 和 0.132，而 official valid 阳性比例为 0.327；因此阈值迁移存在明显分布偏移风险。
- 当前 internal validation 二分类指标排除了 U-MultiClass 的 uncertain 类（类别 2），只在明确 negative/positive 样本上计算。该结果不代表 uncertain 类识别性能。

## 研究决策

正式三标签主线暂统一采用 U-MultiClass：每个标签独立输出 negative / uncertain / positive 三类，三个标签之间不互斥。该选择服务于后续不确定性路由研究，但仍需用 U-Ignore、U-Ones 或按标签最优策略作为消融/敏感性对照，不能仅凭 CheXpert 原论文宣称统一策略最优。

## 复现边界

当前结果是项目基线，不是 CheXpert 论文的严格复现。论文使用了不同的数据处理、模型选择/集成和评估协议。后续如需声称“复现”，必须单独建立论文协议实验，并明确报告差异。
