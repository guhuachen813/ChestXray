# CheXpert 不确定性路由 Agent v1 实验报告

## 1. 实验目的

本阶段验证：在患者级划分、统一温度校准和明确人工复核预算下，Hard QC、Soft QC、U-MultiClass 不确定性以及异构模型级联，能否降低胸片选择性分类风险。

目标疾病为 Cardiomegaly，数据集为 CheXpert-v1.0-small。重点是建立可审计、可复现的初版 Agent，而不是引入复杂的多 Agent 拓扑。

## 2. 数据与标签

原始数据位于 AutoDL：

```text
/root/autodl-tmp/CheXpert-v1.0-small
```

其中包含 `train.csv`、`valid.csv` 及对应图像目录。训练和评估时动态读取图像并缩放为 224，不预先生成新的图像数据集。

主实验采用 U-MultiClass 标签：

```text
原始标签 0    -> negative (0)
原始标签 1    -> positive (1)
原始标签 -1   -> uncertain (2)
原始标签 NaN  -> negative (0)
```

三分类输出为 `p_negative, p_positive, p_uncertain`。疾病二分类指标统一使用 `p_positive`，而不是 argmax 标签。internal/route-validation 的二分类指标排除了 uncertain 标签；official valid 中 Cardiomegaly 没有 uncertain 标签。

## 3. 患者级划分

Agent 实验重新建立患者级划分，避免训练、校准、路由调参和最终测试互相污染。

| 子集 | 用途 | 患者数 | 图像数 |
|---|---|---:|---:|
| model_train | 训练两个模型 | 45,175 | 132,419 |
| model_selection | 选择最佳 checkpoint | 6,453 | 19,904 |
| calibration | 拟合温度参数 | 6,453 | 19,390 |
| route_validation | 诊断和选择路由规则 | 6,453 | 19,314 |
| official_valid | 最终一次性确认 | 200 | 202 |

划分报告为：

```text
patient_overlap = 0
lateral_rows_excluded_from_agent = 32,419
```

Agent 主分析只保留 Frontal 图像；Lateral 图像不进入主分类流程，数量单独报告。

## 4. 模型训练

Model 1：

```text
DenseNet-121 + U-MultiClass
```

Model 2：

```text
ResNet-50 + U-MultiClass
```

两个模型使用相同患者划分、标签定义、图像尺寸和归一化，以不同架构提供结构异质性。

训练配置：

```text
image_size = 224
batch_size = 16
epochs = 10
seed = 42
train_split = model_train
val_split = model_selection
```

训练结果：

- DenseNet-121 最佳验证损失约为 0.2651，出现在第 7 轮；
- ResNet-50 最佳验证损失约为 0.2394，出现在第 7 轮。

## 5. 温度校准

两个模型分别只使用 calibration 集拟合温度：

```text
calibrated_logits = logits / T
probabilities = softmax(calibrated_logits)
```

初始脚本直接优化 T 并使用 clamp，导致 ResNet-50 温度错误地落在下限 0.05。检查 calibration logits 后发现：

```text
T=0.50 的 NLL = 0.505856
T=0.05 的 NLL = 2.769675
```

随后将温度参数化为 log(T) 进行优化，避免 clamp 后梯度失效。

最终温度：

| 模型 | 温度 |
|---|---:|
| DenseNet-121 | 0.63821447 |
| ResNet-50 | 0.54853988 |

校准时使用与训练一致的 image_size=224。

## 6. Hard QC 与 Soft QC

### 6.1 Hard QC

Hard QC 检查输入是否可用或违反输入协议：

- 文件能否读取；
- 图像尺寸是否过小；
- 宽高比是否异常；
- 动态范围是否过窄；
- 近似全黑/全白；
- 前景比例是否极端；
- 是否为 Lateral；
- 边界是否疑似严重截断。

最初前景比例上限为 0.45，导致全部正常 CheXpert 图像失败。实际前景比例约为 0.947，修正阈值后 route-validation 结果为：

```text
rows = 19,314
hard_fail = 5
hard_fail_rate = 0.026%
```

这 5 张均由 aspect_ratio 触发，比例很低，保留该规则。

### 6.2 Soft QC

v1 使用纯图像统计和元数据，不依赖未公开源码的 DeepClean、Hu-style CXR 或 CheXmask 临床质量标签。特征包括：

```text
projection_ap, projection_unknown, foreground_ratio
dynamic_range, mean_intensity, intensity_std, contrast
blur_score, noise_score, black_ratio, white_ratio
border_crop_score, left_right_symmetry, center_offset
quality_risk
```

Soft QC 仅作为风险信号，不称为医生级质量判断。route-validation 平均 quality_risk 为 0.2454。

## 7. 初始路由

初始规则：

```text
Hard QC 失败       -> DEFER_TO_HUMAN
质量风险过高       -> DEFER_TO_HUMAN
p_uncertain 过高   -> CALL_MODEL_2
Model 1 置信度不足 -> CALL_MODEL_2
否则               -> ACCEPT
```

Model 2 调用后最初使用简单融合：

```text
p_fused = (p1_positive + p2_positive) / 2
d = abs(p1_positive - p2_positive)
```

只有融合概率明确且分歧较小时自动接受。

## 8. p_uncertain 诊断

总体分箱中，p_uncertain 从低值到约 0.179 时错误率上升，但最高区间错误率反而下降。进一步控制 Model 1 confidence 后，没有稳定的单调增量关系。

因此不能把 p_uncertain 解释为独立错误概率。本阶段将：

```text
p_uncertain -> CALL_MODEL_2
```

从 v1 主路由中移除，但保留其诊断结果。

## 9. Model 2 互补性与分歧

route-validation 的 18,905 个已知标签样本结果：

| 指标 | 结果 |
|---|---:|
| Model 1 错误率 | 10.70% |
| Model 2 错误率 | 8.76% |
| 简单融合错误率 | 9.11% |
| Model 1 错、Model 2 对 | 764 |
| Model 1 对、Model 2 错 | 398 |
| 两者都错 | 1,258 |

分歧分箱：

| abs(p1-p2) | Model 1 错误率 | 融合错误率 |
|---|---:|---:|
| <0.05 | 3.64% | 3.65% |
| 0.05-0.10 | 12.71% | 12.37% |
| 0.10-0.20 | 19.07% | 17.87% |
| 0.20-0.30 | 30.63% | 25.52% |
| >=0.30 | 49.66% | 29.37% |

分歧是有价值的离线风险信号，但计算分歧需要先运行 Model 2，因此不能直接作为节省 Model 2 调用的门控条件。

在真实调用子集上：

```text
Model 1 错误率：34.49%
Model 2 错误率：25.03%
Fusion 错误率：26.95%
```

调用后直接使用 Model 2 优于简单平均。

## 10. 置信度门控

只使用 Model 1 confidence 决定是否调用 Model 2 的分析：

| Confidence gate | Model 2 调用率 | 总体错误率 |
|---:|---:|---:|
| 0.60 | 6.08% | 9.65% |
| 0.70 | 12.75% | 8.95% |
| 0.75 | 16.34% | 8.82% |
| 0.80 | 20.85% | 8.72% |
| 0.85 | 26.13% | 8.64% |
| 0.90 | 33.54% | 8.64% |
| 0.95 | 46.51% | 8.73% |

gate=0.85 相对全量 Model 2 的风险差为 -0.12 个百分点，bootstrap 95% CI 为 [-0.20, -0.04] 个百分点。但该阈值是在 route-validation 上选择的，只属于探索性结果。

## 11. Agent 结果

含 p_uncertain 的初始版本：

```text
accept_rate = 78.59%
model2_call_rate = 22.42%
defer_rate = 21.41%
coverage = 79.11%
selective risk = 4.53%
```

移除 p_uncertain 后：

```text
accept_rate = 79.17%
model2_call_rate = 21.37%
defer_rate = 20.83%
coverage = 79.69%
selective risk = 4.53%
```

移除 p_uncertain 后接受错误率几乎不变，同时 Model 2 调用率下降约 1.05 个百分点。

## 12. Model 2 泛化检查

official valid 上直接分类结果：

```text
DenseNet-121 error_rate = 24.26%
ResNet-50 error_rate    = 28.22%
Fusion error_rate       = 26.73%
```

被门控调用的 46 张样本上：

```text
Model 1 error_rate = 45.65%
Model 2 error_rate = 63.04%
```

这与 route-validation 上 Model 2 优于 Model 1 的结果相反，说明级联策略对数据划分和样本分布敏感，不能把开发集优势解释为稳定泛化能力。

## 13. 最终 v1 策略

最终采用 DenseNet-121 的校准置信度选择性分类：

```text
Hard QC 失败                  -> DEFER_TO_HUMAN
Frontal 且 confidence >= 0.85 -> ACCEPT
Frontal 且 confidence <  0.85 -> DEFER_TO_HUMAN
不调用 Model 2
不使用 p_uncertain
```

其中：

```python
confidence = max(p1_positive, 1 - p1_positive)
```

固定参数：

```text
model = DenseNet-121 U-MultiClass
temperature = 0.63821447
confidence_threshold = 0.85
image_size = 224
```

最终结果：

| 数据集 | 已知标签数 | 接受数 | Coverage | Selective risk |
|---|---:|---:|---:|---:|
| route-validation | 18,905 | 13,963 | 73.86% | 3.58% |
| official valid | 202 | 156 | 77.23% | 17.95% |

official valid 只有 202 张图像，风险估计不稳定，不能据此宣称统计显著的泛化提升。

## 14. 校准分类指标

### DenseNet-121

| 数据集 | AUROC | AUPRC | Brier | ECE | 多类 Brier | 多类 ECE |
|---|---:|---:|---:|---:|---:|---:|
| route-validation | 0.8091 | 0.2369 | 0.0777 | 0.0041 | 0.2574 | 0.0410 |
| official valid | 0.8499 | 0.7652 | 0.1921 | 0.1639 | 0.3801 | 0.1163 |

### ResNet-50

| 数据集 | AUROC | AUPRC | Brier | ECE | 多类 Brier | 多类 ECE |
|---|---:|---:|---:|---:|---:|---:|
| route-validation | 0.7981 | 0.2209 | 0.0658 | 0.0094 | 0.2738 | 0.0323 |
| official valid | 0.8203 | 0.7085 | 0.2351 | 0.2195 | 0.4866 | 0.1305 |

最终校准评估中，DenseNet-121 在 AUROC、AUPRC、Brier 和 ECE 上均优于 ResNet-50，故作为主分类模型。二分类 Brier 与多类 Brier 定义不同，不能混用。

## 15. 实验结论

1. U-MultiClass 保留 uncertain 类并提供可诊断输出，可作为 Agent 主模型标签策略。
2. p_uncertain 与错误风险没有稳定的独立增量关系，因此不纳入 v1 路由。
3. Hard QC 可以稳定发现少量明显异常输入，修正阈值后失败率约为 0.026%。
4. ResNet-50 在 route-validation 上具有互补性，但在 official valid 上没有泛化，不能作为自动纠错主模型。
5. 模型分歧与错误率明显相关，但更适合做离线风险诊断，不适合在不预运行 Model 2 的情况下节省其调用。
6. DenseNet-121 calibrated confidence 的选择性分类在两个集合上都降低了接受样本风险，但 official valid 样本量很小。
7. 当前最稳妥的主结论是：DenseNet-121 是更强、更稳定的分类基线；规则式异构级联尚未证明能够稳定降低官方集选择性风险。

## 16. 限制与后续

主要限制：

- official valid 只有 202 张图像，置信区间较宽；
- Soft QC 是启发式图像统计，不是临床质量真值；
- v1.1 已生成 reliability diagram、coverage-risk 曲线和 Bootstrap CI；
- 尚未进行多随机种子稳健性实验；
- 当前 NaN -> negative 的 U-MultiClass 处理需要在论文方法部分明确说明。

建议后续：

1. 用多个随机种子重复训练和患者级划分；
2. 在固定 coverage 点比较 Baseline-Threshold 与 Agent；
3. 完成合成扰动及 Hard/Soft QC 追捕率曲线；
4. 增加独立外部验证数据；
5. 如继续研究 Model 2，增加独立验证数据或重新训练，不在 official valid 上继续调参。

## 17. v1.1 覆盖率空间诊断结果

v1.1 不再把固定概率阈值作为跨集合的主要工作点，而是在 route-validation 上按 DenseNet-121 calibrated confidence 排序，选择目标 coverage，再将相同 cutoff 应用到 official-valid。75% 工作点结果为：

| 数据集 | cutoff | observed coverage | DenseNet selective risk |
|---|---:|---:|---:|
| route-validation | 0.840923 | 75.00% | 3.84% |
| official-valid | 同一 cutoff | 77.72% | 17.83% |

这说明 confidence 排序具有一定迁移性，但 accepted 样本的风险在 official-valid 上明显升高，提示存在 selective-risk distribution shift。

在全体已知标签样本上，Bootstrap 结果显示：

- route-validation 中 Fusion 相对 DenseNet 的错误率差为 -1.58 个百分点，95% CI 为 [-1.88, -1.29]；
- official-valid 中 Fusion 相对 DenseNet 的错误率差为 +2.48 个百分点，95% CI 为 [-0.49, 5.47]，不能确认 Fusion 优势；
- route-validation 中 DenseNet 相对 ResNet 的错误率差为 +1.94 个百分点，ResNet 较好；
- official-valid 中 DenseNet 相对 ResNet 的错误率差为 -3.96 个百分点，DenseNet 较好，说明模型排名对数据分布敏感。

固定 75% coverage 后，route-validation 上 Fusion 与 DenseNet 的差异接近 0 且置信区间跨 0；official-valid 的 157 个 accepted 样本中三个模型的二分类错误计数恰好相同。该现象不构成模型等价证明，主要反映小样本和固定排序子集的限制。

因此 v1.1 的主结论是：coverage-based confidence ranking 比固定绝对概率阈值更适合描述跨集合迁移，但它不能消除 official-valid 上的风险分布偏移。ResNet-50 和 Fusion 保留为异构模型消融，不纳入主 Agent 决策。
