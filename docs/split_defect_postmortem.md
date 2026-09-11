# Agent 划分缺陷与重跑说明

## 缺陷

旧版 `make_agent_split.py` 先在每个患者标签组内打乱，再按标签组顺序拼接列表，最后对拼接结果做全局切片。这个过程不是分层随机抽样，会让前缀切片被某一类患者占据。

因此，旧版 Agent 的 calibration、route_validation 和 model_selection 可能具有异常患病率。由这些集合拟合的 temperature、选择的 checkpoint、confidence cutoff 和 coverage-risk 结果均不能作为最终证据。

## 修复

`stratified_patient_split_v2` 按患者级 Cardiomegaly 众数标签独立分层，在每个标签层内分配四个集合，并在写出 CSV 前重新打乱行顺序。没有观察到有效 Cardiomegaly 标签的患者只保留在 `model_train`，不参与 holdout 边界的分层抽样；其数量会写入报告。

## 结果处理规则

旧的 `outputs/agent_*`、`outputs/v11` 和多 seed 结果保留用于缺陷诊断，但必须标记为 `invalid_split_diagnostic`。修复后的结果使用独立的 `data/splits/repaired/` 和 `outputs/repaired_seed*/` 目录，不能覆盖旧 checkpoint 或指标。

## 重跑要求

必须先检查 `agent_split_report.json` 中各集合的已知二分类阳性率和两两患者重叠，再训练模型。temperature 和 75% coverage cutoff 只能使用修复后的 calibration 和 route_validation 选择；official_valid 只能用于最终冻结确认。
