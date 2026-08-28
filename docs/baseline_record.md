# Cardiomegaly 基线记录

## 状态

尚未开始训练。数据已在本地准备完成；PyTorch 训练环境需在 AutoDL/Linux GPU 上配置。

## AutoDL 运行

```bash
python -m pip install -r requirements-autodl.txt
python src/train_baseline.py \
  --manifest data/splits/cardiomegaly_clean_patient_split.csv \
  --data-root data/raw/CheXpert-v1.0-small \
  --output-dir outputs/baseline_seed42
python src/evaluate.py \
  --manifest data/splits/cardiomegaly_clean_patient_split.csv \
  --data-root data/raw/CheXpert-v1.0-small \
  --checkpoint outputs/baseline_seed42/densenet121_cardiomegaly_best.pt \
  --split official_valid \
  --output outputs/baseline_seed42/official_valid_metrics.json
```

CSV 中若仍是 Windows 绝对路径，程序会自动回退到 `--data-root/Path` 解析 Linux 路径。

## 固定记录项

- 数据版本：
- manifest 文件：
- 标签处理：
- 图像尺寸：
- 模型：DenseNet-121
- 随机种子：42
- GPU：
- 训练时间：
- 峰值显存：
- AUROC：
- AUPRC：
- 灵敏度：
- 特异度：
- F1：
- Brier/ECE：
- 备注：
