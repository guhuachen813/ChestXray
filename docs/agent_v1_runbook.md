# Agent v1 AutoDL Runbook

## 1. Agent split

```bash
python src/prepare_manifest.py --data-root /root/autodl-tmp/CheXpert-v1.0-small --output-dir data/manifests/agent --include-lateral
python src/make_agent_split.py \
  --manifest data/manifests/agent/cardiomegaly_all.csv \
  --output-dir data/splits --seed 42
```

## 2. Train both models on `model_train` and validate on `route_validation`

```bash
python src/train_baseline.py --manifest data/splits/cardiomegaly_agent_split.csv --data-root /root/autodl-tmp/CheXpert-v1.0-small --output-dir outputs/agent_dense_seed42 --arch densenet121 --num-classes 3 --split-column agent_split --train-split model_train --val-split model_selection --image-size 320 --batch-size 16 --epochs 10 --num-workers 4 --seed 42
python src/train_baseline.py --manifest data/splits/cardiomegaly_agent_split.csv --data-root /root/autodl-tmp/CheXpert-v1.0-small --output-dir outputs/agent_resnet_seed42 --arch resnet50 --num-classes 3 --split-column agent_split --train-split model_train --val-split model_selection --image-size 320 --batch-size 16 --epochs 10 --num-workers 4 --seed 42
```

## 3. Calibration and QC features

```bash
python src/calibrate.py --manifest data/splits/cardiomegaly_agent_split.csv --data-root /root/autodl-tmp/CheXpert-v1.0-small --checkpoint outputs/agent_dense_seed42/best.pt --output outputs/agent_dense_seed42/calibration.json
python src/calibrate.py --manifest data/splits/cardiomegaly_agent_split.csv --data-root /root/autodl-tmp/CheXpert-v1.0-small --checkpoint outputs/agent_resnet_seed42/best.pt --output outputs/agent_resnet_seed42/calibration.json
python src/quality_features.py --manifest data/splits/cardiomegaly_agent_split.csv --data-root /root/autodl-tmp/CheXpert-v1.0-small --output outputs/qc_agent_features.csv
```

## 4. Evaluate models and route

Use `--split-column agent_split` for this manifest. Route thresholds must be selected on `route_validation`, then frozen before evaluating `official_valid`.

```bash
python src/route_agent.py --manifest data/splits/cardiomegaly_agent_split.csv --data-root /root/autodl-tmp/CheXpert-v1.0-small --model1 outputs/agent_dense_seed42/best.pt --model2 outputs/agent_resnet_seed42/best.pt --split route_validation --qc outputs/qc_agent_features.csv --output outputs/agent_route_route_validation.csv
python src/route_agent.py --manifest data/splits/cardiomegaly_agent_split.csv --data-root /root/autodl-tmp/CheXpert-v1.0-small --model1 outputs/agent_dense_seed42/best.pt --model2 outputs/agent_resnet_seed42/best.pt --split official_valid --qc outputs/qc_agent_features.csv --output outputs/agent_route_official_valid.csv
```

The current route runner accepts explicit temperatures (`--temperature1`, `--temperature2`); use the values saved by `calibrate.py`. It writes one row per image with trigger, action, model probabilities, disagreement, and Hard QC reason.
