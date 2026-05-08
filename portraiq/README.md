# Portraiq

Implementation scaffold for the Composition Assistant described in the parent project `Readme.md`.

## Docker (Recommended)

### 1) Training + Inference (GPU)

Build:
```bash
docker build -f Dockerfile.train_infer -t portraiq:train-infer .
```

Run:
```bash
docker run --gpus all --rm -it \
  -v "$PWD":/workspace/portraiq \
  -w /workspace/portraiq \
  portraiq:train-infer bash
```

Inside container:
```bash
python - << 'PY'
from training.dataset import load_annotation_records, split_records
from pathlib import Path
r = load_annotation_records(Path("data/annotations"))
tr, va, te = split_records(r, 0.8, 0.1, 0.1, 42)
print("total:", len(r), "train:", len(tr), "val:", len(va), "test:", len(te))
PY
python main_train.py --config config.yaml
python main_infer.py --image path/to/photo.jpg --checkpoint models/checkpoints/best.pth
```

### 2) Inference Only (CPU)

Build:
```bash
docker build -f Dockerfile.infer_cpu -t portraiq:infer-cpu .
```

Run:
```bash
docker run --rm -it \
  -v "$PWD":/workspace/portraiq \
  -w /workspace/portraiq \
  portraiq:infer-cpu bash
```

Inside container:
```bash
python main_infer.py --image path/to/photo.jpg --checkpoint models/checkpoints/mobilenet_best.pth --detector none --cpu_optimized
```

Raspberry Pi 4 (64-bit OS):
```bash
docker buildx build --platform linux/arm64 -f Dockerfile.infer_cpu -t portraiq:infer-cpu-arm64 .
```

## No-Sudo Cluster Fallback (Apptainer)

GPU shell:
```bash
apptainer shell --nv \
  --bind "$PWD":/workspace/portraiq \
  docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

CPU shell:
```bash
apptainer shell \
  --bind "$PWD":/workspace/portraiq \
  docker://python:3.10-slim
```

## Manual Fallback (if Docker unavailable)

Training environment:
```bash
pip install -r requirements_train.txt
```

CPU inference environment:
```bash
pip install -r requirements_infer.txt
```

Raspberry Pi helper deps:
```bash
pip install -r requirements_infer_rpi4.txt
```
