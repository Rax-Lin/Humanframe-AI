# Composition Assistant

A deep learning system that scores the overall aesthetic quality of portrait photographs — evaluating lighting, tone, composition, and visual harmony.

Implementation in this repo now lives under `portraiq/` and is scaffolded with runnable training/inference entry points.

---

## Overview

Casual photographers often struggle to consistently produce visually strong portraits without formal training or curated feedback. This project addresses that gap by training a model to estimate overall portrait aesthetic quality and output a continuous score from **0.0 to 10.0**.

The scoring system combines two complementary sources:

- **AI-learned score** — trained from a large corpus of portrait images with human preference signals, learning broad aesthetic patterns

**Score scale:**

| Range | Rating |
|-------|--------|
| 0.0 – 5.0 | Poor |
| 5.0 – 7.5 | Acceptable |
| 7.5 – 9.0 | Good |
| 9.0 – 10.0 | Excellent |

These four levels represent overall aesthetic quality tiers rather than position-only composition quality.

---

## Project Structure

```
portraiq/
│
├── data/                            # Dataset (fully independent from all code)
│   ├── raw/                         # Original portrait images, untouched
│   │   ├── level1_poor/             # AVA score < 5.0
│   │   ├── level2_acceptable/       # AVA score 5.0–7.5
│   │   ├── level3_good/             # AVA score 7.5–9.0
│   │   └── level4_excellent/        # Pexels + Unsplash portraits for excellent class
│   ├── processed/                   # Training-ready portrait images (person-filtered)
│   │   ├── level1_poor/             # AVA score < 5.0
│   │   ├── level2_acceptable/       # AVA score 5.0–7.5
│   │   ├── level3_good/             # AVA score 7.5–9.0
│   │   └── level4_excellent/        # Pexels + Unsplash portraits for excellent class
│   ├── annotations/                 # Per-image score labels (includes level3_ffhq.json, level4_excellent.json, level4_smugmug.json)
│   ├── splits/                      # train / val / test index files
│   └── README.md                    # Data collection/filtering/annotation guide
│
├── models/                          # Model definitions & weights (shared by both pipelines)
│   ├── backbone/                    # Feature extractor (CLIP ViT-L/14)
│   ├── scorer/                      # Regression scoring head (outputs 0.0–10.0)
│   ├── checkpoints/                 # Saved .pth weight files — bridge between train & infer
│   └── configs/                     # YAML hyperparameter files per experiment
│
├── training/                        # ── TRAINING PIPELINE (standalone) ──
│   ├── dataset.py                   # PyTorch Dataset class definition
│   ├── train.py                     # Main training script (multi-GPU accelerated)
│   ├── evaluate.py                  # Validation & test set evaluation
│   ├── augment.py                   # Data augmentation pipeline
│   └── losses.py                    # Loss functions (Smooth L1 + optional variance penalty)
│
├── inference/                       # ── EXECUTION PIPELINE (standalone) ──
│   ├── predict.py                   # Single-image scoring entry point
│   ├── batch_predict.py             # Batch inference over a folder of images
│   ├── scoring_rules.py             # Removed — not applicable to aesthetic scoring model
│   └── visualize.py                 # Overlay score and prediction outputs
│
├── utils/                           # Shared utilities — imported by both pipelines
│   ├── image_utils.py               # Common Pillow / tensor transform helpers
│   ├── pose_utils.py                # Person detection wrapper (YOLO / MediaPipe)
│   ├── gpu_config.py                # CUDA device selection & memory management
│   └── logger.py                    # Training log & experiment tracking
│
├── main_train.py                    # ▶ TRAINING entry point — runs training pipeline only
├── main_infer.py                    # ▶ EXECUTION entry point — scores input images only
├── requirements_train.txt           # Dependencies for training (includes tensorboard, etc.)
├── requirements_infer.txt           # CPU-friendly inference dependencies
├── requirements_infer_full.txt      # Full inference dependencies (YOLO + CLIP)
├── requirements_infer_rpi4.txt      # Raspberry Pi helper deps (torch installed separately)
├── config.yaml                      # Global project configuration
├── config_train_mobilenet.yaml      # Lightweight training config (EfficientNet-B4 profile for Raspberry Pi deployment)
├── config_infer_rpi4.yaml           # Raspberry Pi 4 CPU inference config
└── README.md
```

Repository root also includes `.gitignore` to exclude datasets, checkpoints, cache files, and local runtime outputs from GitHub.

### Two Independent Entry Points

| Script | Purpose | Requires |
|--------|---------|---------|
| `main_train.py` | Train the model, output `.pth` checkpoints | `requirements_train.txt` |
| `main_infer.py` | Score new portrait images using saved weights | `requirements_infer.txt` (CPU-friendly) |

The **only connection** between the two pipelines is `models/checkpoints/` — the trained `.pth` weight files. The execution environment does not need any training dependencies installed.

---

## Module Design Principles

### Separation of concerns

Training code (`training/`) and inference code (`inference/`) are **completely separated**. Both import shared logic from `models/` and `utils/` via standard Python imports — no logic is duplicated. This allows the inference pipeline to be shipped independently without pulling in training dependencies.

### Dataset independence

The `data/` directory is self-contained. Labels are stored as JSON in `annotations/` alongside image paths, making it straightforward to swap datasets or add new annotation rounds without touching any model or training code.

### GPU acceleration

All CUDA configuration is centralized in `utils/gpu_config.py`. The current workstation runs **1× NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)** with CUDA 12.9 (driver 575.64). GPU memory allocation, mixed-precision training (`torch.cuda.amp`), optional data parallelism, and device fallback to CPU are all handled in one place. With this setup, CLIP backbones are practical with moderate batch sizes and input resolutions.

---

## Model Architecture

### Backbone — Feature Extractor

| Option | Parameters | Checkpoint Size | Input Size | Notes |
|--------|------------|-----------------|------------|-------|
| MobileNetV3-Large | ~5M | ~44MB | 224×224 | Previous lightweight baseline |
| EfficientNet-B2 | ~9M | ~80MB | 260×260 | Intermediate option |
| **EfficientNet-B4** | **~19M** | **~176MB** | **380×380** | **Recommended Pi deployment profile (target <200MB)** |
| CLIP ViT-L/14 | ~307M | ~1.1GB | 224×224 | High-accuracy workstation model |

The recommended approach is to use a **pretrained CLIP visual encoder** as the backbone. On the current RTX 4070 SUPER setup, start with ViT-B/32 for faster iteration and move to ViT-L/14 for final accuracy (with smaller batches or gradient accumulation). ViT-L/14 produces richer spatial feature representations that better capture global portrait aesthetics. Only the scoring head is trained from scratch; the backbone is fine-tuned with a low learning rate.

### Scoring Head — Regression

A lightweight-but-higher-capacity regression head attached to the backbone output:

```
Backbone Features → Linear(D, 512) → ReLU → Dropout(0.3)
                 → Linear(512, 256) → ReLU → Dropout(0.2)
                 → Linear(256, 64) → ReLU → Dropout(0.1)
                 → Linear(64, 1) → Clamp(0, 10)
```

`D` depends on backbone (`clip_vit_l14=768`, `clip_vit_b32=512`, `efficientnet_b2=1408`, `efficientnet_b4=1792`, `mobilenet_v3_large=960`) and is selected automatically in `models/backbone/factory.py`. Output is a single float in [0.0, 10.0].

For Raspberry Pi deployment, the EfficientNet-B4 profile uses `380×380` input resolution and targets approximately **176MB** checkpoints (kept under **200MB**) for improved accuracy while remaining deployable on Pi 4.

### Training Strategy

Current training uses:

- **Two-phase freeze/unfreeze training**:  
  Phase 1 (`epoch 1..freeze_epochs`) freezes backbone parameters and trains only the scoring head.  
  Phase 2 (`freeze_epochs+1..end`) unfreezes the backbone and continues fine-tuning with `backbone_lr_scale`.
- **Gradient accumulation** (`gradient_accumulation_steps: 4`) to simulate larger effective batch size without exceeding VRAM limits.  
  Effective batch size = `batch_size × gradient_accumulation_steps` (for example, `8 × 4 = 32`).
- **Weighted sampling** across `level1_poor`, `level2_acceptable`, `level3_good`, and `level4_excellent` to reduce class-imbalance bias from AVA-heavy levels.
- **Smooth L1 (Huber) loss** for stronger robustness to noisy/outlier score labels.
- **Variance penalty regularization** (`variance_penalty: 0.1`) to discourage collapsed predictions around the mean.
- **CosineAnnealingLR** schedule from initial LR down to `1e-6` across total epochs.
- **Early stopping** on validation MAE (`early_stopping_patience: 7`), with countdown paused during Phase 1 and active from Phase 2 onward.
- **Validation prediction-distribution logging** per epoch (`pred_score_min/mean/max/std`) to monitor range compression.

### Scoring Pipeline (Inference)

```
Input Image
    │
    ▼
Person Detection (optional utility)
    │
    └──▶ AI Model Score                ← backbone + scorer
             Learned aesthetic prediction
    │
    ▼
Final Score = AI_score
    │
    ▼
Visualize & Output (visualize.py)
```

---

## Dataset Strategy

All data collection, filtering, and annotation instructions were moved to:

- `portraiq/data/README.md`

This includes AVA two-stage collection, FFHQ/Level-4 source collection, annotation format, and data troubleshooting.

---

## Environment & Requirements

### Hardware

- **GPU:** 1× NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)
- **CUDA:** 12.9
- **Driver:** 575.64
- **Multi-GPU:** single-GPU by default; keep `multi_gpu: false` unless hardware changes
- **RAM:** 32 GB minimum recommended (64 GB+ for large dataset caching)

### Dependencies

Two separate requirements files keep the execution environment lean:

### Deployment Targets

1. **RTX 4070 SUPER workstation (Training + Inference):** current development target for training and inference.
2. **Raspberry Pi 4 (Inference only):** run inference in a dedicated Python virtual environment (`venv`) to avoid system Python package conflicts (PEP 668).

### Local RTX 4070 Workflow (Recommended)

Project root:
```bash
cd ~/emb_project/Humanframe-AI
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install training dependencies (CUDA-enabled PyTorch, compatible versions):
```bash
python -m pip install --no-cache-dir numpy==1.26.4
python -m pip install --no-cache-dir \
  torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir opencv-python==4.10.0.84
python -m pip install -r portraiq/requirements_train.txt
python -m pip install datasets
```

Verify CUDA is available:
```bash
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

Dataset collection commands (AVA/FFHQ/Level-4 sources) are documented in:
`portraiq/data/README.md`

Install the dependency set based on your target:

- Training environment:
```bash
python -m pip install -r portraiq/requirements_train.txt
```
- CPU-friendly inference:
```bash
python -m pip install -r portraiq/requirements_infer.txt
```
- Full inference (YOLO + CLIP):
```bash
python -m pip install -r portraiq/requirements_infer_full.txt
```
- Raspberry Pi helper dependencies:
```bash
python -m pip install -r portraiq/requirements_infer_rpi4.txt
```

Raspberry Pi note:
- On Raspberry Pi OS, do not install into system Python directly.
- Create/activate a project venv first, then install requirements inside that venv.

---

## Quick Start

Training commands below assume local workstation execution with `.venv` activated.

### ▶ Training Pipeline (`main_train.py`)

Before training, verify dataset visibility and split counts:
```bash
cd ~/emb_project/Humanframe-AI/portraiq
source ../.venv/bin/activate
ls -lah data/annotations
python - << 'PY'
from training.dataset import load_annotation_records, split_records
from pathlib import Path
r = load_annotation_records(Path("data/annotations"))
tr, va, te = split_records(r, 0.8, 0.1, 0.1, 42)
print("total:", len(r), "train:", len(tr), "val:", len(va), "test:", len(te))
PY
```

Expected: `train > 0`. If `total: 0`, training will fail because no labels were loaded.
Current loader behavior: non-training JSON files like `ava_raw_download.json` and `ava_skipped_errors.json` are skipped automatically.

Train the model and save checkpoints to `models/checkpoints/`:

```bash
cd ~/emb_project/Humanframe-AI/portraiq
source ../.venv/bin/activate
python main_train.py --config config.yaml
```

Train a Raspberry Pi-friendly checkpoint (EfficientNet-B4 backbone):
```bash
cd ~/emb_project/Humanframe-AI/portraiq
source ../.venv/bin/activate
python main_train.py --config config_train_mobilenet.yaml
```

Resume from a checkpoint:
```bash
cd ~/emb_project/Humanframe-AI/portraiq
source ../.venv/bin/activate
python main_train.py --config config.yaml --resume models/checkpoints/last.pth
```

Run evaluation on the test set:
```bash
cd ~/emb_project/Humanframe-AI/portraiq
source ../.venv/bin/activate
python main_train.py --config config.yaml --mode evaluate --checkpoint models/checkpoints/best.pth
```

### TensorBoard Monitoring

Launch TensorBoard on the workstation:
```bash
cd ~/emb_project/Humanframe-AI/portraiq
tensorboard --logdir=runs/ --host=0.0.0.0 --port=6006
```

Open in your local browser:
```text
http://<workstation-ip>:6006
```

### ▶ Execution Pipeline (`main_infer.py`)

RTX 4070 SUPER workstation (single-image inference):
```bash
cd ~/emb_project/Humanframe-AI/portraiq
python3 main_infer.py --image path/to/photo.jpg --checkpoint models/checkpoints/best.pth
```

RTX 4070 SUPER workstation (batch inference):
```bash
cd ~/emb_project/Humanframe-AI/portraiq
python3 main_infer.py --input_dir ./photos/ --output_dir ./results/ --checkpoint models/checkpoints/best.pth
```

Raspberry Pi 4 CPU mode (venv-based, no container):
```bash
cd /home/pi/Humanframe-AI/portraiq
sudo apt update
sudo apt install -y python3-venv python3-full
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_infer_rpi4.txt
python main_infer.py \
  --config config_infer_rpi4.yaml \
  --image path/to/photo.jpg \
  --checkpoint models/checkpoints/efficientnet_b4/best.pth \
  --cpu_optimized \
  --no_overlay
```

If `pip` in the venv reports resolver/import errors (for example `InconsistentCandidate`), rebuild the venv:
```bash
cd /home/pi/Humanframe-AI/portraiq
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_infer_rpi4.txt
```
Expected CPU latency on Raspberry Pi 4 with EfficientNet-B4 is approximately **2–5 seconds per image** (depends on SD card, thermal throttling, and background load).

### Training Data Troubleshooting (`total: 0`)

See `portraiq/data/README.md` for dataset/annotation troubleshooting steps.

---

## Configuration Files

Use the provided config files directly instead of copying settings from README:

- `portraiq/config.yaml` (default training/inference on workstation)
- `portraiq/config_train_mobilenet.yaml` (EfficientNet-B4 lightweight training profile)
- `portraiq/config_infer_rpi4.yaml` (Raspberry Pi inference profile)

If you need custom hyperparameters, duplicate one of the YAML files and pass it with `--config`.

---

## Roadmap

- [x] Project structure design
- [x] Initial project scaffold and module implementation (`portraiq/`)
- [ ] Dataset collection & annotation pipeline
- [x] Person detection integration (`pose_utils.py`, YOLO + fallback)
- [x] Backbone integration (EfficientNet / CLIP)
- [x] Pure AI scoring pipeline (rule-based scoring removed from train/val/infer)
- [x] Baseline training pipeline with multi-GPU support (`DataParallel`)
- [x] Baseline evaluation metrics (MAE, RMSE)
- [x] Inference CLI & visualization output
- [x] Stronger experiment tracking / logging (TensorBoard wiring)
- [ ] Collect additional high-quality portrait dataset beyond current Level 4 sources (Pexels + Unsplash) to improve Excellent-range accuracy
- [ ] Web demo (optional)

---

## License

MIT License. Dataset usage is subject to each source's individual terms of service.
