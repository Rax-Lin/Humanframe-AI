# Composition Assistant

A deep learning system that scores the compositional quality of portrait photographs — evaluating how well a person is positioned within a scene, taking into account background elements, spatial harmony, and established photography composition principles.

Implementation in this repo now lives under `portraiq/` and is scaffolded with runnable training/inference entry points.

---

## Overview

Casual photographers often struggle to capture well-composed portrait shots without formal training in framing, background selection, and subject placement. This project addresses that gap by training a model to evaluate *person-in-scene* positioning quality and output a continuous score from **0.0 to 10.0**.

The scoring system combines two complementary sources:

- **Rule-based fixed score** — derived from classical composition theory (rule of thirds, subject-to-background ratio, headroom/lead room)
- **AI-learned score** — trained from a large corpus of professional and amateur portrait images, learning positional harmony between the subject and background

**Score scale:**

| Range | Rating |
|-------|--------|
| 0.0 – 5.0 | Poor |
| 5.0 – 7.5 | Acceptable |
| 7.5 – 9.5 | Good |
| 9.5 – 10.0 | Excellent |

---

## Project Structure

```
portraiq/
│
├── data/                            # Dataset (fully independent from all code)
│   ├── raw/                         # Original portrait images, untouched
│   │   ├── level1_poor/             # AVA score < 5.0
│   │   ├── level2_acceptable/       # AVA score 5.0–7.0
│   │   ├── level3_good/             # AVA score >= 7.0
│   │   └── level4_excellent/        # Award-winning set, fixed score 9.5–10.0
│   ├── processed/                   # Training-ready portrait images (person-filtered)
│   │   ├── level1_poor/             # AVA score < 5.0
│   │   ├── level2_acceptable/       # AVA score 5.0–7.0
│   │   ├── level3_good/             # AVA score >= 7.0
│   │   └── level4_excellent/        # Award-winning set, fixed score 9.5–10.0
│   ├── annotations/                 # Per-image score labels in JSON format
│   └── splits/                      # train / val / test index files
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
│   └── losses.py                    # Loss function definitions (MSE / smooth L1)
│
├── inference/                       # ── EXECUTION PIPELINE (standalone) ──
│   ├── predict.py                   # Single-image scoring entry point
│   ├── batch_predict.py             # Batch inference over a folder of images
│   ├── scoring_rules.py             # Legacy reference only (not used in active scoring)
│   └── visualize.py                 # Overlay score, rule-of-thirds grid, and subject box
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
├── config_train_mobilenet.yaml      # Training config for Raspberry Pi deployment backbone
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

All CUDA configuration is centralized in `utils/gpu_config.py`. The workstation runs **8× Tesla V100-SXM2 (32 GB VRAM each, 256 GB total)** with CUDA 12.2. GPU memory allocation, mixed-precision training (`torch.cuda.amp`), multi-GPU data parallelism (`torch.nn.DataParallel` or `DistributedDataParallel`), and device fallback to CPU are all handled in one place. With this setup, large backbone models (ViT-L/14, CLIP ViT-L) and high batch sizes (256+) become feasible.

---

## Model Architecture

### Backbone — Feature Extractor

| Option | Parameters | VRAM (per GPU) | Notes |
|--------|-----------|----------------|-------|
| EfficientNet-B2 | ~9M | < 2 GB | Lightweight baseline; useful for rapid iteration |
| MobileNetV3-Large | ~5M | < 1 GB | Fastest inference; suitable for real-time use |
| CLIP ViT-B/32 | ~86M | ~4 GB | Strong compositional semantics; good accuracy/speed balance |
| **CLIP ViT-L/14** | **~307M** | **~10 GB** | **Recommended — best accuracy; fully feasible on V100 32GB** |
| CLIP ViT-L/14@336 | ~307M | ~14 GB | Higher resolution input; marginal gain over ViT-L/14 |

The recommended approach is to use a **pretrained CLIP ViT-L/14** visual encoder as the backbone. This is a significant upgrade over ViT-B/32 and is only practical with high-VRAM GPUs — the 8× V100 (32 GB each) makes it straightforward. ViT-L/14 produces richer spatial feature representations that better capture composition-relevant details such as subject placement relative to background geometry. Only the scoring head is trained from scratch; the backbone is fine-tuned with a low learning rate.

### Scoring Head — Regression

A lightweight regression head attached to the backbone output:

```
Backbone Features → Linear(D, 256) → ReLU → Dropout(0.3) → Linear(256, 1) → Sigmoid × 10
```

`D` depends on backbone (`clip_vit_l14=768`, `clip_vit_b32=512`, `efficientnet_b2=1408`, `mobilenet_v3=960`) and is selected automatically in `models/backbone/factory.py`. Output is a single float in [0.0, 10.0].

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

The blending weight `α` is configurable in `config.yaml` and defaults to `0.3`.

---

## Dataset Strategy

### Source Scope (Simplified)

This project uses only two sources for dataset construction:

1. **AVA Dataset** for Level 1 to Level 3:
   - `level1_poor/` -> AVA score <= 4.5
   - `level2_acceptable/` -> AVA score 5.0–7.0
   - `level3_good/` -> AVA score >= 7.0
2. **Award-winning photography** (1x.com, Sony World Photography Awards) for Level 4 only:
   - `level4_excellent/` -> manually collected, fixed score 9.5–10.0

Before placing images into `data/raw/level*/` and `data/processed/level*/`, OpenCV/YOLO person detection is used to filter portrait images from each source.

### Labeling Strategy

For bootstrapping with limited manual effort:

1. Collect ~300–500 images manually labeled on the 0–10 scale
2. Train an initial model on this seed set
3. Run the model on unlabeled images and review high-confidence predictions
4. Iteratively expand the labeled set (human-in-the-loop)

For `level4_excellent/` labels, use stable fixed values (for example `9.7`, `9.8`, or a manual split such as `9.5` / `10.0`) instead of random assignment.

### Annotation JSON Format

```json
{
  "image_id": "img_0042",
  "filename": "portrait_042.jpg",
  "category": "portrait",
  "score": 7.4,
  "sub_scores": {
    "rule_of_thirds": 8.0,
    "headroom": 7.5,
    "background_complexity": 6.8,
    "subject_frame_ratio": 7.2
  },
  "annotator": "human",
  "split": "train"
}
```

---

## Environment & Requirements

### Hardware

- **GPU:** 8× NVIDIA Tesla V100-SXM2 (32 GB VRAM each) — 256 GB total VRAM
- **CUDA:** 12.2
- **Driver:** 535.161.08
- **Multi-GPU:** `DataParallel` for single-node multi-GPU training; `DistributedDataParallel` recommended for large-scale runs
- **RAM:** 32 GB minimum recommended (64 GB+ for large dataset caching)

### Dependencies

Two separate requirements files keep the execution environment lean:

### Deployment Targets

1. **V100 workstation (Training + Inference):** Apptainer/Singularity is the only container method available (no sudo, Docker unavailable).
2. **Raspberry Pi 4 (Inference only):** direct pip install with `requirements_infer_rpi4.txt` (no container needed).

### Apptainer/Singularity (Primary, No Sudo Required)

On managed HPC servers, use Apptainer for training/inference.

GPU training/inference shell:
```bash
cd /home/rax10101010/img_project
apptainer shell --nv \
  --bind "$PWD":/workspace/img_project \
  docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

Inside that shell:
```bash
cd /workspace/img_project/portraiq
python -m pip install --user -r requirements_train.txt
# Needed for AVA dataset download/prep scripts
python -m pip install --user datasets pillow tqdm
python main_train.py --config config.yaml
```

AVA dataset preparation is now recommended as a two-stage pipeline to reduce CPU spikes and improve stability:

1) Stage A (`download_raw`): download AVA images + score metadata into `data/raw/` and `data/annotations/ava_raw_download.json` (no person filtering).
2) Stage B (`build_processed`): read raw manifest, run person filtering, copy kept samples to `data/processed/`, and write `data/annotations/ava_from_hf.json`.

Important:
Run Stage A and Stage B sequentially.
Do not run them at the same time.
Wait for Stage A to finish before starting Stage B.

Stage A (download only):
```bash
python /workspace/img_project/data_collect/ava_download.py \
  --mode download_raw \
  --clean-levels \
  --clean-annotations \
  --save-every 200
```

Stage A finished check:
```bash
ls -lh /workspace/img_project/portraiq/data/annotations/ava_raw_download.json
```

Stage B (raw -> processed with person filtering):
```bash
python /workspace/img_project/data_collect/ava_download.py \
  --mode build_processed \
  --person-filter \
  --person-backend torchvision \
  --person-device cuda:0 \
  --person-imgsz 384 \
  --throttle-ms 20 \
  --clean-levels \
  --clean-annotations \
  --save-every 200
```

If Stage B gets interrupted, resume without cleaning:
```bash
python /workspace/img_project/data_collect/ava_download.py \
  --mode build_processed \
  --person-filter \
  --person-backend torchvision \
  --person-device cuda:0 \
  --person-imgsz 384 \
  --throttle-ms 20 \
  --resume \
  --save-every 200
```

CPU inference shell:
```bash
cd portraiq
apptainer shell \
  --bind "$PWD":/workspace/portraiq \
  docker://python:3.10-slim
```

Inside that shell:
```bash
cd /workspace/portraiq
python -m pip install --user --upgrade pip
python -m pip install --user torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --user -r requirements_infer.txt
```

**`requirements_train.txt`** — full training environment:
```
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0,<2
opencv-python>=4.9.0
Pillow>=10.2.0
scikit-image>=0.22.0
ultralytics>=8.1.0        # YOLO for person detection
open-clip-torch>=2.24.0   # CLIP ViT-L/14 backbone
pyyaml>=6.0
tqdm>=4.66.0
tensorboard>=2.16.0       # training only
```

**`requirements_infer.txt`** — CPU-friendly execution environment:
```
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0,<2
Pillow>=10.2.0
pyyaml>=6.0
tqdm>=4.66.0
```

**`requirements_infer_full.txt`** — full inference (optional YOLO + CLIP):
```
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0,<2
Pillow>=10.2.0
pyyaml>=6.0
tqdm>=4.66.0
ultralytics>=8.1.0
open-clip-torch>=2.24.0
```

**`requirements_infer_rpi4.txt`** — Raspberry Pi helper dependencies:
```
numpy>=1.24.0,<2
Pillow>=10.2.0
pyyaml>=6.0
tqdm>=4.66.0
```

Manual install snippets are still available in `portraiq/README.md` if needed.

---

## Quick Start

Training commands below assume you are inside the Apptainer shell.

### ▶ Training Pipeline (`main_train.py`)

Before training, verify dataset visibility and split counts:
```bash
cd /workspace/portraiq
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

Train the model and save checkpoints to `models/checkpoints/`:

```bash
cd /workspace/portraiq
python main_train.py --config config.yaml
```

Train a Raspberry Pi-friendly checkpoint (MobileNetV3 backbone):
```bash
cd /workspace/portraiq
python main_train.py --config config_train_mobilenet.yaml
```

Resume from a checkpoint:
```bash
cd /workspace/portraiq
python main_train.py --config config.yaml --resume models/checkpoints/last.pth
```

Run evaluation on the test set:
```bash
cd /workspace/portraiq
python main_train.py --config config.yaml --mode evaluate --checkpoint models/checkpoints/best.pth
```

### TensorBoard Monitoring

Launch TensorBoard on the workstation:
```bash
cd /workspace/portraiq
tensorboard --logdir=runs/ --host=0.0.0.0 --port=6006
```

Open in your local browser:
```text
http://<workstation-ip>:6006
```

### ▶ Execution Pipeline (`main_infer.py`)

V100 workstation (inside Apptainer shell) single-image inference:
```bash
cd /workspace/portraiq
python main_infer.py --image path/to/photo.jpg --checkpoint models/checkpoints/best.pth
```

V100 workstation (inside Apptainer shell) batch inference:
```bash
cd /workspace/portraiq
python main_infer.py --input_dir ./photos/ --output_dir ./results/ --checkpoint models/checkpoints/best.pth
```

Raspberry Pi 4 CPU mode (pip-based, no container):
```bash
cd /home/pi/portraiq
python -m pip install -r requirements_infer_rpi4.txt
python main_infer.py \
  --config config_infer_rpi4.yaml \
  --image path/to/photo.jpg \
  --checkpoint models/checkpoints/mobilenet_best.pth \
  --cpu_optimized \
  --no_overlay
```

### Training Data Troubleshooting (`total: 0`)

If your check prints `total: 0 train: 0 val: 0 test: 0`, verify:

1. JSON files exist under `data/annotations/` inside the running environment.
2. You are in the correct working directory (`/workspace/portraiq` in Apptainer shell).
3. JSON shape matches project format (single object or list of objects, each with `filename` and `score`).
4. If you use explicit split labels, at least some records must have `"split": "train"`.
5. Image files referenced by `filename` are placed under `data/processed/`.

---

## Configuration (`config.yaml`)

```yaml
model:
  backbone: clip_vit_l14         # Options: efficientnet_b2 | mobilenet_v3 | clip_vit_b32 | clip_vit_l14
  checkpoint: null               # Path to pretrained weights, null = train from scratch
  backbone_lr_scale: 0.1         # Fine-tune backbone with 10× lower LR than head

training:
  epochs: 50
  batch_size: 256                # 8× V100 allows large batches; 32 per GPU × 8 GPUs
  learning_rate: 3.0e-4          # Scale with batch size vs. baseline 1e-4 at bs=32
  weight_decay: 1.0e-5
  mixed_precision: true          # torch.cuda.amp — V100 supports FP16 natively
  multi_gpu: true                # Enable DataParallel / DistributedDataParallel
  num_gpus: 8                    # Set to available GPU count (nvidia-smi: 8× V100)

data:
  image_size: 336                # ViT-L/14@336 input; use 224 for standard ViT-L/14
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  num_workers: 16                # More workers to feed 8 GPUs without bottleneck

gpu:
  device: cuda                   # cuda | cpu
  cudnn_benchmark: true
  visible_devices: "0,1,2,3,4,5,6,7"   # All 8 V100s
```

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
- [ ] DistributedDataParallel (DDP) training path
- [ ] Stronger experiment tracking / logging (TensorBoard wiring)
- [ ] Web demo (optional)

---

## License

MIT License. Dataset usage is subject to each source's individual terms of service.
