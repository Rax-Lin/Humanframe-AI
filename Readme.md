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
| 7.5 – 9.5 | Good |
| 9.5 – 10.0 | Excellent |

These four levels represent overall aesthetic quality tiers rather than position-only composition quality.

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
│   │   └── level4_excellent/        # Pexels + Unsplash portraits for excellent class
│   ├── processed/                   # Training-ready portrait images (person-filtered)
│   │   ├── level1_poor/             # AVA score < 5.0
│   │   ├── level2_acceptable/       # AVA score 5.0–7.0
│   │   ├── level3_good/             # AVA score >= 7.0
│   │   └── level4_excellent/        # Pexels + Unsplash portraits for excellent class
│   ├── annotations/                 # Per-image score labels (includes level3_ffhq.json, level4_excellent.json, level4_smugmug.json)
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

All CUDA configuration is centralized in `utils/gpu_config.py`. The current workstation runs **1× NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)** with CUDA 12.9 (driver 575.64). GPU memory allocation, mixed-precision training (`torch.cuda.amp`), optional data parallelism, and device fallback to CPU are all handled in one place. With this setup, CLIP backbones are practical with moderate batch sizes and input resolutions.

---

## Model Architecture

### Backbone — Feature Extractor

| Option | Parameters | VRAM (per GPU) | Notes |
|--------|-----------|----------------|-------|
| EfficientNet-B2 | ~9M | < 2 GB | Lightweight baseline; useful for rapid iteration |
| MobileNetV3-Large | ~5M | < 1 GB | Fastest inference; suitable for real-time use |
| CLIP ViT-B/32 | ~86M | ~4 GB | Strong compositional semantics; good accuracy/speed balance |
| **CLIP ViT-L/14** | **~307M** | **~10 GB** | **Best accuracy; feasible on RTX 4070 SUPER with reduced batch size** |
| CLIP ViT-L/14@336 | ~307M | ~14 GB | Higher resolution input; usually requires very small batches on 12 GB GPUs |

The recommended approach is to use a **pretrained CLIP visual encoder** as the backbone. On the current RTX 4070 SUPER setup, start with ViT-B/32 for faster iteration and move to ViT-L/14 for final accuracy (with smaller batches or gradient accumulation). ViT-L/14 produces richer spatial feature representations that better capture global portrait aesthetics. Only the scoring head is trained from scratch; the backbone is fine-tuned with a low learning rate.

### Scoring Head — Regression

A lightweight regression head attached to the backbone output:

```
Backbone Features → Linear(D, 256) → ReLU → Dropout(0.3) → Linear(256, 1) → Clamp(0, 10)
```

`D` depends on backbone (`clip_vit_l14=768`, `clip_vit_b32=512`, `efficientnet_b2=1408`, `mobilenet_v3=960`) and is selected automatically in `models/backbone/factory.py`. Output is a single float in [0.0, 10.0].

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

Update: Level-4 training photos now also include SmugMug-collected portraits (high-quality editorial/public portfolio style) after YOLO person filtering.

### Source Scope (Simplified)

This project uses five sources for dataset construction:

1. **AVA Dataset** for Level 1 to Level 3:
   - `level1_poor/` -> AVA score <= 4.5
   - `level2_acceptable/` -> AVA score 5.0–7.0
   - `level3_good/` -> AVA score >= 7.0
2. **FFHQ Dataset (Hugging Face)** for Level 3:
   - downloaded from `datasets/ffhq` via `huggingface_hub`
   - saved under `data/raw/level3_good/ffhq/`, then person-filtered into `data/processed/level3_good/`
   - fixed score `8.0` (`annotator: ffhq_fixed`) because FFHQ portraits are generally high-quality but not necessarily award-winning aesthetics
   - source images are typically `1024x1024` PNG and are resized by the existing training preprocessing pipeline
3. **Pexels API** for Level 4:
   - `level4_excellent/` -> Pexels portrait queries + person filtering, fixed score 9.5
4. **Unsplash API** for Level 4:
   - portrait queries + person filtering, fixed score 9.5
   - only images with `width >= 800` and `height >= 800` are considered before download
5. **SmugMug ImageSearch API** for Level 4:
   - portrait keyword queries + person filtering, fixed score 9.5
   - requires `SMUGMUG_API_KEY` in environment
AVA is used as the primary source because it provides large-scale human aesthetic preference ratings that directly align with this project’s aesthetic quality objective.

For AVA samples, vote-distribution agreement filtering is applied:
- Keep as-is when `score_std < 1.0` (high annotator consensus).
- Keep with smoothing when `1.0 <= score_std < 1.5`:  
  `smoothed_score = raw_score * 0.9 + 5.0 * 0.1`
- Discard when `score_std >= 1.5` (high annotator disagreement).

The computed `score_std` is stored in annotation records alongside `score`.

### Expected Dataset Size After Filtering (Guideline)

| Level | Primary Sources | Typical Scale After Person Filtering |
|-------|------------------|--------------------------------------|
| Level 1 (Poor) | AVA | depends on AVA split/filter |
| Level 2 (Acceptable) | AVA | depends on AVA split/filter |
| Level 3 (Good) | AVA + FFHQ | AVA Level-3 + approximately **50,000–70,000** additional FFHQ portraits |
| Level 4 (Excellent) | Pexels + Unsplash + SmugMug | typically adds approximately **3,000–5,000** SmugMug portraits after YOLO filtering, plus other Level-4 sources |

### Labeling Strategy

For bootstrapping with limited manual effort:

1. Collect ~300–500 images manually labeled on the 0–10 scale
2. Train an initial model on this seed set
3. Run the model on unlabeled images and review high-confidence predictions
4. Iteratively expand the labeled set (human-in-the-loop)

For `level4_excellent/` labels, use a stable fixed value (`9.5`) instead of random assignment.

For FFHQ Level-3 labels, use a stable fixed value (`8.0`) to represent consistently strong portrait quality without forcing them into the top-most excellent tier.

### Level 3 Collection (FFHQ)

Run in project root:
```bash
cd ~/emb_project/Humanframe-AI
source .venv/bin/activate
```

Foreground run:
```bash
python3 data_collect/ffhq_download.py \
  --max_images 70000 \
  --person_conf 0.25 \
  --resume
```

Optional explicit repo selection:
```bash
python3 data_collect/ffhq_download.py \
  --repo_id marcosv/ffhq-dataset \
  --max_images 70000 \
  --person_conf 0.25 \
  --resume
```

This writes:
- Raw images: `portraiq/data/raw/level3_good/ffhq/`
- Processed images: `portraiq/data/processed/level3_good/`
- Annotations: `portraiq/data/annotations/level3_ffhq.json`

### Level 4 Collection (Unified: Pexels + Unsplash)

Run in project root:
```bash
cd ~/emb_project/Humanframe-AI
source .venv/bin/activate
```

Set API keys in environment:
```bash
export PEXELS_API_KEY="<your_pexels_api_key>"
export UNSPLASH_ACCESS_KEY="<your_unsplash_access_key>"
```
Use `--source all` only when both keys are set.

Foreground runs:
```bash
# Download from both sources
python3 data_collect/level4_download.py --source all --max_images 5000 --resume

# Download from Pexels only
python3 data_collect/level4_download.py --source pexels --max_images 5000 --resume

# Download from Unsplash only
python3 data_collect/level4_download.py --source unsplash --max_images 1000 --resume
```

Background run (recommended for long jobs):
```bash
mkdir -p logs
nohup python3 data_collect/level4_download.py \
  --source all \
  --max_images 5000 \
  --per_keyword 2000 \
  --person_conf 0.25 \
  --resume \
  > logs/level4_download.log 2>&1 &
echo $! > logs/level4_download.pid
```

Monitor / stop:
```bash
tail -f logs/level4_download.log
kill "$(cat logs/level4_download.pid)"
```

This writes:
- Raw images: `portraiq/data/raw/level4_excellent/`
- Processed images: `portraiq/data/processed/level4_excellent/`
- Annotations: `portraiq/data/annotations/level4_excellent.json`

### Level 4 Collection (SmugMug)

Set API key:
```bash
export SMUGMUG_API_KEY="your_key_here"
```

Foreground run:
```bash
python3 data_collect/smugmug_download.py \
  --max_images 5000 \
  --per_keyword 1500 \
  --scope /api/v2/user/cmac \
  --discover_scopes \
  --max_scopes 50 \
  --person_conf 0.25 \
  --resume
```

This writes:
- Raw images: `portraiq/data/raw/level4_excellent/`
- Processed images: `portraiq/data/processed/level4_excellent/`
- Annotations: `portraiq/data/annotations/level4_smugmug.json`

### Annotation JSON Format

```json
{
  "image_id": "img_0042",
  "filename": "level2_acceptable/portrait_042.jpg",
  "category": "portrait",
  "score": 6.3,
  "score_std": 0.82,
  "annotator": "ava",
  "split": "train"
}
```

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
2. **Raspberry Pi 4 (Inference only):** direct pip install with `requirements_infer_rpi4.txt` (no container needed).

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

AVA dataset preparation is now recommended as a two-stage pipeline to reduce CPU spikes and improve stability:

1) Stage A (`download_raw`): download AVA images + score metadata into `data/raw/` and `data/annotations/ava_raw_download.json` (no person filtering).
2) Stage B (`build_processed`): read raw manifest, run person filtering, copy kept samples to `data/processed/`, and write `data/annotations/ava_from_hf.json`.

Important:
Run Stage A and Stage B sequentially.
Do not run them at the same time.
Wait for Stage A to finish before starting Stage B.

Stage A (download only):
```bash
python3 data_collect/ava_download.py \
  --mode download_raw \
  --clean-levels \
  --clean-annotations \
  --save-every 200
```

Stage A finished check:
```bash
ls -lh portraiq/data/annotations/ava_raw_download.json
```

Stage B (raw -> processed with person filtering):
```bash
python3 data_collect/ava_download.py \
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
python3 data_collect/ava_download.py \
  --mode build_processed \
  --person-filter \
  --person-backend torchvision \
  --person-device cuda:0 \
  --person-imgsz 384 \
  --throttle-ms 20 \
  --resume \
  --save-every 200
```

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

Train a Raspberry Pi-friendly checkpoint (MobileNetV3 backbone):
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
2. You are in the correct working directory (`~/emb_project/Humanframe-AI/portraiq` for local runs).
3. JSON shape matches project format (single object or list of objects, each with `filename` and `score`).
4. If you use explicit split labels, at least some records must have `"split": "train"`.
5. Image files referenced by `filename` are placed under `data/processed/`.

---

## Configuration Files

Use the provided config files directly instead of copying settings from README:

- `portraiq/config.yaml` (default training/inference on workstation)
- `portraiq/config_train_mobilenet.yaml` (MobileNet training profile)
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
