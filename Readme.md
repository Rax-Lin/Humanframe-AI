# Composition Assistant

A deep learning system that scores the compositional quality of portrait photographs — evaluating how well a person is positioned within a scene, taking into account background elements, spatial harmony, and established photography composition principles.

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
│   ├── processed/                   # Resized & normalized images (Pillow / OpenCV pipeline)
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
│   ├── scoring_rules.py             # Rule-based composition scoring (fixed component)
│   └── visualize.py                 # Overlay score & heatmap on output image
│
├── utils/                           # Shared utilities — imported by both pipelines
│   ├── image_utils.py               # Common Pillow / OpenCV / scikit-image helpers
│   ├── pose_utils.py                # Person detection wrapper (YOLO / MediaPipe)
│   ├── gpu_config.py                # CUDA device selection & memory management
│   └── logger.py                    # Training log & experiment tracking
│
├── main_train.py                    # ▶ TRAINING entry point — runs training pipeline only
├── main_infer.py                    # ▶ EXECUTION entry point — scores input images only
├── requirements_train.txt           # Dependencies for training (includes tensorboard, etc.)
├── requirements_infer.txt           # Minimal dependencies for execution only
├── config.yaml                      # Global project configuration
└── README.md
```

### Two Independent Entry Points

| Script | Purpose | Requires |
|--------|---------|---------|
| `main_train.py` | Train the model, output `.pth` checkpoints | `requirements_train.txt` |
| `main_infer.py` | Score new portrait images using saved weights | `requirements_infer.txt` (lighter) |

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
GlobalAveragePool → Linear(1024, 256) → ReLU → Dropout(0.3) → Linear(256, 1) → Sigmoid × 10
```

The input dimension is 1024 for CLIP ViT-L/14 (vs. 512 for ViT-B/32 — adjust in `models/configs/` if switching backbones). Output is a single float in [0.0, 10.0].

### Scoring Pipeline (Inference)

```
Input Image
    │
    ▼
Person Detection (pose_utils.py)       ← YOLO / MediaPipe
    │
    ├──▶ Rule-based Score              ← scoring_rules.py
    │        Rule of thirds, headroom,
    │        subject-to-frame ratio
    │
    └──▶ AI Model Score                ← backbone + scorer
             Learned positional harmony
    │
    ▼
Final Score = α × rule_score + (1−α) × ai_score
    │
    ▼
Visualize & Output (visualize.py)
```

The blending weight `α` is configurable in `config.yaml` and defaults to `0.3`.

---

## Dataset Strategy

### Recommended Sources

| Source | Type | Usage |
|--------|------|-------|
| **AVA Dataset** (Google Research) | 250k images with aesthetic scores | Primary training data — filter for portrait images |
| **Unsplash Lite** | Professional photography | High-score positive samples |
| **OpenImages v7** | Everyday snapshots | Low/mid-score reference samples |
| **CUHK Person Re-ID / Market-1501** | Person-in-scene images | Diverse positional variety for training |

### Labeling Strategy

For bootstrapping with limited manual effort:

1. Collect ~300–500 images manually labeled on the 0–10 scale
2. Train an initial model on this seed set
3. Run the model on unlabeled images and review high-confidence predictions
4. Iteratively expand the labeled set (human-in-the-loop)

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

**`requirements_train.txt`** — full training environment:
```
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
Pillow>=10.2.0
scikit-image>=0.22.0
ultralytics>=8.1.0        # YOLO for person detection
open-clip-torch>=2.24.0   # CLIP ViT-L/14 backbone
pyyaml>=6.0
tqdm>=4.66.0
tensorboard>=2.16.0       # training only
```

**`requirements_infer.txt`** — minimal execution environment:
```
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
Pillow>=10.2.0
ultralytics>=8.1.0        # YOLO for person detection
open-clip-torch>=2.24.0   # CLIP ViT-L/14 backbone
pyyaml>=6.0
tqdm>=4.66.0
```

Install for training:
```bash
pip install -r requirements_train.txt
```

Install for execution only:
```bash
pip install -r requirements_infer.txt
```

---

## Quick Start

### ▶ Training Pipeline (`main_train.py`)

Train the model and save checkpoints to `models/checkpoints/`:

```bash
python main_train.py --config config.yaml
```

Resume from a checkpoint:
```bash
python main_train.py --config config.yaml --resume models/checkpoints/last.pth
```

Run evaluation on the test set:
```bash
python main_train.py --config config.yaml --mode evaluate --checkpoint models/checkpoints/best.pth
```

### ▶ Execution Pipeline (`main_infer.py`)

Score a single portrait image:
```bash
python main_infer.py --image path/to/photo.jpg --checkpoint models/checkpoints/best.pth
```

Score a folder of images (batch):
```bash
python main_infer.py --input_dir ./photos/ --output_dir ./results/ --checkpoint models/checkpoints/best.pth
```

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

scoring:
  alpha: 0.3                     # Weight for rule-based score (1-alpha for AI score)

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
- [ ] Dataset collection & annotation pipeline
- [ ] Person detection integration (`pose_utils.py`)
- [ ] Backbone integration (EfficientNet / CLIP)
- [ ] Rule-based scoring implementation (`scoring_rules.py`)
- [ ] Training pipeline with 8× V100 multi-GPU support (DataParallel / DDP)
- [ ] Evaluation metrics (MAE, RMSE, score-band accuracy)
- [ ] Inference CLI & visualization output
- [ ] Web demo (optional)

---

## License

MIT License. Dataset usage is subject to each source's individual terms of service.