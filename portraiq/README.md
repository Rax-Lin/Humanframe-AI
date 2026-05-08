# Portraiq

Implementation scaffold for the Composition Assistant described in the parent project `Readme.md`.

## Deployment Modes

1. V100 workstation (training + inference): Apptainer/Singularity (no sudo required).
2. Raspberry Pi 4 (inference only): direct pip install (no container required).

## V100 Workstation (Apptainer)

Launch GPU shell:
```bash
cd /home/rax10101010/img_project/portraiq
apptainer shell --nv \
  --bind "$PWD":/workspace/portraiq \
  docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

Inside shell:
```bash
cd /workspace/portraiq
python -m pip install --user -r requirements_train.txt
python main_train.py --config config.yaml
python main_infer.py --image path/to/photo.jpg --checkpoint models/checkpoints/best.pth
```

## Raspberry Pi 4 (Pip, No Container)

Install deps:
```bash
python -m pip install -r requirements_infer_rpi4.txt
```

Run inference:
```bash
python main_infer.py \
  --config config_infer_rpi4.yaml \
  --image path/to/photo.jpg \
  --checkpoint models/checkpoints/mobilenet_best.pth \
  --detector none \
  --cpu_optimized \
  --no_overlay
```
