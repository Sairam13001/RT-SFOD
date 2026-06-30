# RT-SFOD: Real-Time Source-Free Object Detection

Official implementation of **ECCV 2026** paper **RT-SFOD**.

RT-SFOD adapts real-time dual-head, NMS-free object detectors to an unlabeled target domain without access to source-domain images. The method uses an AdaBN warm start, Dual-Head pseudo-label Fusion (DHF), and Multi-scale Adaptive Representation Diversification (MARD).

## Overview

RT-SFOD is designed for real-time source-free object detection. Given a source-trained dual-head detector and unlabeled target images, the method performs target adaptation in two phases:

1. **Stage 0 + Stage 1: AdaBN warm start**
   - Extract source batch-normalization priors.
   - Adapt batch-normalization statistics on target-domain images.
   - Produce an AdaBN-initialized checkpoint for self-training.

2. **Stage 2: Mean-teacher adaptation**
   - Teacher predicts pseudo-labels on weakly augmented target images.
   - Student learns from strongly augmented target images.
   - Teacher is updated once per epoch using EMA.

The full RT-SFOD objective combines detection loss with MARD:

`L = L_det + lambda(t) * L_mard`

## Method Components

### Dual-Head Pseudo-Label Fusion

DHF uses the complementary behavior of the one-to-one and one-to-many heads:

- O2O predictions provide high-precision anchor boxes.
- O2M predictions improve recall.
- O2M boxes are added only when they are non-overlapping with O2O anchors.
- Class-wise NMS is applied only to the selected O2M extras.

### Multi-scale Adaptive Representation Diversification

MARD regularizes multi-scale PAN features during student training. It samples foreground tokens from pseudo boxes and background tokens from valid non-object regions, then applies variance and covariance objectives to preserve discriminative feature diversity.

## Installation

`conda create -n rtsfod python=3.10 -y`

`conda activate rtsfod`

`pip install -e .`

Use the same PyTorch/CUDA setup as your local YOLO/Ultralytics environment.

## Data Format

Datasets should follow the standard Ultralytics YOLO data YAML format:

`path: /path/to/dataset`

`train: images/train`

`val: images/val`

`names: {0: person, 1: rider, 2: car}`

For source-free adaptation, only unlabeled target-domain training images are used during adaptation.

## Training

### Stage 0 + Stage 1: AdaBN Warm Start

`python scripts/YOLO26/stage0_stage1_adabn_rc_yolo26.py --weights <SOURCE_CHECKPOINT.pt> --data <TARGET_DATA.yaml> --out_dir runs/yolo26m/stage1/ --imgsz 1024 --batch 16 --workers 8 --epochs_adabn 2 --epochs_rc 0 --device 0`

This creates a Stage 1 checkpoint such as:

`runs/yolo26m/stage1/yolo26_stage1_adabnrc_<dataset_name>.pt`

### Stage 2: Mean Teacher + DHF, Without MARD

This is the no-MARD ablation.

`python scripts/YOLO26/stage2_rtsfod_yolo26.py --stage1_model runs/yolo26m/stage1/yolo26_stage1_adabnrc_<dataset_name>.pt --data <TARGET_DATA.yaml> --out_dir runs/yolo26m/stage2_no_mard/ --imgsz 1024 --batch 16 --workers 8 --epochs 60 --lr 1e-4 --tau_o2o 0.5 --tau_o2m 0.5 --tau_no 0.2 --tau_dup 0.7 --mard_lambda0 0.0 --ema_momentum 0.999 --device 0`

### Stage 2: Full RT-SFOD With MARD

`python scripts/YOLO26/stage2_rtsfod_yolo26.py --stage1_model runs/yolo26m/stage1/yolo26_stage1_adabnrc_<dataset_name>.pt --data <TARGET_DATA.yaml> --out_dir runs/yolo26m/stage2_mard/ --imgsz 1024 --batch 16 --workers 8 --epochs 60 --lr 1e-4 --tau_o2o 0.5 --tau_o2m 0.5 --tau_no 0.2 --tau_dup 0.7 --mard_lambda0 0.05 --mard_lambda_max 0.2 --mard_gamma 1.0 --mard_alpha 1.0 --mard_beta 0.1 --mard_warmup_epochs 5 --mard_gate_threshold 0.5 --mard_topk_boxes 15 --mard_fg_points 8 --mard_bg_points 128 --mard_eta 12.0 --ema_momentum 0.999 --device 0`


## Repository Structure

- `scripts/YOLO26/stage0_stage1_adabn_rc_yolo26.py`: AdaBN warm start
- `scripts/YOLO26/stage2_rtsfod_yolo26.py`: Paper-aligned RT-SFOD Stage 2
- `scripts/YOLO26/stage2_mard_yolo26_end2end.py`: Earlier experimental Stage 2 script
- `scripts/YOLO26/validate_stage2_student_checkpoints.py`: Checkpoint validation
- `ultralytics/`: Detector implementation

## Citation

If you use this code, please cite:

`@inproceedings{vcr2026rtsfod,
  title={Real-Time Source-Free Object Detection},
  author={VCR, Sairam and Gopal, Varun and Jain, Poornima and NB, Vineeth},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}`

## Acknowledgements

This repository builds on the Ultralytics YOLO codebase. Please also follow the original Ultralytics license and citation requirements.

## License

Please see `LICENSE` for details.
