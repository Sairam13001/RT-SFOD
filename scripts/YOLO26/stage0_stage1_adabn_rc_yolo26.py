import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml

# Ensure this script uses the local editable Ultralytics checkout, not site-packages.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO


MODEL_FAMILY = "YOLO26"
MODEL_TAG = "yolo26"


# Some older local checkpoints may contain this class in the pickle namespace.
class DetectInputFeatureHook:
    def _hook_fn(self, *args, **kwargs):
        pass


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def val_device_arg(device: torch.device) -> str:
    return str(device.index if device.type == "cuda" and device.index is not None else device)


def list_images_from_yaml(data_yaml: str) -> list[str]:
    with open(data_yaml, "r") as f:
        y = yaml.safe_load(f)

    root = Path(y.get("path", "."))
    train = y["train"]
    train_path = root / train
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = []

    if train_path.is_file():
        with open(train_path, "r") as tf:
            for line in tf:
                p = Path(line.strip())
                if p.suffix.lower() in exts and p.exists():
                    imgs.append(str(p))
    else:
        for dp, _, files in os.walk(train_path):
            for fn in files:
                if Path(fn).suffix.lower() in exts:
                    imgs.append(str(Path(dp) / fn))

    imgs.sort()
    return imgs


class TargetImgDataset(data.Dataset):
    """Target-domain image loader for YOLO26 AdaBN/RC.

    The adaptation is architecture-agnostic: no labels are loaded and no YOLO26
    layer names are assumed. YOLO26-specific behavior, including end-to-end
    heads if present, comes from the checkpoint passed through --weights.
    """

    def __init__(self, img_paths: list[str], img_size: int = 640, strong_aug: bool = False):
        self.imgs = img_paths
        self.img_size = img_size
        self.strong_aug = strong_aug

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.imgs[idx]
        im = cv2.imread(path)
        assert im is not None, f"Failed to read {path}"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        h, w = im.shape[:2]
        scale = self.img_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        top = (self.img_size - nh) // 2
        left = (self.img_size - nw) // 2
        canvas[top : top + nh, left : left + nw] = imr
        im = canvas

        if random.random() < 0.5:
            im = np.ascontiguousarray(np.fliplr(im))

        if random.random() < 0.8:
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
            if self.strong_aug:
                hgain = (random.random() * 0.4 - 0.2) * 180
                sgain = (random.random() * 0.6 - 0.3) * 255
                vgain = (random.random() * 0.6 - 0.3) * 255
            else:
                hgain = (random.random() * 0.2 - 0.1) * 180
                sgain = (random.random() * 0.2 - 0.1) * 255
                vgain = (random.random() * 0.2 - 0.1) * 255
            hsv[..., 0] = np.clip(hsv[..., 0] + hgain, 0, 179)
            hsv[..., 1] = np.clip(hsv[..., 1] + sgain, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + vgain, 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if self.strong_aug and random.random() < 0.5:
            alpha = 0.8 + random.random() * 0.4
            beta = -20 + random.random() * 40
            im = np.clip(alpha * im.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return torch.from_numpy(im).permute(2, 0, 1).float() / 255.0


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


def dump_bn_priors_from_model(model: nn.Module, out_path: Path) -> dict[str, list]:
    """Stage 0: save BN running statistics from the source-trained YOLO26 checkpoint."""
    state_dict = model.state_dict()
    bn_priors = {
        k: v.detach().cpu().tolist()
        for k, v in state_dict.items()
        if k.endswith("running_mean") or k.endswith("running_var")
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bn_priors, f)
    print(f"[Stage 0] Saved {MODEL_FAMILY} BN priors to {out_path} ({len(bn_priors)} tensors)")
    return bn_priors


def bn_priors_to_device(bn_priors: dict[str, list], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in bn_priors.items()}


def blend_bn_running_stats(model: nn.Module, bn_priors: dict[str, torch.Tensor], alpha: float) -> int:
    n_applied = 0
    n_bn_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            n_bn_layers += 1
            mean_key = f"{name}.running_mean" if name else "running_mean"
            var_key = f"{name}.running_var" if name else "running_var"

            if mean_key in bn_priors and hasattr(module, "running_mean"):
                with torch.no_grad():
                    module.running_mean.copy_((1.0 - alpha) * module.running_mean + alpha * bn_priors[mean_key])
                n_applied += 1

            if var_key in bn_priors and hasattr(module, "running_var"):
                with torch.no_grad():
                    module.running_var.copy_((1.0 - alpha) * module.running_var + alpha * bn_priors[var_key])
                n_applied += 1

    if n_applied == 0 and n_bn_layers > 0:
        print(f"[WARN] Found {n_bn_layers} BN layers, but no BN priors matched checkpoint keys.")
    return n_applied


def set_bn_train_no_grad(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.train()
    for param in model.parameters():
        param.requires_grad = False


def restore_bn_buffers(model: nn.Module, state_dict: dict[str, torch.Tensor], device: torch.device) -> None:
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            for attr in ("running_mean", "running_var", "num_batches_tracked"):
                key = f"{name}.{attr}" if name else attr
                if key in state_dict and hasattr(module, attr):
                    with torch.no_grad():
                        getattr(module, attr).copy_(state_dict[key].to(device))


def validate(model_wrapper: YOLO, net: nn.Module, args, device: torch.device, epoch_idx: int) -> float:
    model_wrapper.model = net
    metrics = model_wrapper.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch * 2,
        conf=0.001,
        iou=0.6,
        device=val_device_arg(device),
        verbose=False,
        plots=False,
    )
    map50 = metrics.box.map50 if hasattr(metrics.box, "map50") else 0.0
    print(f"[Validation] Epoch {epoch_idx}: mAP50={map50:.4f}")
    set_bn_train_no_grad(net)
    return float(map50)


def main(args) -> None:
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] {MODEL_FAMILY} combined Stage0+Stage1")
    print(f"[Setup] Using device: {device}")
    print(f"[Setup] Loading weights: {args.weights}")
    model_wrapper = YOLO(args.weights)
    net = model_wrapper.model.to(device).float()

    bn_priors_path = Path(args.bn_priors_out) if args.bn_priors_out else out_dir / f"{MODEL_TAG}_stage0_bn_priors.json"
    bn_priors_cpu = dump_bn_priors_from_model(net, bn_priors_path)
    bn_priors = bn_priors_to_device(bn_priors_cpu, device)

    imgs = list_images_from_yaml(args.data)
    assert len(imgs) > 0, f"No target training images found in {args.data}"
    dataset = TargetImgDataset(imgs, img_size=args.imgsz, strong_aug=args.strong_aug)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    set_bn_train_no_grad(net)
    best_map50 = -1.0
    best_epoch = -1
    best_state_dict = None

    def run_epoch(epoch_idx: int, blend_alpha: float) -> bool:
        nonlocal best_map50, best_epoch, best_state_dict
        set_bn_train_no_grad(net)
        t0 = time.time()
        n_img = 0
        for ims in dataloader:
            ims = ims.to(device, non_blocking=True)
            with torch.no_grad():
                _ = net(ims)
            n_img += ims.shape[0]

        n_applied = blend_bn_running_stats(net, bn_priors, blend_alpha) if blend_alpha > 0.0 else 0
        print(
            f"[Epoch {epoch_idx}] imgs={n_img} blend_alpha={blend_alpha:.4f} "
            f"bn_blended_tensors={n_applied} time={time.time() - t0:.1f}s"
        )

        if args.early_stop and (epoch_idx % args.val_interval == 0):
            map50 = validate(model_wrapper, net, args, device, epoch_idx)
            if map50 > best_map50:
                best_map50 = map50
                best_epoch = epoch_idx
                best_state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                print(f"[Validation] New best mAP50={best_map50:.4f} at epoch {best_epoch}")
            if epoch_idx - best_epoch >= args.patience:
                print(f"[Early Stop] No mAP50 improvement for {args.patience} epochs.")
                return True
        return False

    print("[Stage 1A] AdaBN")
    stopped = False
    for epoch in range(args.epochs_adabn):
        stopped = run_epoch(epoch, blend_alpha=0.0)
        if stopped:
            break

    if not stopped and args.epochs_rc > 0:
        print("[Stage 1B] Roto-Calibration")
        for rc_epoch in range(args.epochs_rc):
            if args.epochs_rc <= 1:
                alpha_t = args.alpha0
            else:
                alpha_t = args.alpha0 * 0.5 * (1.0 + math.cos(math.pi * rc_epoch / (args.epochs_rc - 1)))
            stopped = run_epoch(args.epochs_adabn + rc_epoch, blend_alpha=alpha_t)
            if stopped:
                break

    if args.early_stop and best_state_dict is not None:
        print(f"[Early Stop] Restoring best BN buffers from epoch {best_epoch} (mAP50={best_map50:.4f})")
        restore_bn_buffers(net, best_state_dict, device)

    dataset_name = Path(args.data).stem
    ckpt_path = out_dir / f"{MODEL_TAG}_stage1_adabnrc_{dataset_name}.pt"
    model_wrapper.model = net
    model_wrapper.save(str(ckpt_path))
    print(f"[Stage 1] Saved adapted {MODEL_FAMILY} checkpoint to: {ckpt_path}")

    if args.eval:
        print(f"[Eval] Comparing source vs adapted {MODEL_FAMILY} on {dataset_name}")
        source_model = YOLO(args.weights)
        source_metrics = source_model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch * 2,
            conf=0.001,
            iou=0.6,
            device=val_device_arg(device),
            verbose=True,
            plots=False,
            save_json=False,
            save_hybrid=False,
        )

        adapted_model = YOLO(str(ckpt_path))
        adapted_metrics = adapted_model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch * 2,
            conf=0.001,
            iou=0.6,
            device=val_device_arg(device),
            verbose=True,
            plots=True,
            save_json=False,
            save_hybrid=False,
        )

        rows = [
            ("source", source_metrics.box.map, source_metrics.box.map50, source_metrics.box.map75),
            ("stage1_adapted", adapted_metrics.box.map, adapted_metrics.box.map50, adapted_metrics.box.map75),
        ]
        metrics_csv = out_dir / f"{MODEL_TAG}_stage1_evaluation_comparison.csv"
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "map", "map50", "map75"])
            for row in rows:
                writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])
        print(f"[Eval] Saved comparison metrics to: {metrics_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 combined Stage0 BN-prior dump + Stage1 AdaBN/RC")
    parser.add_argument("--weights", type=str, required=True, help="Source-trained YOLO26 checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Target-domain data YAML")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--bn_priors_out", type=str, default=None, help="Optional path for saved Stage0 BN priors JSON")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs_adabn", type=int, default=2)
    parser.add_argument("--epochs_rc", type=int, default=0, help="Roto-Calibration epochs. Default 0 disables RC.")
    parser.add_argument("--alpha0", type=float, default=0.10)
    parser.add_argument("--strong_aug", action="store_true", help="Use stronger target-domain augmentations")
    parser.add_argument("--eval", action="store_true", help="Evaluate source and adapted checkpoints")
    parser.add_argument("--early_stop", action="store_true", help="Validate during adaptation and restore best BN buffers")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default="0", help="Local CUDA index after CUDA_VISIBLE_DEVICES, or cpu")
    main(parser.parse_args())
