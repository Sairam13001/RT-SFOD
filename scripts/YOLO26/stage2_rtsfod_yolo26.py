"""Stage 2 RT-SFOD adaptation for YOLO26 end-to-end detectors.

This script implements the ECCV 2026 RT-SFOD Stage 2 methodology:
mean-teacher self-training, Dual-Head pseudo-label Fusion (DHF), and
Multi-scale Adaptive Representation Diversification (MARD). The defaults match
the paper's implementation details and Table S.1/S.2.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou


DEFAULT_IMGSZ = 1024
DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-4
DEFAULT_GRAD_CLIP = 10.0

TAU_O2O = 0.5
TAU_O2M = 0.5
TAU_NO = 0.2
TAU_DUP = 0.7

MARD_LAMBDA0 = 0.05
MARD_LAMBDA_MAX = 0.2
MARD_GAMMA = 1.0
MARD_ALPHA = 1.0
MARD_BETA = 0.1
MARD_WARMUP_EPOCHS = 5.0
MARD_GATE_THRESHOLD = 0.5
MARD_TOPK_BOXES = 15
MARD_FG_POINTS = 8
MARD_BG_POINTS = 128
MARD_ETA = 12.0
MARD_INTERVAL = 1

EMA_MOMENTUM = 0.999
MARD_BOX_CONF_THRESHOLD = 0.5


def seed_everything(seed: int, deterministic: bool = False) -> None:
    if seed is None or seed < 0:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            warnings.warn(f"Could not enable deterministic algorithms: {exc}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def val_device_arg(device: torch.device) -> str:
    return str(device.index if device.type == "cuda" and device.index is not None else device)


def scalarize(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().mean().item())
    return float(x)


def make_divisible(x: int, divisor: int = 32) -> int:
    return int(math.ceil(float(x) / divisor) * divisor)


def list_images_from_yaml(data_yaml: str) -> list[str]:
    with open(data_yaml, "r") as f:
        y = yaml.safe_load(f)

    root = Path(y.get("path", "."))
    train = y["train"]
    train_path = root / train
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs: list[str] = []

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


class TargetTeacherStudentDataset(data.Dataset):
    """Unlabeled target images with weak and strong RT-SFOD augmentations."""

    def __init__(self, img_paths: list[str], img_size: int = DEFAULT_IMGSZ):
        self.imgs = img_paths
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.imgs)

    def resize_long_edge(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized, {"scale": scale, "final_size": (nh, nw), "flipped": False}

    def weak_view(self, img: np.ndarray, flip: bool) -> tuple[np.ndarray, dict]:
        img, info = self.resize_long_edge(img)
        if flip:
            img = np.ascontiguousarray(np.fliplr(img))
        info["flipped"] = flip
        return np.ascontiguousarray(img), info

    def strong_view(self, img: np.ndarray, flip: bool) -> tuple[np.ndarray, dict]:
        img, info = self.resize_long_edge(img)
        if flip:
            img = np.ascontiguousarray(np.fliplr(img))
        h, w = img.shape[:2]

        scale_translate_matrix = None
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            tx = int(random.uniform(-0.05, 0.05) * w)
            ty = int(random.uniform(-0.05, 0.05) * h)
            scale_translate_matrix = np.float32([[scale, 0, tx], [0, scale, ty]])
            img = cv2.warpAffine(img, scale_translate_matrix, (w, h), borderValue=(114, 114, 114))

        perspective_matrix = None
        if random.random() < 0.3:
            distortion = int(round(0.03 * min(w, h)))
            src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst = np.float32(
                [
                    [random.randint(-distortion, distortion), random.randint(-distortion, distortion)],
                    [w + random.randint(-distortion, distortion), random.randint(-distortion, distortion)],
                    [random.randint(-distortion, distortion), h + random.randint(-distortion, distortion)],
                    [w + random.randint(-distortion, distortion), h + random.randint(-distortion, distortion)],
                ]
            )
            perspective_matrix = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, perspective_matrix, (w, h), borderValue=(114, 114, 114))

        if random.random() < 0.8:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = np.clip(hsv[..., 0] + random.uniform(-0.15, 0.15) * 180, 0, 179)
            hsv[..., 1] = np.clip(hsv[..., 1] + random.uniform(-0.2, 0.2) * 255, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + random.uniform(-0.2, 0.2) * 255, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if random.random() < 0.6:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.3)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            img = cv2.LUT(img, table)

        if random.random() < 0.2:
            order = [0, 1, 2]
            random.shuffle(order)
            img = img[:, :, order]

        if random.random() < 0.4:
            kernel = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            img = cv2.GaussianBlur(img, (kernel, kernel), sigma)

        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            prob = random.uniform(0.01, 0.05)
            mask = np.random.random(img.shape[:2])
            img[mask < prob / 2] = 0
            img[mask > 1 - prob / 2] = 255

        info.update(
            {
                "flipped": flip,
                "scale_translate_matrix": scale_translate_matrix,
                "perspective_matrix": perspective_matrix,
            }
        )
        return np.ascontiguousarray(img), info

    def __getitem__(self, idx: int):
        path = self.imgs[idx]
        img = cv2.imread(path)
        assert img is not None, f"Failed to read {path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        flip = random.random() < 0.5
        weak_img, weak_info = self.weak_view(img.copy(), flip)
        strong_img, strong_info = self.strong_view(img.copy(), flip)

        weak_tensor = torch.from_numpy(weak_img).permute(2, 0, 1).float() / 255.0
        strong_tensor = torch.from_numpy(strong_img).permute(2, 0, 1).float() / 255.0
        return weak_tensor, strong_tensor, path, weak_info, strong_info


def collate_fn(batch):
    weak_imgs, strong_imgs, paths, weak_infos, strong_infos = zip(*batch)
    max_h = max(max(im.shape[1] for im in weak_imgs), max(im.shape[1] for im in strong_imgs))
    max_w = max(max(im.shape[2] for im in weak_imgs), max(im.shape[2] for im in strong_imgs))
    max_h = make_divisible(max_h, 32)
    max_w = make_divisible(max_w, 32)

    def pad(imgs):
        padded = []
        for im in imgs:
            c, h, w = im.shape
            canvas = torch.full((c, max_h, max_w), 114 / 255.0, dtype=im.dtype)
            canvas[:, :h, :w] = im
            padded.append(canvas)
        return torch.stack(padded, dim=0)

    return pad(weak_imgs), pad(strong_imgs), list(paths), list(weak_infos), list(strong_infos)


def transform_boxes_weak_to_strong(
    boxes: torch.Tensor,
    weak_info: dict,
    strong_info: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, torch.ones(0, dtype=torch.bool, device=boxes.device)

    assert weak_info["flipped"] == strong_info["flipped"], "Weak and strong flips must be shared."
    device = boxes.device
    boxes_np = boxes.detach().cpu().numpy()
    n = boxes_np.shape[0]

    corners = np.zeros((n, 4, 2), dtype=np.float32)
    corners[:, 0] = boxes_np[:, [0, 1]]
    corners[:, 1] = boxes_np[:, [2, 1]]
    corners[:, 2] = boxes_np[:, [2, 3]]
    corners[:, 3] = boxes_np[:, [0, 3]]

    matrix = strong_info.get("scale_translate_matrix")
    if matrix is not None:
        for i in range(n):
            for j in range(4):
                corners[i, j] = matrix @ np.array([corners[i, j, 0], corners[i, j, 1], 1.0], dtype=np.float32)

    matrix = strong_info.get("perspective_matrix")
    if matrix is not None:
        for i in range(n):
            for j in range(4):
                pt = matrix @ np.array([corners[i, j, 0], corners[i, j, 1], 1.0], dtype=np.float32)
                corners[i, j] = pt[:2] / max(pt[2], 1e-6)

    h_strong, w_strong = strong_info["final_size"]
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, 0] = corners[:, :, 0].min(axis=1)
    out[:, 1] = corners[:, :, 1].min(axis=1)
    out[:, 2] = corners[:, :, 0].max(axis=1)
    out[:, 3] = corners[:, :, 1].max(axis=1)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w_strong)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h_strong)

    valid = ((out[:, 2] - out[:, 0]) >= 2.0) & ((out[:, 3] - out[:, 1]) >= 2.0)
    return torch.from_numpy(out).to(device), torch.from_numpy(valid).to(device)


def ensure_yolo26_end2end_model(model: nn.Module, label: str) -> None:
    head = getattr(model, "model", [None])[-1]
    if not getattr(model, "end2end", False):
        raise ValueError(f"{label} must be an end-to-end YOLO26-style detector.")
    if head is None or "detect" not in head.__class__.__name__.lower():
        raise ValueError(f"{label} final module is not a Detect head: {type(head)}")
    if not hasattr(head, "one2one") or not hasattr(head, "one2many"):
        raise ValueError(f"{label} Detect head must expose one2one and one2many branches.")


def ensure_detection_loss_args(model: nn.Module, epochs: int) -> None:
    existing = getattr(model, "args", None)
    if existing is None:
        existing = argparse.Namespace()
    elif isinstance(existing, dict):
        existing = argparse.Namespace(**existing)

    defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": int(epochs)}
    for key, value in defaults.items():
        if not hasattr(existing, key) or getattr(existing, key) is None:
            setattr(existing, key, value)
    model.args = existing


def setup_teacher_student(checkpoint: str, device: torch.device):
    teacher_wrapper = YOLO(checkpoint)
    teacher_model = teacher_wrapper.model.to(device).float()
    ensure_yolo26_end2end_model(teacher_model, "Teacher")
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_wrapper = YOLO(checkpoint)
    student_model = student_wrapper.model.to(device).float()
    ensure_yolo26_end2end_model(student_model, "Student")
    for param in student_model.parameters():
        param.requires_grad = True

    return teacher_model, student_model, student_wrapper


def classwise_nms(boxes: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    kept = []
    classes = boxes[:, 5].long().unique(sorted=True)
    for cls in classes:
        idx = torch.where(boxes[:, 5].long() == cls)[0]
        keep = torchvision.ops.nms(boxes[idx, :4], boxes[idx, 4], iou_threshold)
        kept.append(boxes[idx[keep]])
    return torch.cat(kept, dim=0) if kept else boxes.new_zeros((0, 6))


def dual_head_fusion(
    boxes_one2one: Optional[torch.Tensor],
    boxes_one2many: Optional[torch.Tensor],
    device: torch.device,
    tau_o2o: float,
    tau_o2m: float,
    tau_no: float,
    tau_dup: float,
) -> torch.Tensor:
    if boxes_one2one is None or boxes_one2one.numel() == 0:
        anchors = torch.zeros((0, 6), device=device)
    else:
        anchors = boxes_one2one.to(device)
        anchors = anchors[anchors[:, 4] >= tau_o2o]

    if boxes_one2many is None or boxes_one2many.numel() == 0:
        candidates = torch.zeros((0, 6), device=device)
    else:
        candidates = boxes_one2many.to(device)
        candidates = candidates[candidates[:, 4] >= tau_o2m]

    if candidates.numel() == 0:
        fused = anchors
    elif anchors.numel() == 0:
        extras = classwise_nms(candidates, tau_dup)
        fused = extras
    else:
        max_iou = box_iou(candidates[:, :4], anchors[:, :4]).max(dim=1).values
        extras = candidates[max_iou <= tau_no]
        extras = classwise_nms(extras, tau_dup)
        fused = torch.cat([anchors, extras], dim=0) if extras.numel() else anchors

    if fused.numel():
        fused = fused[torch.argsort(fused[:, 4], descending=True)]
    return fused


@torch.no_grad()
def generate_pseudo_labels(
    teacher_model: nn.Module,
    weak_imgs: torch.Tensor,
    tau_o2o: float,
    tau_o2m: float,
    tau_no: float,
    tau_dup: float,
) -> list[torch.Tensor]:
    teacher_model.eval()
    outputs = teacher_model(weak_imgs, augment=False, visualize=False)
    if not (isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict)):
        raise ValueError("YOLO26 teacher must return (one2one_predictions, branch_predictions) in eval mode.")

    final_one2one, branch_preds = outputs
    head = teacher_model.model[-1]
    one2many_decoded = head._inference(branch_preds["one2many"]).permute(0, 2, 1)
    final_one2many = head.postprocess(one2many_decoded)

    labels = []
    for i in range(weak_imgs.shape[0]):
        boxes_o2o = final_one2one[i]
        boxes_o2m = final_one2many[i]
        boxes_o2o = boxes_o2o[boxes_o2o[:, 4] > 0] if boxes_o2o.numel() else None
        boxes_o2m = boxes_o2m[boxes_o2m[:, 4] > 0] if boxes_o2m.numel() else None
        labels.append(dual_head_fusion(boxes_o2o, boxes_o2m, weak_imgs.device, tau_o2o, tau_o2m, tau_no, tau_dup))
    return labels


def map_pseudo_labels_to_strong(
    pseudo_labels: list[torch.Tensor],
    weak_infos: list[dict],
    strong_infos: list[dict],
) -> list[torch.Tensor]:
    mapped = []
    for labels, weak_info, strong_info in zip(pseudo_labels, weak_infos, strong_infos):
        if labels.numel() == 0:
            mapped.append(labels)
            continue
        boxes, valid = transform_boxes_weak_to_strong(labels[:, :4], weak_info, strong_info)
        if valid.any():
            mapped.append(torch.cat([boxes[valid], labels[valid, 4:6]], dim=1))
        else:
            mapped.append(labels.new_zeros((0, 6)))
    return mapped


def compute_student_loss(student_outputs, pseudo_labels, student_model, criterion, input_shape):
    from ultralytics.utils.ops import xyxy2xywh

    if not isinstance(student_outputs, dict) or "one2one" not in student_outputs or "one2many" not in student_outputs:
        raise ValueError("Student outputs must contain one2one and one2many predictions.")

    device = student_outputs["one2one"]["boxes"].device
    width = int(input_shape[3])
    height = int(input_shape[2])
    norm = torch.tensor([width, height, width, height], device=device, dtype=torch.float32)

    all_batch_idx = []
    all_classes = []
    all_boxes = []

    for img_idx, labels in enumerate(pseudo_labels):
        if labels.numel() == 0:
            continue
        all_batch_idx.append(torch.full((labels.shape[0],), img_idx, device=device, dtype=torch.long))
        all_classes.append(labels[:, 5])
        all_boxes.append(xyxy2xywh(labels[:, :4]) / norm)

    if not all_boxes:
        zero = sum((p.sum() * 0.0) for p in student_model.parameters())
        loss_dict = {"box_loss": zero, "cls_loss": zero, "dfl_loss": zero}
        return loss_dict, zero

    batch = {
        "batch_idx": torch.cat(all_batch_idx, dim=0),
        "cls": torch.cat(all_classes, dim=0),
        "bboxes": torch.cat(all_boxes, dim=0),
    }
    total_loss, loss_items = criterion(student_outputs, batch)
    total_loss = total_loss.sum()
    return {"box_loss": loss_items[0], "cls_loss": loss_items[1], "dfl_loss": loss_items[2]}, total_loss


def update_teacher_ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


class DetectInputFeatureHook:
    """Capture PAN feature maps passed into the Detect head."""

    def __init__(self, yolo_model: nn.Module):
        self.latest: Optional[list[torch.Tensor]] = None
        self.detect_module = self._find_detect_module(yolo_model)
        self.handle = self.detect_module.register_forward_pre_hook(self._hook_fn)

    @staticmethod
    def _is_feature_list(x) -> bool:
        return isinstance(x, (list, tuple)) and len(x) >= 3 and all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in x[:3])

    def _hook_fn(self, _module, inputs) -> None:
        if len(inputs) == 1 and self._is_feature_list(inputs[0]):
            self.latest = list(inputs[0][:3])
        elif self._is_feature_list(inputs):
            self.latest = list(inputs[:3])
        else:
            self.latest = None

    @staticmethod
    def _find_detect_module(yolo_model: nn.Module) -> nn.Module:
        modules = getattr(yolo_model, "model", None)
        if modules is not None and len(modules) > 0 and "detect" in modules[-1].__class__.__name__.lower():
            return modules[-1]
        for module in reversed(list(yolo_model.modules())):
            if "detect" in module.__class__.__name__.lower():
                return module
        raise RuntimeError("Could not locate Detect head for MARD feature hook.")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def variance_loss(tokens: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    std = torch.sqrt(tokens.var(dim=0, unbiased=False) + eps)
    return torch.relu(gamma - std).mean()


def covariance_loss(tokens: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    n, c = tokens.shape
    z = tokens - tokens.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + eps)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / (c * (c - 1) + 1e-6)


def assign_boxes_to_levels(boxes: torch.Tensor, stride3: float, stride4: float, eta: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    sizes = torch.sqrt((boxes[:, 2] - boxes[:, 0]).clamp(min=1.0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0))
    levels = torch.empty_like(sizes, dtype=torch.long)
    levels[sizes <= eta * stride3] = 0
    levels[(sizes > eta * stride3) & (sizes <= eta * stride4)] = 1
    levels[sizes > eta * stride4] = 2
    return levels


def feature_rect(
    box: torch.Tensor,
    h_f: int,
    w_f: int,
    h_pad: int,
    w_pad: int,
    h_valid_f: int,
    w_valid_f: int,
) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box.float()
    x1f = int(torch.floor(x1 * w_f / max(w_pad, 1)).item())
    y1f = int(torch.floor(y1 * h_f / max(h_pad, 1)).item())
    x2f = int(torch.ceil(x2 * w_f / max(w_pad, 1)).item()) - 1
    y2f = int(torch.ceil(y2 * h_f / max(h_pad, 1)).item()) - 1

    x1f = max(0, min(x1f, w_f - 1, w_valid_f - 1))
    x2f = max(0, min(x2f, w_f - 1, w_valid_f - 1))
    y1f = max(0, min(y1f, h_f - 1, h_valid_f - 1))
    y2f = max(0, min(y2f, h_f - 1, h_valid_f - 1))
    if x2f < x1f or y2f < y1f:
        return None
    return x1f, y1f, x2f, y2f


def sample_level_tokens_for_image(
    fmap: torch.Tensor,
    boxes: torch.Tensor,
    levels: torch.Tensor,
    target_level: int,
    h_pad: int,
    w_pad: int,
    h_valid: int,
    w_valid: int,
    fg_points: int,
    bg_points: int,
) -> Optional[torch.Tensor]:
    device = fmap.device
    _, h_f, w_f = fmap.shape
    h_valid_f = max(1, min(int(math.ceil(h_valid * h_f / max(h_pad, 1))), h_f))
    w_valid_f = max(1, min(int(math.ceil(w_valid * w_f / max(w_pad, 1))), w_f))

    fg_tokens = []
    fg_mask = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
    level_mask = levels == target_level

    for box in boxes[level_mask]:
        rect = feature_rect(box, h_f, w_f, h_pad, w_pad, h_valid_f, w_valid_f)
        if rect is None:
            continue
        x1f, y1f, x2f, y2f = rect
        xs = torch.randint(x1f, x2f + 1, (fg_points,), device=device)
        ys = torch.randint(y1f, y2f + 1, (fg_points,), device=device)
        fg_tokens.append(fmap[:, ys, xs].T)
        fg_mask[y1f : y2f + 1, x1f : x2f + 1] = True

    valid_mask = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
    valid_mask[:h_valid_f, :w_valid_f] = True
    bg_coords = (valid_mask & (~fg_mask)).nonzero(as_tuple=False)
    if bg_coords.numel():
        idx = torch.randint(0, bg_coords.shape[0], (min(bg_points, bg_coords.shape[0]),), device=device)
        bg_sel = bg_coords[idx]
        bg_tokens = fmap[:, bg_sel[:, 0], bg_sel[:, 1]].T
    else:
        ys = torch.randint(0, h_valid_f, (bg_points,), device=device)
        xs = torch.randint(0, w_valid_f, (bg_points,), device=device)
        bg_tokens = fmap[:, ys, xs].T

    tokens = fg_tokens + [bg_tokens]
    return torch.cat(tokens, dim=0) if tokens else None


def compute_mard_loss(
    feats: list[torch.Tensor],
    pseudo_labels: list[torch.Tensor],
    strong_infos: list[dict],
    h_pad: int,
    w_pad: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = feats[0].new_zeros(())
    stats: dict[str, float] = {}
    stride3 = float(w_pad) / float(feats[0].shape[3])
    stride4 = float(w_pad) / float(feats[1].shape[3])

    for level_idx, fmap in enumerate(feats[:3]):
        tokens_all = []
        for b in range(fmap.shape[0]):
            labels = pseudo_labels[b]
            if labels.numel() == 0:
                continue

            boxes = labels[:, :4]
            confs = labels[:, 4]
            keep = confs >= MARD_BOX_CONF_THRESHOLD
            if keep.sum() == 0:
                continue
            boxes, confs = boxes[keep], confs[keep]
            if boxes.shape[0] > args.mard_topk_boxes:
                order = torch.argsort(confs, descending=True)[: args.mard_topk_boxes]
                boxes, confs = boxes[order], confs[order]

            levels = assign_boxes_to_levels(boxes, stride3=stride3, stride4=stride4, eta=args.mard_eta)
            h_valid, w_valid = strong_infos[b]["final_size"]
            tokens = sample_level_tokens_for_image(
                fmap=fmap[b],
                boxes=boxes,
                levels=levels,
                target_level=level_idx,
                h_pad=h_pad,
                w_pad=w_pad,
                h_valid=h_valid,
                w_valid=w_valid,
                fg_points=args.mard_fg_points,
                bg_points=args.mard_bg_points,
            )
            if tokens is not None:
                tokens_all.append(tokens)

        if tokens_all:
            z = torch.cat(tokens_all, dim=0)
            var = variance_loss(z, gamma=args.mard_gamma)
            cov = covariance_loss(z)
        else:
            var = fmap.new_zeros(())
            cov = fmap.new_zeros(())

        level_loss = args.mard_alpha * var + args.mard_beta * cov
        total = total + level_loss
        stats[f"p{level_idx + 3}_var"] = float(var.detach().item())
        stats[f"p{level_idx + 3}_cov"] = float(cov.detach().item())

    stats["mard"] = float(total.detach().item())
    return total, stats


def mard_weight(args: argparse.Namespace, global_step: int, steps_per_epoch: int, avg_conf: float) -> float:
    warmup_steps = max(1, int(args.mard_warmup_epochs * steps_per_epoch))
    ramp = min(1.0, float(global_step) / float(warmup_steps))
    gate = (avg_conf - args.mard_gate_threshold) / max(1.0 - args.mard_gate_threshold, 1e-6)
    gate = float(np.clip(gate, 0.0, 1.0))
    weight = args.mard_lambda0 * ramp * gate
    return min(weight, args.mard_lambda_max) if args.mard_lambda_max > 0 else weight


def average_pseudo_confidence(pseudo_labels: list[torch.Tensor]) -> float:
    confs = [labels[:, 4] for labels in pseudo_labels if labels.numel()]
    if not confs:
        return 0.0
    return float(torch.cat(confs).mean().item())


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images_from_yaml(args.data)
    assert imgs, f"No target training images found in {args.data}"
    print(
        f"[Stage 2] RT-SFOD YOLO26 | device={device} images={len(imgs)} imgsz={args.imgsz} "
        f"batch={args.batch} epochs={args.epochs}",
        flush=True,
    )
    print(
        f"[Stage 2] DHF: tau_o2o={args.tau_o2o} tau_o2m={args.tau_o2m} "
        f"tau_no={args.tau_no} tau_dup={args.tau_dup}",
        flush=True,
    )
    print(
        f"[Stage 2] MARD: lambda0={args.mard_lambda0} gamma={args.mard_gamma} "
        f"alpha={args.mard_alpha} beta={args.mard_beta} warmup={args.mard_warmup_epochs}ep",
        flush=True,
    )

    dataset = TargetTeacherStudentDataset(imgs, img_size=args.imgsz)
    generator = None
    if args.seed is not None and args.seed >= 0:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.workers > 0,
    )

    teacher_model, student_model, student_wrapper = setup_teacher_student(args.stage1_model, device)
    ensure_detection_loss_args(student_model, args.epochs)
    student_model.criterion = student_model.init_criterion()
    criterion = student_model.criterion

    optimizer = optim.SGD(
        student_model.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    hook: Optional[DetectInputFeatureHook] = DetectInputFeatureHook(student_model)

    global_step = 0
    try:
        for epoch in range(args.epochs):
            student_model.train()
            epoch_start = time.time()
            epoch_batches = 0
            skipped_batches = 0
            pseudo_boxes = 0
            sums = {"loss": 0.0, "det": 0.0, "mard": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}

            print(f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] start", flush=True)
            for batch_i, (weak_imgs, strong_imgs, _paths, weak_infos, strong_infos) in enumerate(dataloader, start=1):
                weak_imgs = weak_imgs.to(device, non_blocking=True)
                strong_imgs = strong_imgs.to(device, non_blocking=True)

                pseudo_weak = generate_pseudo_labels(
                    teacher_model,
                    weak_imgs,
                    tau_o2o=args.tau_o2o,
                    tau_o2m=args.tau_o2m,
                    tau_no=args.tau_no,
                    tau_dup=args.tau_dup,
                )
                pseudo_strong = map_pseudo_labels_to_strong(pseudo_weak, weak_infos, strong_infos)
                valid = [labels.numel() > 0 for labels in pseudo_strong]

                if not any(valid):
                    skipped_batches += 1
                    global_step += 1
                    if args.print_freq > 0 and (batch_i == 1 or batch_i % args.print_freq == 0):
                        print(
                            f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] "
                            f"batch {batch_i:04d}/{len(dataloader):04d} skipped=no_pseudo_labels",
                            flush=True,
                        )
                    continue

                strong_valid = strong_imgs[valid]
                labels_valid = [labels for labels, keep in zip(pseudo_strong, valid) if keep]
                infos_valid = [info for info, keep in zip(strong_infos, valid) if keep]
                avg_conf = average_pseudo_confidence(labels_valid)

                if hook is not None:
                    hook.latest = None
                student_outputs = student_model(strong_valid)
                feats = hook.latest if hook is not None else None
                loss_dict, det_loss = compute_student_loss(
                    student_outputs=student_outputs,
                    pseudo_labels=labels_valid,
                    student_model=student_model,
                    criterion=criterion,
                    input_shape=strong_valid.shape,
                )

                reg_loss = det_loss.new_zeros(())
                lambda_mard = 0.0
                if feats is not None and global_step % MARD_INTERVAL == 0:
                    reg_loss, _ = compute_mard_loss(
                        feats=feats,
                        pseudo_labels=labels_valid,
                        strong_infos=infos_valid,
                        h_pad=int(strong_valid.shape[2]),
                        w_pad=int(strong_valid.shape[3]),
                        args=args,
                    )
                    lambda_mard = mard_weight(args, global_step, len(dataloader), avg_conf)

                total_loss = det_loss + lambda_mard * reg_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                optimizer.step()

                global_step += 1
                epoch_batches += 1
                batch_boxes = sum(labels.shape[0] for labels in labels_valid)
                pseudo_boxes += batch_boxes

                values = {
                    "loss": scalarize(total_loss),
                    "det": scalarize(det_loss),
                    "mard": scalarize(reg_loss),
                    "box": scalarize(loss_dict["box_loss"]),
                    "cls": scalarize(loss_dict["cls_loss"]),
                    "dfl": scalarize(loss_dict["dfl_loss"]),
                }
                for key, value in values.items():
                    sums[key] += value

                if args.print_freq > 0 and (
                    batch_i == 1 or batch_i == len(dataloader) or batch_i % args.print_freq == 0
                ):
                    print(
                        f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] "
                        f"batch {batch_i:04d}/{len(dataloader):04d} "
                        f"loss={values['loss']:.4f} det={values['det']:.4f} "
                        f"box={values['box']:.4f} cls={values['cls']:.4f} dfl={values['dfl']:.4f} "
                        f"mard={values['mard']:.4f} lambda={lambda_mard:.4f} "
                        f"pseudo_boxes={batch_boxes} avg_conf={avg_conf:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6g}",
                        flush=True,
                    )

            scheduler.step()
            if hasattr(criterion, "update"):
                criterion.update()
            update_teacher_ema(teacher_model, student_model, momentum=args.ema_momentum)

            denom = max(epoch_batches, 1)
            print(
                f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] done "
                f"time={time.time() - epoch_start:.1f}s valid_batches={epoch_batches} "
                f"skipped={skipped_batches} pseudo_boxes={pseudo_boxes} "
                f"loss={sums['loss'] / denom:.4f} det={sums['det'] / denom:.4f} "
                f"box={sums['box'] / denom:.4f} cls={sums['cls'] / denom:.4f} "
                f"dfl={sums['dfl'] / denom:.4f} mard={sums['mard'] / denom:.4f}",
                flush=True,
            )

            save_this_epoch = (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs
            eval_this_epoch = args.eval and (epoch + 1) % args.val_interval == 0
            detach_hook_for_io = save_this_epoch or eval_this_epoch
            if detach_hook_for_io and hook is not None:
                hook.latest = None
                hook.close()
                hook = None

            if save_this_epoch:
                ckpt_path = checkpoint_dir / f"yolo26_stage2_rtsfod_epoch_{epoch + 1}.pt"
                student_wrapper.model = student_model
                student_wrapper.save(str(ckpt_path))
                print(f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] saved student: {ckpt_path}", flush=True)

            if eval_this_epoch:
                student_model.eval()
                val_wrapper = YOLO(args.stage1_model)
                val_wrapper.model = student_model
                metrics = val_wrapper.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    conf=0.001,
                    iou=0.6,
                    device=val_device_arg(device),
                    plots=False,
                    verbose=False,
                )
                map50 = getattr(metrics.box, "map50", 0.0)
                print(f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] eval student mAP50={map50:.4f}", flush=True)
                student_model.train()

            if detach_hook_for_io:
                hook = DetectInputFeatureHook(student_model)
    finally:
        if hook is not None:
            hook.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RT-SFOD Stage 2 for YOLO26 end-to-end detectors")
    parser.add_argument("--stage1_model", type=str, required=True, help="AdaBN-initialized Stage 1 checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Target-domain data YAML")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--tau_o2o", type=float, default=TAU_O2O)
    parser.add_argument("--tau_o2m", type=float, default=TAU_O2M)
    parser.add_argument("--tau_no", type=float, default=TAU_NO)
    parser.add_argument("--tau_dup", type=float, default=TAU_DUP)

    parser.add_argument("--mard_lambda0", type=float, default=MARD_LAMBDA0)
    parser.add_argument("--mard_lambda_max", type=float, default=MARD_LAMBDA_MAX)
    parser.add_argument("--mard_gamma", type=float, default=MARD_GAMMA)
    parser.add_argument("--mard_alpha", type=float, default=MARD_ALPHA)
    parser.add_argument("--mard_beta", type=float, default=MARD_BETA)
    parser.add_argument("--mard_warmup_epochs", type=float, default=MARD_WARMUP_EPOCHS)
    parser.add_argument("--mard_gate_threshold", type=float, default=MARD_GATE_THRESHOLD)
    parser.add_argument("--mard_topk_boxes", type=int, default=MARD_TOPK_BOXES)
    parser.add_argument("--mard_fg_points", type=int, default=MARD_FG_POINTS)
    parser.add_argument("--mard_bg_points", type=int, default=MARD_BG_POINTS)
    parser.add_argument("--mard_eta", type=float, default=MARD_ETA)
    parser.add_argument("--ema_momentum", type=float, default=EMA_MOMENTUM)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--eval", action="store_true", help="Evaluate the student at val_interval epochs")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29, help="Use -1 to disable seeding")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
