import argparse
import os
import random
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import yaml
import cv2
import numpy as np
import warnings

# Ensure this script uses the local editable Ultralytics checkout, not site-packages.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# -----------------------------
# Helpers
# -----------------------------
def seed_everything(seed: int, deterministic: bool = False):
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
        except Exception as e:
            warnings.warn(f"Could not enable torch deterministic algorithms: {e}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def scalarize(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().mean().item())
    return float(x)


def list_images_from_yaml(data_yaml: str) -> List[str]:
    with open(data_yaml, "r") as f:
        y = yaml.safe_load(f)
    root = Path(y.get("path", "."))
    train = y["train"]
    train_dir = root / train
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = []
    if (root / train).is_file():
        with open(root / train, "r") as tf:
            for line in tf:
                p = Path(line.strip())
                if p.suffix.lower() in exts and p.exists():
                    imgs.append(str(p))
    else:
        for dp, _, files in os.walk(train_dir):
            for fn in files:
                if Path(fn).suffix.lower() in exts:
                    imgs.append(str(Path(dp) / fn))
    imgs.sort()
    return imgs


# -----------------------------
# Dataset
# -----------------------------
class TeacherStudentDataset(data.Dataset):
    """
    Returns both weak and strong augmented versions of target images.
    Weak aug: stable teacher pseudo-labels
    Strong aug: student learns robustness

    MODIFICATION:
      - Elastic deformation augmentation REMOVED completely.
    """

    def __init__(self, img_paths, img_size=640, weak_aug=True, strong_aug=True):
        self.imgs = img_paths
        self.img_size = img_size
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.imgs)

    def resize_keep_aspect(self, img, target_long):
        h, w = img.shape[:2]
        scale = target_long / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        info = {"scale": scale, "resized_h": nh, "resized_w": nw, "flipped": False}
        return img_resized, info

    def apply_weak_augmentation(self, img, flip_decision):
        h_orig, w_orig = img.shape[:2]
        applied_transforms = []

        img, resize_info = self.resize_keep_aspect(img, self.img_size)
        h_after_resize, w_after_resize = img.shape[:2]
        applied_transforms.append(
            {
                "type": "resize",
                "scale": resize_info["scale"],
                "original_size": (h_orig, w_orig),
                "new_size": (h_after_resize, w_after_resize),
            }
        )

        if flip_decision:
            img = np.ascontiguousarray(np.fliplr(img))
            applied_transforms.append({"type": "horizontal_flip", "width": w_after_resize})

        transform_info = {
            **resize_info,
            "flipped": flip_decision,
            "final_size": (img.shape[0], img.shape[1]),
            "applied_transforms": applied_transforms,
        }
        return img, transform_info

    def apply_strong_augmentation(self, img, flip_decision):
        """
        Strong aug = resize + same flip + several photometric + some geometric (affine/perspective).
        MODIFICATION: Elastic deformation removed.
        """
        h_orig, w_orig = img.shape[:2]
        applied_transforms = []

        img, resize_info = self.resize_keep_aspect(img, self.img_size)
        h_after_resize, w_after_resize = img.shape[:2]
        applied_transforms.append(
            {
                "type": "resize",
                "scale": resize_info["scale"],
                "original_size": (h_orig, w_orig),
                "new_size": (h_after_resize, w_after_resize),
            }
        )

        if flip_decision:
            img = np.ascontiguousarray(np.fliplr(img))
            applied_transforms.append({"type": "horizontal_flip", "width": w_after_resize})

        # 3) Random scaling and translation
        scale_translate_applied = False
        scale_factor = 1.0
        translate_x = 0
        translate_y = 0
        scale_translate_matrix = None
        if random.random() < 0.5:
            h, w = img.shape[:2]
            scale = 0.9 + random.random() * 0.2
            tx = int((random.random() - 0.5) * 0.1 * w)
            ty = int((random.random() - 0.5) * 0.1 * h)

            M = np.float32([[scale, 0, tx], [0, scale, ty]])
            img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
            scale_translate_applied = True
            scale_factor = scale
            translate_x = tx
            translate_y = ty
            scale_translate_matrix = M
            applied_transforms.append(
                {"type": "scale_translate", "scale": scale, "translate_x": tx, "translate_y": ty, "matrix": M.copy()}
            )

        # 4) HSV jitter
        hsv_applied = False
        hsv_gains = None
        if random.random() < 0.8:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hgain = (random.random() * 0.3 - 0.15) * 180
            sgain = (random.random() * 0.4 - 0.2) * 255
            vgain = (random.random() * 0.4 - 0.2) * 255
            hsv[..., 0] = np.clip(hsv[..., 0] + hgain, 0, 179)
            hsv[..., 1] = np.clip(hsv[..., 1] + sgain, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + vgain, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            hsv_applied = True
            hsv_gains = (hgain, sgain, vgain)
            applied_transforms.append(
                {"type": "hsv_jitter", "hue_gain": hgain, "saturation_gain": sgain, "value_gain": vgain}
            )

        # 5) Brightness/contrast
        contrast_brightness_applied = False
        alpha_val = 1.0
        beta_val = 0
        if random.random() < 0.6:
            alpha = 0.8 + random.random() * 0.4
            beta = -20 + random.random() * 40
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            contrast_brightness_applied = True
            alpha_val = alpha
            beta_val = beta
            applied_transforms.append({"type": "contrast_brightness", "alpha": alpha, "beta": beta})

        # 6) Gaussian blur
        blur_applied = False
        blur_kernel = 0
        blur_sigma = 0
        if random.random() < 0.4:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            blur_applied = True
            blur_kernel = kernel_size
            blur_sigma = sigma
            applied_transforms.append({"type": "gaussian_blur", "kernel_size": kernel_size, "sigma": sigma})

        # 7) Noise
        noise_applied = False
        noise_params = None
        if random.random() < 0.3:
            noise_type = random.choice(["gaussian", "salt_pepper"])
            if noise_type == "gaussian":
                noise_std = random.uniform(5, 15)
                noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                noise_params = {"type": "gaussian", "std": noise_std}
            else:
                noise_prob = random.uniform(0.01, 0.05)
                salt_pepper = np.random.random(img.shape[:2])
                img[salt_pepper < noise_prob / 2] = 0
                img[salt_pepper > 1 - noise_prob / 2] = 255
                noise_params = {"type": "salt_pepper", "prob": noise_prob}
            noise_applied = True
            applied_transforms.append({"type": "noise", **noise_params})

        # 8) Channel shuffle
        channel_shuffle_applied = False
        channel_order = None
        if random.random() < 0.2:
            channels = [0, 1, 2]
            random.shuffle(channels)
            img = img[:, :, channels]
            channel_shuffle_applied = True
            channel_order = channels
            applied_transforms.append({"type": "channel_shuffle", "order": channels})

        # 9) Gamma
        gamma_applied = False
        gamma_value = 1.0
        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.3)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            img = cv2.LUT(img, table)
            gamma_applied = True
            gamma_value = gamma
            applied_transforms.append({"type": "gamma_correction", "gamma": gamma})

        # 10) Perspective
        perspective_applied = False
        perspective_matrix = None
        perspective_points = None
        if random.random() < 0.3:
            h, w = img.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            distortion = int(0.03 * min(w, h))
            pts2 = np.float32(
                [
                    [random.randint(-distortion, distortion), random.randint(-distortion, distortion)],
                    [w + random.randint(-distortion, distortion), random.randint(-distortion, distortion)],
                    [random.randint(-distortion, distortion), h + random.randint(-distortion, distortion)],
                    [w + random.randint(-distortion, distortion), h + random.randint(-distortion, distortion)],
                ]
            )
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (w, h), borderValue=(114, 114, 114))
            perspective_applied = True
            perspective_matrix = M
            perspective_points = (pts1, pts2)
            applied_transforms.append(
                {"type": "perspective", "matrix": M.copy(), "src_points": pts1.copy(), "dst_points": pts2.copy()}
            )

        transform_info = {
            **resize_info,
            "flipped": flip_decision,
            "final_size": (img.shape[0], img.shape[1]),
            "applied_transforms": applied_transforms,

            "scale_translate_applied": scale_translate_applied,
            "scale_factor": scale_factor,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "scale_translate_matrix": scale_translate_matrix,

            "perspective_applied": perspective_applied,
            "perspective_matrix": perspective_matrix,
            "perspective_points": perspective_points,

            "hsv_applied": hsv_applied,
            "hsv_gains": hsv_gains,

            "contrast_brightness_applied": contrast_brightness_applied,
            "alpha": alpha_val,
            "beta": beta_val,

            "blur_applied": blur_applied,
            "blur_kernel": blur_kernel,
            "blur_sigma": blur_sigma,

            "noise_applied": noise_applied,
            "noise_params": noise_params,

            "channel_shuffle_applied": channel_shuffle_applied,
            "channel_order": channel_order,

            "gamma_applied": gamma_applied,
            "gamma_value": gamma_value
        }
        return img, transform_info

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = cv2.imread(path)
        assert img is not None, f"Failed to read {path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        flip_decision = random.random() < 0.5

        if self.weak_aug:
            weak_img, weak_info = self.apply_weak_augmentation(img.copy(), flip_decision)
        else:
            weak_img, weak_info = self.resize_keep_aspect(img.copy(), self.img_size)
            weak_info["flipped"] = False

        if self.strong_aug:
            strong_img, strong_info = self.apply_strong_augmentation(img.copy(), flip_decision)
        else:
            strong_img, strong_info = self.resize_keep_aspect(img.copy(), self.img_size)
            strong_info["flipped"] = False

        weak_tensor = torch.from_numpy(weak_img).permute(2, 0, 1).float() / 255.0
        strong_tensor = torch.from_numpy(strong_img).permute(2, 0, 1).float() / 255.0
        return weak_tensor, strong_tensor, path, weak_info, strong_info


def collate_fn(batch):
    weak_imgs, strong_imgs, paths, weak_infos, strong_infos = zip(*batch)
    max_h_weak = max(im.shape[1] for im in weak_imgs)
    max_w_weak = max(im.shape[2] for im in weak_imgs)
    max_h_strong = max(im.shape[1] for im in strong_imgs)
    max_w_strong = max(im.shape[2] for im in strong_imgs)

    def pad_to_max(imgs, max_h, max_w):
        padded = []
        for im in imgs:
            c, h, w = im.shape
            canvas = torch.full((c, max_h, max_w), 114 / 255.0, device=im.device)
            canvas[:, :h, :w] = im
            padded.append(canvas)
        return torch.stack(padded, dim=0)

    weak_batch = pad_to_max(weak_imgs, max_h_weak, max_w_weak)
    strong_batch = pad_to_max(strong_imgs, max_h_strong, max_w_strong)
    return weak_batch, strong_batch, list(paths), list(weak_infos), list(strong_infos)


# -----------------------------
# Box transform weak->strong
# -----------------------------
def transform_boxes_weak_to_strong(boxes, weak_info, strong_info):
    """
    MODIFICATION:
      - Elastic deformation removed, so no elastic mapping branch.
    """
    if len(boxes) == 0:
        return boxes, torch.ones(0, dtype=torch.bool, device=boxes.device)

    device = boxes.device
    N = len(boxes)
    boxes_np = boxes.detach().cpu().numpy()

    h_strong, w_strong = strong_info["final_size"]
    assert weak_info["flipped"] == strong_info["flipped"], "Weak and strong flips must match."

    corners = np.zeros((N, 4, 2), dtype=np.float32)
    corners[:, 0, :] = boxes_np[:, [0, 1]]
    corners[:, 1, :] = boxes_np[:, [2, 1]]
    corners[:, 2, :] = boxes_np[:, [2, 3]]
    corners[:, 3, :] = boxes_np[:, [0, 3]]

    if strong_info.get("scale_translate_applied", False):
        M_scale = strong_info["scale_translate_matrix"]
        for i in range(N):
            for j in range(4):
                pt = np.array([corners[i, j, 0], corners[i, j, 1], 1.0])
                corners[i, j] = M_scale @ pt

    if strong_info.get("perspective_applied", False):
        M_persp = strong_info["perspective_matrix"]
        for i in range(N):
            for j in range(4):
                pt = np.array([corners[i, j, 0], corners[i, j, 1], 1.0])
                transformed = M_persp @ pt
                corners[i, j] = transformed[:2] / transformed[2]

    transformed_boxes = np.zeros((N, 4), dtype=np.float32)
    transformed_boxes[:, 0] = corners[:, :, 0].min(axis=1)
    transformed_boxes[:, 1] = corners[:, :, 1].min(axis=1)
    transformed_boxes[:, 2] = corners[:, :, 0].max(axis=1)
    transformed_boxes[:, 3] = corners[:, :, 1].max(axis=1)

    transformed_boxes[:, [0, 2]] = np.clip(transformed_boxes[:, [0, 2]], 0, w_strong)
    transformed_boxes[:, [1, 3]] = np.clip(transformed_boxes[:, [1, 3]], 0, h_strong)

    widths = transformed_boxes[:, 2] - transformed_boxes[:, 0]
    heights = transformed_boxes[:, 3] - transformed_boxes[:, 1]
    min_box_size = 2
    valid_mask = (widths >= min_box_size) & (heights >= min_box_size)

    transformed_boxes = torch.from_numpy(transformed_boxes).to(device)
    valid_mask = torch.from_numpy(valid_mask).to(device)
    return transformed_boxes, valid_mask


# -----------------------------
# Teacher/Student setup
# -----------------------------
def ensure_yolo26_end2end_model(model, label: str):
    """Validate that the checkpoint is a YOLO26-style end-to-end detection model."""
    head = getattr(model, "model", [None])[-1]
    if not getattr(model, "end2end", False):
        raise ValueError(
            f"{label} is not an end-to-end model. This Stage 2 script is tailored for YOLO26 detection, "
            "whose Detect head must expose one2one/one2many branches."
        )
    if head is None or "detect" not in head.__class__.__name__.lower():
        raise ValueError(f"{label} final module is not a Detect head: {type(head)}")
    if not hasattr(head, "one2one") or not hasattr(head, "one2many"):
        raise ValueError(f"{label} Detect head does not expose one2one/one2many branches.")


def ensure_detection_loss_args(model, args):
    """Make YOLO26's E2E loss usable when running outside the Ultralytics Trainer."""
    from argparse import Namespace

    existing = getattr(model, "args", None)
    if existing is None:
        existing = Namespace()
    elif isinstance(existing, dict):
        existing = Namespace(**existing)

    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "epochs": int(getattr(args, "epochs", 1) or 1),
    }
    for key, value in defaults.items():
        if not hasattr(existing, key) or getattr(existing, key) is None:
            setattr(existing, key, value)
    model.args = existing


def setup_teacher_student(stage1_checkpoint, device, conf_threshold=0.5, imgsz=640):
    teacher_wrapper = YOLO(stage1_checkpoint)
    teacher_model = teacher_wrapper.model.to(device)
    ensure_yolo26_end2end_model(teacher_model, "Teacher")
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_wrapper = YOLO(stage1_checkpoint)
    student_model = student_wrapper.model.to(device)
    ensure_yolo26_end2end_model(student_model, "Student")
    for param in student_model.parameters():
        param.requires_grad = True

    return teacher_model, student_model, teacher_wrapper, student_wrapper


# ----------------------------------------------------------------------
# Dual-Head Pseudo-label Fusion
# ----------------------------------------------------------------------
def fuse_dual_head_pseudo_labels(
    boxes_one2one,
    boxes_one2many,
    device,
    conf_thr_one2one=0.5,
    conf_thr_one2many=0.5,
    iou_consistency=0.6,   # kept for backward compatibility; NOT used in new logic
    iou_extra=0.2,         # Paper default tau_no: max IoU allowed between O2M candidate and ANY O2O box
    iou_duplicate=0.7,     # Paper default tau_dup: NMS IoU among selected O2M extras
    reliability_gamma=0.5, # kept for backward compatibility; NOT used in new logic
    return_debug: bool = False,
):
    """
    Dual-head fusion defaults follow the paper's Table S.1:
      tau_o2o = 0.5, tau_o2m = 0.5, tau_no = 0.2, tau_dup = 0.7.

    1) Keep all O2O boxes with conf >= conf_thr_one2one.
    2) Consider O2M boxes with conf >= conf_thr_one2many.
    3) Add ONLY those O2M boxes whose max IoU w.r.t. ANY kept O2O box is <= iou_extra (class-agnostic overlap check).
    4) Apply NMS to the selected O2M extras only (to remove duplicates inside O2M).
    5) Concatenate O2O + extras and sort by confidence.

    Notes:
      - "iou_extra" is the paper's tau_no non-overlap threshold.
      - If no O2O boxes are kept, we fall back to using O2M boxes (after threshold + NMS), otherwise you'd get empty pseudo labels.
    """

    n_o2o_raw = int(len(boxes_one2one)) if boxes_one2one is not None else 0
    n_o2m_raw = int(len(boxes_one2many)) if boxes_one2many is not None else 0

    if boxes_one2one is None or len(boxes_one2one) == 0:
        boxes_o2o = torch.zeros((0, 6), device=device)
    else:
        boxes_o2o = boxes_one2one.to(device)

    if boxes_one2many is None or len(boxes_one2many) == 0:
        boxes_o2m = torch.zeros((0, 6), device=device)
    else:
        boxes_o2m = boxes_one2many.to(device)

    # 1) Thresholding
    if boxes_o2o.numel():
        boxes_o2o = boxes_o2o[boxes_o2o[:, 4] >= conf_thr_one2one]
    if boxes_o2m.numel():
        boxes_o2m = boxes_o2m[boxes_o2m[:, 4] >= conf_thr_one2many]

    n_o2o_kept = int(boxes_o2o.shape[0])
    n_o2m_kept = int(boxes_o2m.shape[0])

    # Debug stats (reuse old names for compatibility with your logging)
    agreement_mean_iou = 0.0  # here: mean max IoU of O2M->O2O
    support_rate = 0.0        # here: fraction of O2M that overlap O2O above iou_extra
    extra_ratio = 0.0         # here: fraction of O2M that are selected as extras (non-overlapping)

    # Early exit if nothing
    if not boxes_o2o.numel() and not boxes_o2m.numel():
        fused = torch.zeros((0, 6), device=device)
        if return_debug:
            dbg = {
                "n_o2o_raw": n_o2o_raw,
                "n_o2m_raw": n_o2m_raw,
                "n_o2o_kept": n_o2o_kept,
                "n_o2m_kept": n_o2m_kept,
                "agreement_mean_iou": agreement_mean_iou,
                "support_rate": support_rate,
                "extra_ratio": extra_ratio,
                "n_fused": 0,
            }
            return fused, dbg
        return fused

    # If there are NO O2O boxes, fall back to O2M (after NMS) so training doesn't stall.
    if not boxes_o2o.numel():
        extra_boxes = boxes_o2m
        if extra_boxes.numel():
            keep = torchvision.ops.nms(extra_boxes[:, :4], extra_boxes[:, 4], iou_duplicate)
            extra_boxes = extra_boxes[keep]
            order = torch.argsort(extra_boxes[:, 4], descending=True)
            fused = extra_boxes[order]
        else:
            fused = torch.zeros((0, 6), device=device)

        if return_debug:
            dbg = {
                "n_o2o_raw": n_o2o_raw,
                "n_o2m_raw": n_o2m_raw,
                "n_o2o_kept": n_o2o_kept,
                "n_o2m_kept": n_o2m_kept,
                "agreement_mean_iou": 0.0,
                "support_rate": 0.0,
                "extra_ratio": 1.0 if n_o2m_kept > 0 else 0.0,
                "n_fused": int(fused.shape[0]),
            }
            return fused, dbg
        return fused

    # If there ARE O2O boxes, only add NON-overlapping O2M
    extra_boxes = torch.zeros((0, 6), device=device)
    if boxes_o2m.numel():
        # class-agnostic overlap: IoU between O2M and O2O boxes only
        ious = box_iou(boxes_o2m[:, :4], boxes_o2o[:, :4])  # [n_o2m, n_o2o]
        max_iou, _ = ious.max(dim=1) if ious.numel() else (torch.zeros((boxes_o2m.shape[0],), device=device), None)

        if max_iou.numel():
            agreement_mean_iou = float(max_iou.mean().item())

        overlap_mask = max_iou > float(iou_extra)
        support_rate = float(overlap_mask.float().mean().item()) if overlap_mask.numel() else 0.0

        nonoverlap_mask = max_iou <= float(iou_extra)
        extra_boxes = boxes_o2m[nonoverlap_mask]

        extra_ratio = float(nonoverlap_mask.float().mean().item()) if nonoverlap_mask.numel() else 0.0

        # NMS among extras to reduce O2M duplicates
        if extra_boxes.numel():
            keep = torchvision.ops.nms(extra_boxes[:, :4], extra_boxes[:, 4], iou_duplicate)
            extra_boxes = extra_boxes[keep]

    # Final fuse = all O2O + selected extras
    if extra_boxes.numel():
        fused = torch.cat([boxes_o2o, extra_boxes], dim=0)
    else:
        fused = boxes_o2o

    # Sort by confidence desc
    if fused.numel():
        order = torch.argsort(fused[:, 4], descending=True)
        fused = fused[order]

    if return_debug:
        dbg = {
            "n_o2o_raw": n_o2o_raw,
            "n_o2m_raw": n_o2m_raw,
            "n_o2o_kept": n_o2o_kept,
            "n_o2m_kept": n_o2m_kept,
            "agreement_mean_iou": agreement_mean_iou,
            "support_rate": support_rate,
            "extra_ratio": extra_ratio,
            "n_fused": int(fused.shape[0]),
        }
        return fused, dbg

    return fused


@torch.no_grad()
def generate_pseudo_labels_yolo26(
    teacher_model,
    weak_imgs,
    weak_infos,
    paths,
    conf_threshold_one2one=0.5,
    conf_threshold_one2many=0.5,
    dual_iou_consistency=0.6,  # kept for backward compatibility; NOT used in new fusion
    dual_iou_extra=0.2,        # Paper default tau_no: non-overlap IoU threshold for adding O2M extras
    dual_iou_duplicate=0.7,
    reliability_gamma=0.5,     # kept for backward compatibility; NOT used in new fusion
    return_debug: bool = False,
):
    device = weak_imgs.device
    teacher_model.eval()
    batch_size = weak_imgs.shape[0]

    outputs = teacher_model(weak_imgs, augment=False, visualize=False)
    if not (isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict)):
        raise ValueError(
            "YOLO26 teacher must return (final_predictions, branch_predictions) in eval mode. "
            f"Got {type(outputs)} instead."
        )

    # YOLO26's Detect head is end-to-end. In eval mode it returns:
    #   final_predictions: postprocessed one2one boxes, [B, max_det, 6]
    #   branch_predictions: raw one2many/one2one dicts used by the loss path.
    final_one2one, branch_preds = outputs
    head = teacher_model.model[-1]
    raw_one2many = branch_preds["one2many"]
    one2many_decoded = head._inference(raw_one2many).permute(0, 2, 1)
    final_one2many = head.postprocess(one2many_decoded)

    batch_pseudo_labels = []
    debug_list = [] if return_debug else None

    for img_idx in range(batch_size):
        boxes_o = final_one2one[img_idx]
        boxes_m = final_one2many[img_idx]
        boxes_o = boxes_o[boxes_o[:, 4] > 0] if boxes_o.numel() else None
        boxes_m = boxes_m[boxes_m[:, 4] > 0] if boxes_m.numel() else None

        if return_debug:
            fused, dbg = fuse_dual_head_pseudo_labels(
                boxes_o,
                boxes_m,
                device=device,
                conf_thr_one2one=conf_threshold_one2one,
                conf_thr_one2many=conf_threshold_one2many,
                iou_consistency=dual_iou_consistency,
                iou_extra=dual_iou_extra,          # non-overlap threshold
                iou_duplicate=dual_iou_duplicate,
                reliability_gamma=reliability_gamma,
                return_debug=True,
            )
            dbg["image_idx"] = int(img_idx)
            dbg["path"] = paths[img_idx] if img_idx < len(paths) else ""
            debug_list.append(dbg)
        else:
            fused = fuse_dual_head_pseudo_labels(
                boxes_o,
                boxes_m,
                device=device,
                conf_thr_one2one=conf_threshold_one2one,
                conf_thr_one2many=conf_threshold_one2many,
                iou_consistency=dual_iou_consistency,
                iou_extra=dual_iou_extra,          # non-overlap threshold
                iou_duplicate=dual_iou_duplicate,
                reliability_gamma=reliability_gamma,
                return_debug=False,
            )

        if fused is None or fused.numel() == 0:
            fused = torch.zeros((0, 6), device=device)
        batch_pseudo_labels.append(fused)

    if return_debug:
        return batch_pseudo_labels, debug_list
    return batch_pseudo_labels


# -----------------------------
# Student detection loss (Ultralytics)
# -----------------------------
def compute_student_loss(student_outputs, pseudo_labels, student_model, criterion, weak_img_size, strong_img_size):
    from ultralytics.utils.ops import xyxy2xywh

    assert weak_img_size[2:] == strong_img_size[2:], "Weak and strong image sizes must match."
    height, width = strong_img_size[2], strong_img_size[3]

    if not isinstance(student_outputs, dict):
        raise ValueError(f"Expected dict, got {type(student_outputs)}")
    if "one2one" not in student_outputs or "one2many" not in student_outputs:
        raise ValueError("Student outputs must contain 'one2one' and 'one2many' keys")

    device = student_outputs["one2one"]["boxes"].device

    all_boxes = []
    all_classes = []
    all_conf = []
    all_batch_idx = []

    for img_idx, labels in enumerate(pseudo_labels):
        if len(labels) == 0:
            continue
        boxes_xyxy = labels[:, :4]
        confs = labels[:, 4]
        classes = labels[:, 5]
        boxes_xywh = xyxy2xywh(boxes_xyxy)
        boxes_xywh_norm = boxes_xywh / torch.tensor([width, height, width, height], device=device, dtype=boxes_xywh.dtype)

        all_boxes.append(boxes_xywh_norm)
        all_classes.append(classes)
        all_conf.append(confs)
        all_batch_idx.append(torch.full((len(labels),), img_idx, device=device, dtype=torch.long))

    if len(all_boxes) == 0:
        loss_dict = {
            "box_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "cls_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "dfl_loss": torch.tensor(0.0, device=device, requires_grad=True),
        }
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss_dict, total_loss

    batch_dict = {
        "batch_idx": torch.cat(all_batch_idx, dim=0),
        "cls": torch.cat(all_classes, dim=0),
        "bboxes": torch.cat(all_boxes, dim=0),
        "conf": torch.cat(all_conf, dim=0),
    }

    total_loss, loss_items = criterion(student_outputs, batch_dict)
    total_loss = total_loss.sum()

    loss_dict = {
        "box_loss": loss_items[0],
        "cls_loss": loss_items[1],
        "dfl_loss": loss_items[2],
    }

    # keep graph safe
    try:
        dummy = torch.tensor(0.0, device=device)
        for p in student_model.parameters():
            dummy = dummy + (p.sum() * 0.0)
        total_loss = total_loss + dummy
    except Exception:
        pass

    return loss_dict, total_loss


# -----------------------------
# EMA update
# -----------------------------
def update_teacher_ema(teacher, student, momentum=0.999):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1.0 - momentum) * student_param.data


# =============================================================================
# MARD (VAR + COV ONLY)  [unchanged]
# =============================================================================
class DetectInputFeatureHook:
    """
    Captures detect-head input feature maps (P3,P4,P5) during forward.
    """

    def __init__(self, yolo_model: nn.Module, detach: bool = False, verbose: bool = True):
        self.detach = detach
        self.latest: Optional[List[torch.Tensor]] = None
        self.handle = None
        self.detect_module = None
        self.detect_name = None

        self.detect_name, self.detect_module = self._find_detect_module(yolo_model)

        if self.detect_module is None:
            raise RuntimeError(
                "[MARD] Could not locate Detect head module to hook. "
                "Please inspect model.named_modules() and update _find_detect_module()."
            )

        self.handle = self.detect_module.register_forward_pre_hook(self._hook_fn)

    def _is_feat_list(self, x) -> bool:
        if not isinstance(x, (list, tuple)) or len(x) < 3:
            return False
        for t in x[:3]:
            if not (isinstance(t, torch.Tensor) and t.dim() == 4):
                return False
        return True

    def _hook_fn(self, _module, inputs):
        try:
            if len(inputs) == 1 and self._is_feat_list(inputs[0]):
                feats = inputs[0]
            elif self._is_feat_list(inputs):
                feats = inputs
            else:
                self.latest = None
                return

            self.latest = [t.detach() for t in feats[:3]] if self.detach else list(feats[:3])
        except Exception:
            self.latest = None

    def _find_detect_module(self, yolo_model: nn.Module):
        mm = getattr(yolo_model, "model", None)
        if mm is not None and hasattr(mm, "__len__") and hasattr(mm, "__getitem__") and len(mm) > 0:
            cand = mm[-1]
            if "detect" in cand.__class__.__name__.lower():
                return "model[-1]", cand

        detect_candidates = []
        for name, m in yolo_model.named_modules():
            if "detect" in m.__class__.__name__.lower():
                detect_candidates.append((name, m))
        if detect_candidates:
            return detect_candidates[-1][0], detect_candidates[-1][1]

        if mm is not None and hasattr(mm, "__len__") and hasattr(mm, "__getitem__") and len(mm) > 0:
            return "model[-1](fallback)", mm[-1]

        return None, None

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _vicreg_variance_loss(x: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if x is None or x.numel() == 0 or x.shape[0] < 2:
        return x.new_zeros(())
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return torch.relu(gamma - std).mean()


def _vicreg_covariance_loss(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x is None or x.numel() == 0 or x.shape[0] < 2:
        return x.new_zeros(())
    n, d = x.shape
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True) + eps)
    cov = (x.T @ x) / max(n - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return (off ** 2).sum() / (d * (d - 1) + 1e-6)


def _box_level_assign(
    boxes_xyxy: torch.Tensor,
    W_pad: int,
    Wf3: int,
    Wf4: int,
    level_cells: float = 12.0,
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0,), dtype=torch.long)

    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    s = torch.sqrt(w * h)

    stride3 = float(W_pad) / float(Wf3)
    stride4 = float(W_pad) / float(Wf4)

    t3 = level_cells * stride3
    t4 = level_cells * stride4

    lvl = torch.empty_like(s, dtype=torch.long)
    lvl[s <= t3] = 0
    lvl[(s > t3) & (s <= t4)] = 1
    lvl[s > t4] = 2
    return lvl


def _pixel_box_to_fmap_rect(
    box_xyxy: torch.Tensor,
    Hf: int,
    Wf: int,
    H_pad: int,
    W_pad: int,
    h_valid_f: int,
    w_valid_f: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box_xyxy.float()

    x1f = int(torch.floor(x1 * Wf / max(W_pad, 1)).item())
    y1f = int(torch.floor(y1 * Hf / max(H_pad, 1)).item())
    x2f = int(torch.ceil(x2 * Wf / max(W_pad, 1)).item()) - 1
    y2f = int(torch.ceil(y2 * Hf / max(H_pad, 1)).item()) - 1

    x1f = max(0, min(x1f, Wf - 1))
    y1f = max(0, min(y1f, Hf - 1))
    x2f = max(0, min(x2f, Wf - 1))
    y2f = max(0, min(y2f, Hf - 1))

    x2f = min(x2f, w_valid_f - 1)
    y2f = min(y2f, h_valid_f - 1)
    x1f = min(x1f, w_valid_f - 1)
    y1f = min(y1f, h_valid_f - 1)

    if x2f < x1f or y2f < y1f:
        return None
    return x1f, y1f, x2f, y2f


def _sample_fg_tokens_for_image_level(
    fmap_b: torch.Tensor,          # [C,Hf,Wf]
    boxes_xyxy: torch.Tensor,      # [N,4] pixel
    confs: torch.Tensor,           # [N]
    levels: torch.Tensor,          # [N] in {0,1,2}
    target_level: int,
    H_pad: int,
    W_pad: int,
    h_valid: int,
    w_valid: int,
    pts_per_box: int,
    topk: int,
    conf_thr: float,
) -> Optional[torch.Tensor]:
    device = fmap_b.device
    C, Hf, Wf = fmap_b.shape

    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return None

    keep = confs >= conf_thr
    if keep.sum() == 0:
        return None

    boxes_xyxy = boxes_xyxy[keep]
    confs = confs[keep]
    levels = levels[keep]

    if boxes_xyxy.shape[0] > topk:
        order = torch.argsort(confs, descending=True)[:topk]
        boxes_xyxy = boxes_xyxy[order]
        levels = levels[order]

    w_valid_f = int(np.ceil(float(w_valid) * Wf / max(W_pad, 1)))
    h_valid_f = int(np.ceil(float(h_valid) * Hf / max(H_pad, 1)))
    w_valid_f = max(1, min(w_valid_f, Wf))
    h_valid_f = max(1, min(h_valid_f, Hf))

    toks = []
    for i in range(boxes_xyxy.shape[0]):
        if int(levels[i].item()) != int(target_level):
            continue
        rect = _pixel_box_to_fmap_rect(
            boxes_xyxy[i], Hf=Hf, Wf=Wf, H_pad=H_pad, W_pad=W_pad,
            h_valid_f=h_valid_f, w_valid_f=w_valid_f,
        )
        if rect is None:
            continue
        x1f, y1f, x2f, y2f = rect
        xs = torch.randint(low=x1f, high=x2f + 1, size=(pts_per_box,), device=device)
        ys = torch.randint(low=y1f, high=y2f + 1, size=(pts_per_box,), device=device)
        tok = fmap_b[:, ys, xs].T
        toks.append(tok)

    if not toks:
        return None
    return torch.cat(toks, dim=0)


def _sample_bg_tokens_for_image_level(
    fmap_b: torch.Tensor,          # [C,Hf,Wf]
    boxes_xyxy: torch.Tensor,      # [N,4] pixel
    confs: torch.Tensor,           # [N]
    levels: torch.Tensor,          # [N]
    target_level: int,
    H_pad: int,
    W_pad: int,
    h_valid: int,
    w_valid: int,
    bg_pts: int,
    topk: int,
    conf_thr: float,
) -> Optional[torch.Tensor]:
    device = fmap_b.device
    C, Hf, Wf = fmap_b.shape

    w_valid_f = int(np.ceil(float(w_valid) * Wf / max(W_pad, 1)))
    h_valid_f = int(np.ceil(float(h_valid) * Hf / max(H_pad, 1)))
    w_valid_f = max(1, min(w_valid_f, Wf))
    h_valid_f = max(1, min(h_valid_f, Hf))

    valid_mask = torch.zeros((Hf, Wf), dtype=torch.bool, device=device)
    valid_mask[:h_valid_f, :w_valid_f] = True

    fg_mask = torch.zeros((Hf, Wf), dtype=torch.bool, device=device)

    if boxes_xyxy is not None and boxes_xyxy.numel() > 0:
        keep = confs >= conf_thr
        if keep.sum() > 0:
            boxes_xyxy_k = boxes_xyxy[keep]
            confs_k = confs[keep]
            levels_k = levels[keep]

            if boxes_xyxy_k.shape[0] > topk:
                order = torch.argsort(confs_k, descending=True)[:topk]
                boxes_xyxy_k = boxes_xyxy_k[order]
                levels_k = levels_k[order]

            for i in range(boxes_xyxy_k.shape[0]):
                if int(levels_k[i].item()) != int(target_level):
                    continue
                rect = _pixel_box_to_fmap_rect(
                    boxes_xyxy_k[i], Hf=Hf, Wf=Wf, H_pad=H_pad, W_pad=W_pad,
                    h_valid_f=h_valid_f, w_valid_f=w_valid_f,
                )
                if rect is None:
                    continue
                x1f, y1f, x2f, y2f = rect
                fg_mask[y1f:y2f + 1, x1f:x2f + 1] = True

    bg_mask = valid_mask & (~fg_mask)
    coords = bg_mask.nonzero(as_tuple=False)  # [K,2]

    if coords.numel() == 0:
        ys = torch.randint(low=0, high=h_valid_f, size=(bg_pts,), device=device)
        xs = torch.randint(low=0, high=w_valid_f, size=(bg_pts,), device=device)
        return fmap_b[:, ys, xs].T

    n = min(bg_pts, coords.shape[0])
    idx = torch.randint(low=0, high=coords.shape[0], size=(n,), device=device)
    sel = coords[idx]
    ys, xs = sel[:, 0], sel[:, 1]
    return fmap_b[:, ys, xs].T


def compute_mard_loss(
    feats: List[torch.Tensor],               # [P3,P4,P5], each [B,C,Hf,Wf]
    pseudo_labels: List[torch.Tensor],       # list of [Ni,6] (xyxy,conf,cls) in strong coords
    strong_infos: List[dict],                # list of dicts, includes final_size (h,w)
    H_pad: int,
    W_pad: int,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    MARD loss across FPN levels using ONLY:
      - var_loss (VICReg variance)
      - cov_loss (VICReg covariance)
    Returns (reg_loss_tensor, stats_dict).
    """
    device = feats[0].device
    total = torch.zeros((), device=device)

    lvl_names = ["p3", "p4", "p5"]
    active_levels = set([s.strip().lower() for s in args.mard_levels.split(",") if s.strip()])
    active_idx = [i for i, n in enumerate(lvl_names) if n in active_levels]

    # for level assignment thresholds
    Wf3 = feats[0].shape[3]
    Wf4 = feats[1].shape[3]

    stats: Dict[str, float] = {}

    for li in active_idx:
        fmap = feats[li]  # [B,C,Hf,Wf]
        B, C, Hf, Wf = fmap.shape

        fg_tokens_all = []
        bg_tokens_all = []

        for b in range(B):
            pl = pseudo_labels[b]
            if pl is None or pl.numel() == 0:
                continue

            boxes = pl[:, :4]
            confs = pl[:, 4]

            if args.mard_use_level_assign and boxes.numel():
                levels = _box_level_assign(
                    boxes, W_pad=W_pad, Wf3=Wf3, Wf4=Wf4, level_cells=args.mard_level_cells
                )
            else:
                levels = boxes.new_full((boxes.shape[0],), li, dtype=torch.long)

            h_valid, w_valid = strong_infos[b]["final_size"]

            fg_tok = _sample_fg_tokens_for_image_level(
                fmap[b], boxes, confs, levels, target_level=li,
                H_pad=H_pad, W_pad=W_pad,
                h_valid=h_valid, w_valid=w_valid,
                pts_per_box=args.mard_pts_per_box,
                topk=args.mard_topk_boxes,
                conf_thr=args.mard_conf_box_thr,
            )
            if fg_tok is not None:
                fg_tokens_all.append(fg_tok)

            bg_tok = _sample_bg_tokens_for_image_level(
                fmap[b], boxes, confs, levels, target_level=li,
                H_pad=H_pad, W_pad=W_pad,
                h_valid=h_valid, w_valid=w_valid,
                bg_pts=args.mard_bg_pts,
                topk=args.mard_topk_boxes,
                conf_thr=args.mard_conf_box_thr,
            )
            if bg_tok is not None:
                bg_tokens_all.append(bg_tok)

        fg_tokens = torch.cat(fg_tokens_all, dim=0) if fg_tokens_all else None
        bg_tokens = torch.cat(bg_tokens_all, dim=0) if bg_tokens_all else None

        def _subsample(x: Optional[torch.Tensor], max_n: int) -> Optional[torch.Tensor]:
            if x is None or x.numel() == 0:
                return x
            if x.shape[0] <= max_n:
                return x
            idx = torch.randperm(x.shape[0], device=x.device)[:max_n]
            return x[idx]

        fg_tokens = _subsample(fg_tokens, args.mard_max_tokens)
        bg_tokens = _subsample(bg_tokens, args.mard_max_tokens)

        # choose main tokens for var/cov
        if fg_tokens is not None and fg_tokens.shape[0] >= args.mard_min_tokens:
            main_tokens = fg_tokens
        elif (fg_tokens is not None and bg_tokens is not None
              and (fg_tokens.shape[0] + bg_tokens.shape[0] >= args.mard_min_tokens)):
            main_tokens = torch.cat([fg_tokens, bg_tokens], dim=0)
        elif bg_tokens is not None and bg_tokens.shape[0] >= args.mard_min_tokens:
            main_tokens = bg_tokens
        else:
            main_tokens = None

        var_loss = torch.zeros((), device=device)
        cov_loss = torch.zeros((), device=device)

        if main_tokens is not None and main_tokens.shape[0] >= 2:
            var_loss = _vicreg_variance_loss(main_tokens, gamma=args.mard_var_gamma)
            cov_loss = _vicreg_covariance_loss(main_tokens)

        level_loss = args.mard_var_weight * var_loss + args.mard_cov_weight * cov_loss
        total = total + level_loss

        stats[f"mard_{lvl_names[li]}_var"] = float(var_loss.detach().item())
        stats[f"mard_{lvl_names[li]}_cov"] = float(cov_loss.detach().item())

    stats["mard_total"] = float(total.detach().item())
    return total, stats


def compute_lambda_reg(
    args,
    global_step: int,
    steps_per_epoch: int,
    avg_conf: float,
) -> float:
    """
    lambda = lambda0 * warmup_ramp * conf_gate
    """
    if not args.use_mard:
        return 0.0

    warmup_steps = max(1, int(args.mard_warmup_epochs * steps_per_epoch))
    ramp = min(1.0, float(global_step) / float(warmup_steps))

    c0 = float(args.mard_conf_gate_thr)
    conf_gate = (avg_conf - c0) / max(1.0 - c0, 1e-6)
    conf_gate = float(np.clip(conf_gate, 0.0, 1.0))

    lam = float(args.mard_lambda0) * ramp * conf_gate
    if args.mard_lambda_max > 0:
        lam = min(lam, float(args.mard_lambda_max))
    return lam


# -----------------------------
# Main
# -----------------------------
def main(args):
    if args.device.isdigit():
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seed_everything(args.seed, deterministic=args.deterministic)

    conf_o2o = args.conf_thr_one2one if args.conf_thr_one2one is not None else args.conf_threshold
    conf_o2m = args.conf_thr_one2many if args.conf_thr_one2many is not None else args.conf_threshold

    print(f"[stage2] reading images from data yaml: {args.data}", flush=True)
    imgs = list_images_from_yaml(args.data)
    assert len(imgs) > 0, "No training images found in YAML"
    print(
        f"[stage2] device={device} imgs={len(imgs)} imgsz={args.imgsz} batch={args.batch} "
        f"epochs={args.epochs} use_mard={args.use_mard}",
        flush=True,
    )
    print(
        f"[stage2] DHF defaults: tau_o2o={conf_o2o:.3f} tau_o2m={conf_o2m:.3f} "
        f"tau_no={args.dual_iou_extra:.3f} tau_dup={args.dual_iou_duplicate:.3f}",
        flush=True,
    )
    if args.use_mard:
        print(
            f"[stage2] MARD defaults: lambda0={args.mard_lambda0:.4f} "
            f"lambda_max={args.mard_lambda_max:.4f} warmup_epochs={args.mard_warmup_epochs} "
            f"conf_gate={args.mard_conf_gate_thr:.3f}",
            flush=True,
        )

    dataset = TeacherStudentDataset(imgs, img_size=args.imgsz, weak_aug=True, strong_aug=True)

    g = None
    if args.seed is not None and args.seed >= 0:
        g = torch.Generator()
        g.manual_seed(args.seed)

    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.workers > 0),
    )

    print(f"[stage2] loading YOLO26 teacher/student from: {args.stage1_model}", flush=True)
    teacher_model, student_model, teacher_wrapper, student_wrapper = setup_teacher_student(
        args.stage1_model, device, conf_threshold=args.conf_threshold, imgsz=args.imgsz
    )
    print("[stage2] teacher/student loaded; initializing loss and optimizer", flush=True)
    ensure_detection_loss_args(student_model, args)
    student_model.criterion = student_model.init_criterion()
    criterion = student_model.criterion

    hook = DetectInputFeatureHook(student_model, detach=False) if args.use_mard else None

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            student_model.parameters(),
            lr=args.lr,
            momentum=0.937,
            weight_decay=0.0005,
            nesterov=True,
        )
    else:
        optimizer = optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=0.0005)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_student_map50 = 0.0
    best_teacher_map50 = 0.0
    global_step = 0
    one_epoch_time_sec = None

    try:
        for epoch in range(args.epochs):
            student_model.train()
            epoch_stats = {"total_batches": 0, "skipped_batches": 0}
            epoch_loss_sum = 0.0
            epoch_det_loss_sum = 0.0
            epoch_reg_loss_sum = 0.0
            epoch_box_loss_sum = 0.0
            epoch_cls_loss_sum = 0.0
            epoch_dfl_loss_sum = 0.0
            epoch_valid_batches = 0
            epoch_pseudo_boxes = 0
            epoch_start_time = time.time()
            print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] start", flush=True)

            for batch_i, (weak_imgs, strong_imgs, paths, weak_infos, strong_infos) in enumerate(dataloader, start=1):
                weak_imgs = weak_imgs.to(device)
                strong_imgs = strong_imgs.to(device)

                pseudo_labels = generate_pseudo_labels_yolo26(
                    teacher_model,
                    weak_imgs,
                    weak_infos,
                    paths,
                    conf_threshold_one2one=conf_o2o,
                    conf_threshold_one2many=conf_o2m,
                    dual_iou_consistency=args.dual_iou_consistency,
                    dual_iou_extra=args.dual_iou_extra,
                    dual_iou_duplicate=args.dual_iou_duplicate,
                    reliability_gamma=args.dual_reliability_gamma,
                    return_debug=False,
                )

                transformed_pseudo_labels = []
                for pl, weak_info, strong_info in zip(pseudo_labels, weak_infos, strong_infos):
                    if len(pl) > 0:
                        boxes = pl[:, :4]
                        confs = pl[:, 4]
                        classes = pl[:, 5]
                        transformed_boxes, valid_box_mask = transform_boxes_weak_to_strong(boxes, weak_info, strong_info)
                        if valid_box_mask.any():
                            transformed_boxes = transformed_boxes[valid_box_mask]
                            confs = confs[valid_box_mask]
                            classes = classes[valid_box_mask]
                            transformed_pl = torch.cat(
                                [transformed_boxes, confs.unsqueeze(1), classes.unsqueeze(1)], dim=1
                            )
                            transformed_pseudo_labels.append(transformed_pl)
                        else:
                            transformed_pseudo_labels.append(torch.zeros((0, 6), device=pl.device))
                    else:
                        transformed_pseudo_labels.append(pl)

                valid_mask = [len(pl) > 0 for pl in transformed_pseudo_labels]
                epoch_stats["total_batches"] += 1

                if not any(valid_mask):
                    epoch_stats["skipped_batches"] += 1
                    global_step += 1
                    if args.print_freq > 0 and (batch_i % args.print_freq == 0 or batch_i == 1):
                        print(
                            f"[epoch {epoch + 1:03d}/{args.epochs:03d}] "
                            f"batch {batch_i:04d}/{len(dataloader):04d} skipped=no_pseudo_labels "
                            f"skipped={epoch_stats['skipped_batches']}",
                            flush=True,
                        )
                    continue

                strong_imgs_valid = strong_imgs[valid_mask]
                pseudo_labels_valid = [pl for pl, v in zip(transformed_pseudo_labels, valid_mask) if v]
                strong_infos_valid = [si for si, v in zip(strong_infos, valid_mask) if v]
                batch_pseudo_boxes = int(sum(len(pl) for pl in pseudo_labels_valid))

                avg_conf = float(
                    np.mean([pl[:, 4].mean().item() for pl in pseudo_labels_valid if len(pl) > 0])
                ) if pseudo_labels_valid else 0.0

                if hook is not None:
                    hook.latest = None
                student_outputs = student_model(strong_imgs_valid)
                feats = hook.latest if hook is not None else None

                loss_dict, det_loss = compute_student_loss(
                    student_outputs,
                    pseudo_labels_valid,
                    student_model=student_model,
                    criterion=criterion,
                    weak_img_size=weak_imgs.shape,
                    strong_img_size=strong_imgs_valid.shape,
                )

                reg_loss = det_loss.new_zeros(())
                lambda_reg = 0.0

                if args.use_mard and feats is not None and (global_step % max(args.mard_interval, 1) == 0):
                    H_pad = int(strong_imgs_valid.shape[2])
                    W_pad = int(strong_imgs_valid.shape[3])
                    reg_loss, _ = compute_mard_loss(
                        feats=feats,
                        pseudo_labels=pseudo_labels_valid,
                        strong_infos=strong_infos_valid,
                        H_pad=H_pad,
                        W_pad=W_pad,
                        args=args,
                    )
                    lambda_reg = compute_lambda_reg(
                        args=args,
                        global_step=global_step,
                        steps_per_epoch=len(dataloader),
                        avg_conf=avg_conf,
                    )

                total_loss = det_loss + (float(lambda_reg) * reg_loss)

                optimizer.zero_grad()
                total_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                optimizer.step()
                global_step += 1
                epoch_valid_batches += 1
                epoch_pseudo_boxes += batch_pseudo_boxes

                total_loss_f = scalarize(total_loss)
                det_loss_f = scalarize(det_loss)
                reg_loss_f = scalarize(reg_loss)
                box_loss_f = scalarize(loss_dict.get("box_loss", 0.0))
                cls_loss_f = scalarize(loss_dict.get("cls_loss", 0.0))
                dfl_loss_f = scalarize(loss_dict.get("dfl_loss", 0.0))

                epoch_loss_sum += total_loss_f
                epoch_det_loss_sum += det_loss_f
                epoch_reg_loss_sum += reg_loss_f
                epoch_box_loss_sum += box_loss_f
                epoch_cls_loss_sum += cls_loss_f
                epoch_dfl_loss_sum += dfl_loss_f

                if args.print_freq > 0 and (batch_i % args.print_freq == 0 or batch_i == 1 or batch_i == len(dataloader)):
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(
                        f"[epoch {epoch + 1:03d}/{args.epochs:03d}] "
                        f"batch {batch_i:04d}/{len(dataloader):04d} "
                        f"loss={total_loss_f:.4f} det={det_loss_f:.4f} "
                        f"box={box_loss_f:.4f} cls={cls_loss_f:.4f} dfl={dfl_loss_f:.4f} "
                        f"mard={reg_loss_f:.4f} lambda={float(lambda_reg):.4f} "
                        f"pseudo_boxes={batch_pseudo_boxes} avg_conf={avg_conf:.4f} "
                        f"lr={lr_now:.6g} skipped={epoch_stats['skipped_batches']}",
                        flush=True,
                    )

            scheduler.step()
            if hasattr(criterion, "update"):
                criterion.update()

            if args.use_ema:
                update_teacher_ema(teacher_model, student_model, momentum=args.ema_momentum)

            epoch_time = time.time() - epoch_start_time
            if args.epochs == 1:
                one_epoch_time_sec = float(epoch_time)
            denom = max(epoch_valid_batches, 1)
            print(
                f"[epoch {epoch + 1:03d}/{args.epochs:03d}] done "
                f"time={epoch_time:.1f}s valid_batches={epoch_valid_batches}/{epoch_stats['total_batches']} "
                f"skipped={epoch_stats['skipped_batches']} pseudo_boxes={epoch_pseudo_boxes} "
                f"loss={epoch_loss_sum / denom:.4f} det={epoch_det_loss_sum / denom:.4f} "
                f"box={epoch_box_loss_sum / denom:.4f} cls={epoch_cls_loss_sum / denom:.4f} "
                f"dfl={epoch_dfl_loss_sum / denom:.4f} mard={epoch_reg_loss_sum / denom:.4f}",
                flush=True,
            )

            save_this_epoch = (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs
            eval_this_epoch = args.eval and (epoch + 1) % args.val_interval == 0

            hook_detached_for_io = False
            if hook is not None and (save_this_epoch or eval_this_epoch):
                # Ultralytics save() deep-copies the model. A registered MARD hook can reference
                # graph-attached feature tensors, so detach it before checkpoint/eval I/O.
                hook.latest = None
                hook.close()
                hook = None
                hook_detached_for_io = True

            if save_this_epoch:
                checkpoint_path = checkpoint_dir / f"yolo26_stage2_student_epoch_{epoch+1}.pt"
                student_wrapper.model = student_model
                student_wrapper.save(str(checkpoint_path))
                print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] saved student: {checkpoint_path}", flush=True)

                if args.use_ema:
                    teacher_checkpoint_path = checkpoint_dir / f"yolo26_stage2_teacher_ema_epoch_{epoch+1}.pt"
                    teacher_wrapper.model = teacher_model
                    teacher_wrapper.save(str(teacher_checkpoint_path))
                    print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] saved teacher EMA: {teacher_checkpoint_path}", flush=True)

            if eval_this_epoch:
                student_model.eval()
                val_wrapper = YOLO(args.stage1_model)
                val_wrapper.model = student_model

                try:
                    metrics = val_wrapper.val(
                        data=args.data,
                        imgsz=args.imgsz,
                        batch=args.batch,
                        conf=0.001,
                        iou=0.6,
                        device=device,
                        plots=False,
                        verbose=False,
                    )
                    map50 = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
                    print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] eval student mAP50={map50:.4f}", flush=True)
                    if map50 > best_student_map50:
                        best_student_map50 = map50
                        best_model_path = checkpoint_dir / "best_student.pt"
                        torch.save({"epoch": epoch + 1, "model": student_model.state_dict(), "map50": map50}, best_model_path)
                        print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] new best student: {best_model_path}", flush=True)

                    if args.use_ema:
                        val_wrapper.model = teacher_model
                        teacher_metrics = val_wrapper.val(
                            data=args.data,
                            imgsz=args.imgsz,
                            batch=args.batch,
                            conf=0.001,
                            iou=0.6,
                            device=device,
                            plots=False,
                            verbose=False,
                        )
                        teacher_map50 = teacher_metrics.results_dict.get("metrics/mAP50(B)", 0.0)
                        print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] eval teacher EMA mAP50={teacher_map50:.4f}", flush=True)
                        if teacher_map50 > best_teacher_map50:
                            best_teacher_map50 = teacher_map50
                            best_teacher_path = checkpoint_dir / "best_teacher_ema.pt"
                            torch.save(
                                {"epoch": epoch + 1, "model": teacher_model.state_dict(), "map50": teacher_map50},
                                best_teacher_path,
                            )
                            print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] new best teacher EMA: {best_teacher_path}", flush=True)
                except Exception:
                    print(f"[epoch {epoch + 1:03d}/{args.epochs:03d}] eval failed; continuing", flush=True)
                finally:
                    student_model.train()

            if hook_detached_for_io:
                hook = DetectInputFeatureHook(student_model, detach=False)
    finally:
        if hook is not None:
            hook.close()

    if args.epochs == 1 and one_epoch_time_sec is not None:
        print(f"ONE_EPOCH_TIME_SEC: {one_epoch_time_sec:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage-2 SFOD (Teacher-Student YOLO26 end-to-end) + optional MARD (VAR/COV only)")

    # Model & Data
    ap.add_argument("--stage1_model", type=str, required=True, help="Path to Stage1 (AdaBN/RC) checkpoint")
    ap.add_argument("--data", type=str, required=True, help="Path to target domain YAML")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # Training hyperparameters
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    ap.add_argument("--device", type=str, default="0")

    # Pseudo-labeling
    ap.add_argument("--conf_threshold", type=float, default=0.5, help="Fallback pseudo-label confidence threshold")
    ap.add_argument("--conf_thr_one2one", type=float, default=0.5, help="Paper tau_o2o: keep O2O pseudo boxes with conf >= this")
    ap.add_argument("--conf_thr_one2many", type=float, default=0.5, help="Paper tau_o2m: candidate O2M pseudo boxes with conf >= this")

    # NOTE: Args kept for compatibility, but fusion semantics changed:
    ap.add_argument("--dual_iou_consistency", type=float, default=0.6, help="(Deprecated in fusion) kept for compatibility")
    ap.add_argument(
        "--dual_iou_extra",
        type=float,
        default=0.2,
        help="Paper tau_no: add O2M boxes with max IoU <= this against any kept O2O box.",
    )
    ap.add_argument("--dual_iou_duplicate", type=float, default=0.7, help="Paper tau_dup: NMS IoU for O2M extras")
    ap.add_argument("--dual_reliability_gamma", type=float, default=0.5, help="(Deprecated in fusion) kept for compatibility")

    # EMA teacher
    ap.add_argument("--use_ema", action="store_true", default=True)
    ap.add_argument("--ema_momentum", type=float, default=0.999)

    # Training controls
    ap.add_argument("--grad_clip", type=float, default=10.0)
    ap.add_argument("--print_freq", type=int, default=10)
    ap.add_argument("--flush_freq", type=int, default=1)
    ap.add_argument("--save_interval", type=int, default=1)

    # Eval
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--val_interval", type=int, default=1)

    # -----------------------------
    # MARD Regularizer (VAR/COV ONLY)
    # -----------------------------
    ap.add_argument("--use_mard", action="store_true", help="Enable MARD (VAR/COV only)")
    ap.add_argument("--mard_lambda0", type=float, default=0.05, help="Paper lambda0: base regularizer weight")
    ap.add_argument("--mard_lambda_max", type=float, default=0.2, help="Paper lambda_max: clamp lambda_reg to this (<=0 disables clamp)")
    ap.add_argument("--mard_warmup_epochs", type=float, default=5.0, help="Paper warmup: epochs for ramping lambda")
    ap.add_argument("--mard_interval", type=int, default=1, help="Compute reg every N steps")
    ap.add_argument("--mard_conf_gate_thr", type=float, default=0.5, help="Paper confidence gate threshold for lambda")

    # Token sampling controls (needed for var/cov)
    ap.add_argument("--mard_conf_box_thr", type=float, default=0.5, help="Filter pseudo boxes used in MARD sampling")
    ap.add_argument("--mard_topk_boxes", type=int, default=15, help="Paper Kb: top boxes used for MARD sampling")
    ap.add_argument("--mard_pts_per_box", type=int, default=8, help="Paper foreground points per box")
    ap.add_argument("--mard_bg_pts", type=int, default=128, help="Paper background sample points")
    ap.add_argument("--mard_min_tokens", type=int, default=64)
    ap.add_argument("--mard_max_tokens", type=int, default=4096)

    # Which FPN levels to regularize
    ap.add_argument("--mard_levels", type=str, default="p3,p4,p5", help="Comma-separated: p3,p4,p5")
    ap.add_argument("--mard_use_level_assign", type=int, default=1, help="Assign boxes to levels by size (1/0)")
    ap.add_argument("--mard_level_cells", type=float, default=12.0, help="Paper eta: size threshold in cells for level assignment")

    # VICReg-style terms
    ap.add_argument("--mard_var_gamma", type=float, default=1.0)
    ap.add_argument("--mard_var_weight", type=float, default=1.0)
    ap.add_argument("--mard_cov_weight", type=float, default=0.1)

    # Reproducibility
    ap.add_argument("--seed", type=int, default=29, help="Random seed for reproducibility. Use -1 to disable seeding.")
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (slower).")

    args = ap.parse_args()
    main(args)
