import argparse
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Ensure this script uses the local editable Ultralytics checkout, not site-packages.
SCRIPT_DIR = Path(__file__).resolve().parent
ULTRALYTICS_REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = ULTRALYTICS_REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(ULTRALYTICS_REPO_ROOT))


DEFAULT_CHECKPOINTS_DIR = str(
    ULTRALYTICS_REPO_ROOT / "runs/May42026/YOLO11M/C2F/stage2_mard/checkpoints"
)
DEFAULT_PATTERN = "*student_epoch_*.pt"
DEFAULT_DATA = "/data/ai20resch13001/SFOD-done-efficiently/data/foggy_cityscapes/yolo/foggy_cityscapes.yaml"
DEFAULT_OUT_NAME = "validation_all_student_checkpoints.txt"


# Compatibility shim for checkpoints that may reference script-local classes.
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


def epoch_from_checkpoint(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def discover_checkpoints(checkpoints_dir: Path, pattern: str, start_epoch: int, end_epoch: int) -> list[Path]:
    checkpoints = [p for p in checkpoints_dir.glob(pattern) if p.is_file()]
    checkpoints = sorted(checkpoints, key=lambda p: (epoch_from_checkpoint(p), p.name))
    if start_epoch > 0:
        checkpoints = [p for p in checkpoints if epoch_from_checkpoint(p) >= start_epoch]
    if end_epoch > 0:
        checkpoints = [p for p in checkpoints if epoch_from_checkpoint(p) <= end_epoch]
    return checkpoints


def default_out_file(checkpoints_dir: Path) -> Path:
    run_dir = checkpoints_dir.parent if checkpoints_dir.name == "checkpoints" else checkpoints_dir
    return run_dir / DEFAULT_OUT_NAME


def metric_value(metrics: Any, key: str) -> float:
    results = getattr(metrics, "results_dict", {}) or {}
    value = results.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def box_value(metrics: Any, attr: str) -> float:
    box = getattr(metrics, "box", None)
    value = getattr(box, attr, float("nan")) if box is not None else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def speed_value(metrics: Any, key: str) -> float:
    speed = getattr(metrics, "speed", {}) or {}
    value = speed.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6f}"


def write_header(f, args, checkpoints: list[Path], device: torch.device) -> None:
    f.write("# Stage-2 student checkpoint validation on foggy Cityscapes\n")
    f.write(f"# created: {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"# checkpoints_dir: {Path(args.checkpoints_dir).resolve()}\n")
    f.write(f"# pattern: {args.pattern}\n")
    f.write(f"# checkpoint_count: {len(checkpoints)}\n")
    f.write(f"# data: {args.data}\n")
    f.write(f"# imgsz: {args.imgsz}\n")
    f.write(f"# batch: {args.batch}\n")
    f.write(f"# conf: {args.conf}\n")
    f.write(f"# iou: {args.iou}\n")
    f.write(f"# device: {device}\n")
    f.write(
        "epoch\tcheckpoint\tprecision\trecall\tmAP50\tmAP50-95\tfitness\t"
        "preprocess_ms\tinference_ms\tloss_ms\tpostprocess_ms\tstatus\terror\n"
    )
    f.flush()


def row_to_line(row: dict[str, Any]) -> str:
    return "\t".join(
        [
            str(row["epoch"]),
            row["checkpoint"],
            fmt_float(row["precision"]),
            fmt_float(row["recall"]),
            fmt_float(row["map50"]),
            fmt_float(row["map"]),
            fmt_float(row["fitness"]),
            fmt_float(row["preprocess_ms"]),
            fmt_float(row["inference_ms"]),
            fmt_float(row["loss_ms"]),
            fmt_float(row["postprocess_ms"]),
            row["status"],
            row["error"].replace("\t", " ").replace("\n", " "),
        ]
    )


def validate_checkpoint(ckpt: Path, args, device: torch.device) -> dict[str, Any]:
    epoch = epoch_from_checkpoint(ckpt)
    row = {
        "epoch": epoch,
        "checkpoint": str(ckpt),
        "precision": float("nan"),
        "recall": float("nan"),
        "map50": float("nan"),
        "map": float("nan"),
        "fitness": float("nan"),
        "preprocess_ms": float("nan"),
        "inference_ms": float("nan"),
        "loss_ms": float("nan"),
        "postprocess_ms": float("nan"),
        "status": "ok",
        "error": "",
    }

    model = None
    try:
        from ultralytics import YOLO

        model = YOLO(str(ckpt))
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=val_device_arg(device),
            workers=args.workers,
            plots=args.plots,
            verbose=args.ultralytics_verbose,
            project=args.val_project,
            name=f"epoch_{epoch:03d}" if epoch >= 0 else ckpt.stem,
            exist_ok=True,
        )

        row["precision"] = metric_value(metrics, "metrics/precision(B)")
        row["recall"] = metric_value(metrics, "metrics/recall(B)")
        row["map50"] = metric_value(metrics, "metrics/mAP50(B)")
        row["map"] = metric_value(metrics, "metrics/mAP50-95(B)")
        row["fitness"] = metric_value(metrics, "fitness")

        # Fall back to box attributes if results_dict is missing a key.
        if not math.isfinite(row["precision"]):
            row["precision"] = box_value(metrics, "mp")
        if not math.isfinite(row["recall"]):
            row["recall"] = box_value(metrics, "mr")
        if not math.isfinite(row["map50"]):
            row["map50"] = box_value(metrics, "map50")
        if not math.isfinite(row["map"]):
            row["map"] = box_value(metrics, "map")

        row["preprocess_ms"] = speed_value(metrics, "preprocess")
        row["inference_ms"] = speed_value(metrics, "inference")
        row["loss_ms"] = speed_value(metrics, "loss")
        row["postprocess_ms"] = speed_value(metrics, "postprocess")
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return row


def print_best_summary(f, rows: list[dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok" and math.isfinite(r["map50"])]
    if not ok_rows:
        f.write("# best_mAP50: none\n")
        f.write("# best_mAP50-95: none\n")
        return

    best_map50 = max(ok_rows, key=lambda r: r["map50"])
    best_map = max(ok_rows, key=lambda r: r["map"] if math.isfinite(r["map"]) else -1.0)
    f.write(
        f"# best_mAP50: epoch={best_map50['epoch']} mAP50={best_map50['map50']:.6f} "
        f"mAP50-95={best_map50['map']:.6f} checkpoint={best_map50['checkpoint']}\n"
    )
    f.write(
        f"# best_mAP50-95: epoch={best_map['epoch']} mAP50={best_map['map50']:.6f} "
        f"mAP50-95={best_map['map']:.6f} checkpoint={best_map['checkpoint']}\n"
    )


def main(args) -> None:
    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {checkpoints_dir}")

    if args.out_file is None:
        args.out_file = str(default_out_file(checkpoints_dir))

    out_file = Path(args.out_file).expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.val_project is None:
        args.val_project = str(out_file.parent / "validation_runs")

    device = resolve_device(args.device)
    checkpoints = discover_checkpoints(checkpoints_dir, args.pattern, args.start_epoch, args.end_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir} with pattern {args.pattern}")

    print(f"[validate] device={device} checkpoints={len(checkpoints)} data={args.data}", flush=True)
    print(f"[validate] writing results to: {out_file}", flush=True)

    rows = []
    start_time = time.time()
    with out_file.open("w", buffering=1) as f:
        write_header(f, args, checkpoints, device)

        for index, ckpt in enumerate(checkpoints, start=1):
            epoch = epoch_from_checkpoint(ckpt)
            print(f"[validate] {index:02d}/{len(checkpoints):02d} epoch={epoch} checkpoint={ckpt.name}", flush=True)
            row = validate_checkpoint(ckpt, args, device)
            rows.append(row)
            f.write(row_to_line(row) + "\n")
            f.flush()

            if row["status"] == "ok":
                print(
                    f"[validate] epoch={epoch} mAP50={row['map50']:.4f} "
                    f"mAP50-95={row['map']:.4f} precision={row['precision']:.4f} recall={row['recall']:.4f}",
                    flush=True,
                )
            else:
                print(f"[validate] epoch={epoch} failed: {row['error']}", flush=True)

        elapsed_min = (time.time() - start_time) / 60.0
        f.write(f"# elapsed_minutes: {elapsed_min:.2f}\n")
        print_best_summary(f, rows)

    print(f"[validate] done in {(time.time() - start_time) / 60.0:.2f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate Stage-2 student checkpoints on foggy Cityscapes.")
    ap.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR)
    ap.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob used inside checkpoints_dir. Default matches YOLO11/YOLO26 student epoch checkpoints.",
    )
    ap.add_argument("--data", type=str, default=DEFAULT_DATA)
    ap.add_argument(
        "--out_file",
        type=str,
        default=None,
        help=f"Results TSV path. Defaults to <run_dir>/{DEFAULT_OUT_NAME}.",
    )
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--start_epoch", type=int, default=0, help="Optional inclusive lower epoch filter.")
    ap.add_argument("--end_epoch", type=int, default=0, help="Optional inclusive upper epoch filter.")
    ap.add_argument("--val_project", type=str, default=None, help="Directory for Ultralytics validation run folders.")
    ap.add_argument("--plots", action="store_true", help="Save Ultralytics validation plots.")
    ap.add_argument("--ultralytics_verbose", action="store_true", help="Enable Ultralytics per-class validation output.")
    main(ap.parse_args())
