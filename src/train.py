"""
Train a YOLOv8 model for solar-panel defect detection.

Method A:
    Single merged hotspot class.

Method B:
    Separate hotspot_obvious and hotspot_doubt classes, with optional
    class-weighted classification loss.

See README.md for setup instructions, experiment configurations,
class definitions, and example commands.
"""

import argparse

import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8DetectionLoss


class ClassWeightedBCE(nn.Module):
    """Element-wise BCE multiplied by a per-class weight vector (broadcast over the last dim)."""

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return loss * self.w.to(device=loss.device, dtype=loss.dtype)


WEIGHTS_BY_NAME = {}


class WeightedModel(DetectionModel):
    def init_criterion(self):
        crit = v8DetectionLoss(self)
        names = [self.names[i] for i in range(len(self.names))]
        crit.bce = ClassWeightedBCE(
            [WEIGHTS_BY_NAME.get(n, 1.0) for n in names]
        )
        if RANK in (-1, 0):
            print(
                "Class-weighted classification loss:",
                {n: WEIGHTS_BY_NAME.get(n, 1.0) for n in names},
            )
        return crit


class WeightedTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = WeightedModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model 


def make_weighted_trainer(weights_by_name):
    global WEIGHTS_BY_NAME
    WEIGHTS_BY_NAME = dict(weights_by_name)
    return WeightedTrainer


def parse_weights(text):
    out = {}
    for part in text.split(","):
        if part.strip():
            k, v = part.split("=")
            out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt", help="try yolov8s.pt if the GPU allows it")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8, help="lower to 4 if you run out of GPU memory")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", required=True)
    ap.add_argument("--class-weights", default="", help="e.g. hotspot_doubt=0.4,hotspot_obvious=1.0")
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu for CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use (smoke tests)")
    args = ap.parse_args()

    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=args.device, workers=args.workers,
        seed=args.seed, deterministic=True, fraction=args.fraction,
        # --- augmentation ---
        degrees=0.0,      # rotation inflates the boxes of thin scratches
        flipud=0.5, fliplr=0.5,
        hsv_h=0.0,        # no hue shift: the thermal palette colour encodes temperature
        hsv_s=0.4, hsv_v=0.3,
        translate=0.1, scale=0.3,
        mosaic=1.0, close_mosaic=10,
        mixup=0.0,        # blending an RGB and a thermal image creates unrealistic samples
        project="runs", name=args.name, exist_ok=True,
    )

    model = YOLO(args.model)
    if args.class_weights:
        weights = parse_weights(args.class_weights)
        names = yaml.safe_load(open(args.data))["names"]
        names = list(names.values()) if isinstance(names, dict) else list(names)
        unknown = [k for k in weights if k not in names]
        if unknown:
            raise SystemExit(f"Unknown class names in --class-weights: {unknown}. data.yaml has: {names}")
        model.train(trainer=make_weighted_trainer(weights), **train_kwargs)
    else:
        model.train(**train_kwargs)

    print(f"\nBest weights: runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
