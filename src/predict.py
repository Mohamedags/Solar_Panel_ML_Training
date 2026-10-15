"""
Run YOLOv8 inference on solar-panel images for scratch and hotspot detection.

Example:
    python predict.py --model runs/detect/runs/B_w04_GPU_batch8/weights/best.pt --source new_images --imgsz 1024 --name method_B

Outputs:
    - Annotated images
    - detections.csv

Modality filtering (based on the project's annotation scheme):
    - RGB images (_V): remove hotspot detections.
    - Thermal images (_T): remove scratch detections.
    - Unknown suffix: do not apply modality filtering.

Use --no-modality-rule to retain all model detections.
"""
import argparse
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8DetectionLoss

from train import ClassWeightedBCE

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
    
def modality_of(path):
    stem = Path(path).stem
    if stem.endswith("_T"):
        return "thermal"
    if stem.endswith("_V"):
        return "rgb"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="folder with images")
    ap.add_argument("--imgsz", type=int, default=1024, help="use the same value as in training")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--name", default="predict")
    ap.add_argument("--no-modality-rule", action="store_true")

    
    args = ap.parse_args()
    source_dir = Path(args.source)
    if not source_dir.is_dir():
        raise SystemExit(f"Source folder not found: {source_dir}")

    paths = sorted(
        p for p in source_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not paths:
        raise SystemExit(f"No images found in {source_dir}")

    model = YOLO(args.model)

    # Create a unique output directory for each prediction run
    base_dir = Path("runs") / "predict" / "runs" / args.name
    out_dir = base_dir
    counter = 2

    while out_dir.exists():
        out_dir = base_dir.parent / f"{base_dir.name}{counter}"
        counter += 1

    out_dir.mkdir(parents=True, exist_ok=False)


    rows = []
    for p in paths:
        r = model.predict(source=str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        mod = modality_of(p)
        keep = []
        for b in r.boxes:
            name = r.names[int(b.cls)]
            ok = True
            if not args.no_modality_rule:
                if mod == "rgb" and name.startswith("hotspot"):
                    ok = False
                if mod == "thermal" and name.startswith("scratch"):
                    ok = False
            keep.append(ok)
        if keep:
            r.boxes = r.boxes[torch.tensor(keep, dtype=torch.bool)]
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            rows.append([p.name, mod, r.names[int(b.cls)], f"{float(b.conf):.3f}",
                         f"{x1:.0f}", f"{y1:.0f}", f"{x2:.0f}", f"{y2:.0f}"])
        output_path = out_dir / p.name
        if not cv2.imwrite(str(output_path), r.plot()):
            print(f"WARNING: Could not save {output_path}")
        print(f"{p.name} [{mod}]: {len(r.boxes)} detection(s)")

    with open(out_dir / "detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "modality", "class", "conf", "x1", "y1", "x2", "y2"])
        w.writerows(rows)
    print(f"\nSaved annotated images and detections.csv to {out_dir}/")


if __name__ == "__main__":
    main()
