
"""
Prepare a YOLO-format dataset from a CVAT COCO 1.0 export.

Supports three hotspot-label modes:
    merge: merge obvious and doubtful hotspots into one hotspot class.
    split: preserve obvious and doubtful hotspots as separate classes.
    drop: exclude doubtful hotspots from the generated labels.

RGB and thermal images belonging to the same scene are assigned to
the same train/validation split. The split assignment is saved to JSON
so it can be reused across dataset variants.

See README.md for usage instructions and methodological details.
"""

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

MODALITY_SUFFIX = {"_T": "thermal", "_V": "rgb"}   # DJI naming: ..._T.JPG thermal, ..._V.JPG visible


def modality_and_scene(file_name):
    stem = Path(file_name).stem
    for suf, mod in MODALITY_SUFFIX.items():
        if stem.endswith(suf):
            return mod, stem[: -len(suf)]
    return "unknown", stem


def certainty_from(value):
    v = str(value).strip().lower()
    if "doubt" in v or v.startswith("minor") or v in ("lt5", "<5"):
        return "doubt"
    if "obvious" in v or v.startswith("moderate") or v.startswith("severe"):
        return "obvious"
    return None


def get_certainty(ann, attr_keys):
    attrs = ann.get("attributes", {}) or {}
    for key in attr_keys:
        if key in attrs:
            c = certainty_from(attrs[key])
            if c:
                return c
    return None


def base_class(name):
    n = name.strip().lower()
    return "hotspot" if n.startswith("hotspot") else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="CVAT COCO export json")
    ap.add_argument("--images", required=True, help="folder with the original images")
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--mode", choices=["merge", "split", "drop"], required=True)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-file", default="split.json",
                    help="created on first run, reused afterwards so all modes share one split")
    ap.add_argument("--attr-keys", default="hotspot_type,certainty,delta_t_band",
                    help="attribute names to read the certainty from, in priority order")
    ap.add_argument("--unknown-hotspot", choices=["doubt", "obvious", "error"], default="doubt",
                    help="what to do with hotspots that have no readable certainty attribute")
    ap.add_argument("--include-unannotated", action="store_true",
                    help="by default images with NO annotation are skipped (they are the unlabeled ones); "
                         "this flag keeps them as empty background images")
    ap.add_argument("--frame-range", default="",
                    help="e.g. 0-499: only use the images at these positions (0-based, inclusive) "
                         "of the CVAT export order")
    ap.add_argument("--max-side", type=int, default=0,
                    help="if >0, downscale images whose long side exceeds this (e.g. 2048 for the 8000x6000 RGB). "
                         "Labels are normalized, so they stay valid.")
    args = ap.parse_args()
    attr_keys = [k.strip() for k in args.attr_keys.split(",") if k.strip()]

    coco = json.load(open(args.coco, "r", encoding="utf-8"))
    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # ---- keep only the labeled images
    n_total = len(images)
    if args.frame_range:
        lo, hi = [int(x) for x in args.frame_range.split("-")]
        keep_ids = set(sorted(images)[lo:hi + 1])
        images = {i: im for i, im in images.items() if i in keep_ids}
    if not args.include_unannotated:
        images = {i: im for i, im in images.items() if anns_by_img.get(i)}
    print(f"Images in JSON: {n_total} | used: {len(images)} "
          f"(skipped: {n_total - len(images)} unannotated or outside the frame range)")

    # ---- inventory, so you can check the attribute names/values really are what the script expects
    attr_inventory = Counter()
    for a in coco["annotations"]:
        if a["image_id"] not in images:
            continue
        if base_class(cats[a["category_id"]]) == "hotspot":
            for k, v in (a.get("attributes") or {}).items():
                attr_inventory[(k, str(v))] += 1
    print("Categories in JSON:", sorted(cats.values()))
    print("Hotspot attribute values found (name, value): count")
    for (k, v), n in sorted(attr_inventory.items()):
        print(f"   {k} = {v}: {n}")

    # ---- class list
    if args.mode == "split":
        class_names = ["panel", "hotspot_obvious", "hotspot_doubt", "scratch"]
    else:
        class_names = ["panel", "hotspot", "scratch"]
    seen_other = sorted({base_class(n) for n in cats.values()} - {"panel", "hotspot", "scratch"})
    class_names += seen_other
    cls_idx = {n: i for i, n in enumerate(class_names)}

    # ---- scenes and split
    scenes = defaultdict(list)
    for iid, im in images.items():
        mod, scene = modality_and_scene(im["file_name"])
        scenes[scene].append(iid)

    split_path = Path(args.split_file)
    if split_path.exists():
        assignment = json.load(open(split_path))
        print(f"Reusing split from {split_path}")
    else:
        strata = defaultdict(list)
        for scene, ids in scenes.items():
            bases = {base_class(cats[a["category_id"]]) for iid in ids for a in anns_by_img[iid]}
            strata[("hotspot" in bases, "scratch" in bases)].append(scene)
        rng = random.Random(args.seed)
        assignment = {}
        for key, sc in sorted(strata.items()):
            sc = sorted(sc)
            rng.shuffle(sc)
            n_val = round(len(sc) * args.val_split)
            if n_val == 0 and len(sc) >= 2:
                n_val = 1
            for i, s in enumerate(sc):
                assignment[s] = "val" if i < n_val else "train"
        json.dump(assignment, open(split_path, "w"), indent=1)
        print(f"Created split -> {split_path}")
    missing_scenes = [s for s in scenes if s not in assignment]
    if missing_scenes:
        print(f"WARNING: {len(missing_scenes)} scenes not in split file, put in train: {missing_scenes[:5]}")
        for s in missing_scenes:
            assignment[s] = "train"

    # ---- write dataset
    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.images)
    stats = Counter()
    unknown_cert = 0
    dropped_doubt = 0
    missing_files = []
    resize = None
    if args.max_side > 0:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        resize = Image

    for iid, im in images.items():
        mod, scene = modality_and_scene(im["file_name"])
        sp = assignment[scene]
        src = img_dir / im["file_name"]
        if not src.exists():
            src = img_dir / Path(im["file_name"]).name
        if not src.exists():
            missing_files.append(im["file_name"])
            continue
        dst = out / "images" / sp / src.name
        if resize is not None:
            pil = resize.open(src)
            if max(pil.size) > args.max_side:
                scale = args.max_side / max(pil.size)
                pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), resize.LANCZOS)
                pil.convert("RGB").save(dst, quality=95)
            else:
                shutil.copy(src, dst)
        else:
            shutil.copy(src, dst)

        W, H = im["width"], im["height"]
        lines = []
        for a in anns_by_img[iid]:
            base = base_class(cats[a["category_id"]])
            if base == "hotspot":
                cert = get_certainty(a, attr_keys)
                if cert is None:
                    unknown_cert += 1
                    if args.unknown_hotspot == "error":
                        raise SystemExit(f"Hotspot without certainty attribute in {im['file_name']} (ann id {a['id']})")
                    cert = args.unknown_hotspot
                if args.mode == "split":
                    cname = f"hotspot_{cert}"
                elif args.mode == "drop":
                    if cert == "doubt":
                        dropped_doubt += 1
                        continue
                    cname = "hotspot"
                else:
                    cname = "hotspot"
            else:
                cname = base
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            nw, nh = min(nw, 1), min(nh, 1)
            lines.append(f"{cls_idx[cname]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[(sp, mod, cname)] += 1
        (out / "labels" / sp / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        stats[(sp, mod, "_images")] += 1

    yaml_lines = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(class_names)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"\nMode: {args.mode} | classes: {class_names}")
    for sp in ("train", "val"):
        print(f"[{sp}]")
        for mod in ("rgb", "thermal", "unknown"):
            n_img = stats[(sp, mod, "_images")]
            if n_img:
                per_cls = {c: stats[(sp, mod, c)] for c in class_names if stats[(sp, mod, c)]}
                print(f"   {mod}: {n_img} images, boxes {per_cls}")
    if unknown_cert:
        print(f"NOTE: {unknown_cert} hotspots had no readable certainty attribute -> treated as '{args.unknown_hotspot}'")
    if dropped_doubt:
        print(f"NOTE: {dropped_doubt} doubtful hotspots were dropped (they become unlabeled background for the model)")
    if missing_files:
        print(f"WARNING: {len(missing_files)} images not found in {args.images}: {missing_files[:5]}")
    print(f"Done -> {out.resolve()}")


if __name__ == "__main__":
    main()