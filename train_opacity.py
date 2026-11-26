"""
Train a pneumonia-opacity segmentation model (U-Net) on chest X-rays with
radiologist opacity masks, then use it in pneumonia_zones.py (--opacity-model).

Data layout
  images/   <stem>.png|jpg        chest X-rays
  masks/    <stem>.png|json       opacity annotation (non-zero = opacity; or polygon JSON).
                                  A MISSING mask file = normal image (empty mask).
  patients.csv (strongly recommended)   columns: image,patient_id
                                  -> the train/val split is done BY PATIENT so the same child
                                     never appears in both sets (otherwise results are inflated).
  lungs/    <stem>.png            (optional) combined lung masks; if given, validation also
                                  reports per-lung % error and severity-grade agreement.

Run
  python train_opacity.py --images images --masks masks --patients patients.csv \
         --lungs lungs --out opacity_unet.pt --size 512 --epochs 60

Needs: torch, numpy, opencv-python.   GPU strongly recommended at size 512.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pneumonia_zones import (Config, analyze, iter_image_files, load_opacity_mask, split_left_right)

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_IMAGES_DIR = Path(__file__).resolve().parent.parent / "BDCXR-3257"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32, depth=5):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList([_block(1 if i == 0 else ch[i - 1], c) for i, c in enumerate(ch)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth - 1, 0, -1)])
        self.dec = nn.ModuleList([_block(ch[i - 1] * 2, ch[i - 1]) for i in range(depth - 1, 0, -1)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)  # logits


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def preprocess(gray: np.ndarray, size: int) -> np.ndarray:
    g = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0


def find_mask(mask_dir: Path, img_path: Path | str, images_root: Path | None = None):
    p = Path(img_path)
    search_dirs = [mask_dir]
    if images_root is not None:
        rel_parent = p.relative_to(images_root).parent
        if rel_parent != Path("."):
            search_dirs = [mask_dir / rel_parent, mask_dir]
    for d in search_dirs:
        for ext in (".png", ".json", ".jpg", ".tif"):
            cand = d / f"{p.stem}{ext}"
            if cand.exists():
                return cand
    return None


class CXRDataset(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_p, mask_p = self.items[i]
        g = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        h, w = g.shape
        m = load_opacity_mask(mask_p, (h, w)) if mask_p else np.zeros((h, w), np.uint8)
        x = preprocess(g, self.size)
        y = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if self.train:
            x, y = self.augment(x, y)
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]

    def augment(self, x, y):
        s = self.size
        ang, sc = random.uniform(-8, 8), random.uniform(0.92, 1.08)
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
        M[:, 2] += [random.uniform(-0.04, 0.04) * s, random.uniform(-0.04, 0.04) * s]
        x = cv2.warpAffine(x, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        y = cv2.warpAffine(y, M, (s, s), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        x = np.clip(x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08), 0, 1)
        if random.random() < 0.5:                       # mask flips with the image, so this is safe
            x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
        return x.astype(np.float32), y.astype(np.float32)


def build_items(images, masks):
    images_root = Path(images)
    mask_root = Path(masks)
    return [(p, find_mask(mask_root, p, images_root)) for p in iter_image_files(images_root)]


def split_by_patient(items, patients_csv, val_frac, seed, images_root=None):
    rng = random.Random(seed)
    if patients_csv:
        pid = {r["image"]: r["patient_id"] for r in csv.DictReader(open(patients_csv))}
        def key(p):
            rel = p.relative_to(images_root).as_posix() if images_root else p.name
            return pid.get(rel) or pid.get(p.name) or pid.get(p.stem) or p.stem
    else:
        print("WARNING: no --patients csv -> splitting by image. If a child has several X-rays "
              "this LEAKS data between train and val and inflates the scores.")
        key = lambda p: p.stem
    groups = {}
    for it in items:
        groups.setdefault(key(it[0]), []).append(it)
    ids = sorted(groups)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    tr = [it for k in ids if k not in val_ids for it in groups[k]]
    va = [it for k in ids if k in val_ids for it in groups[k]]
    return tr, va


# --------------------------------------------------------------------------- #
# Loss / metrics
# --------------------------------------------------------------------------- #
def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


def dice_score(pred, y, eps=1.0):
    inter = (pred * y).sum((1, 2, 3))
    return ((2 * inter + eps) / (pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps))


@torch.no_grad()
def validate(model, loader, dev, thr=0.5):
    model.eval()
    d_pos, d_all = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        pred = (torch.sigmoid(model(x)) > thr).float()
        d = dice_score(pred, y).cpu().numpy()
        has = y.sum((1, 2, 3)).cpu().numpy() > 0
        d_all += list(d)
        d_pos += list(d[has])
    return (float(np.mean(d_pos)) if d_pos else float("nan")), float(np.mean(d_all))


# --------------------------------------------------------------------------- #
# Inference (used by pneumonia_zones.py)
# --------------------------------------------------------------------------- #
def load_model(ckpt_path, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=dev)
    m = UNet(ck["base"], ck["depth"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["size"], dev


@torch.no_grad()
def predict_opacity(gray: np.ndarray, ckpt_path: str, thr: float | None = None, _cache={}):
    """Return a binary opacity mask at the original image size."""
    if ckpt_path not in _cache:
        _cache[ckpt_path] = load_model(ckpt_path)
    model, size, dev = _cache[ckpt_path]
    ck_thr = torch.load(ckpt_path, map_location="cpu").get("threshold", 0.5) if thr is None else thr
    x = torch.from_numpy(preprocess(gray, size))[None, None].to(dev)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    p = cv2.resize(p, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (p > ck_thr).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Clinical-level validation: % involvement + grade agreement per lung
# --------------------------------------------------------------------------- #
def clinical_eval(model, items, lungs_dir, size, dev, thr, cfg=Config(), images_root=None):
    model.eval()
    err, grade_ok, n = [], 0, 0
    for img_p, mask_p in items:
        lp = find_mask(Path(lungs_dir), img_p, images_root)
        if lp is None:
            continue
        g = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        h, w = g.shape
        lungs = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        lungs = cv2.resize(lungs, (w, h), interpolation=cv2.INTER_NEAREST)
        R, L = split_left_right(lungs, cfg.flip_lr)
        gt = load_opacity_mask(mask_p, (h, w)) if mask_p else np.zeros((h, w), np.uint8)
        x = torch.from_numpy(preprocess(g, size))[None, None].to(dev)
        with torch.no_grad():
            p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        pr = (cv2.resize(p, (w, h)) > thr).astype(np.uint8)
        rg, _ = analyze(R, L, gt, cfg)
        rp, _ = analyze(R, L, pr, cfg)
        for a, b in ((rg.right, rp.right), (rg.left, rp.left)):
            err.append(abs(a.percent - b.percent))
            grade_ok += int(a.grade == b.grade)
            n += 1
    if n == 0:
        return None
    return {"pct_MAE": float(np.mean(err)), "grade_agreement": grade_ok / n, "lungs_evaluated": n}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(DEFAULT_IMAGES_DIR),
                    help=f"folder of chest X-rays (default: {DEFAULT_IMAGES_DIR})")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--patients")
    ap.add_argument("--lungs")
    ap.add_argument("--out", default="opacity_unet.pt")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    images_root = Path(a.images)
    items = build_items(images_root, a.masks)
    tr, va = split_by_patient(items, a.patients, a.val_frac, a.seed, images_root)
    pos = sum(1 for _, m in tr if m)
    print(f"{len(items)} images | train {len(tr)} ({pos} with opacity) | val {len(va)} | device {dev}")

    tl = DataLoader(CXRDataset(tr, a.size, True), a.batch, shuffle=True, num_workers=2, drop_last=len(tr) > a.batch)
    vl = DataLoader(CXRDataset(va, a.size, False), a.batch, num_workers=2)

    model = UNet(a.base, a.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bce = nn.BCEWithLogitsLoss()

    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        d_pos, d_all = validate(model, vl, dev)
        score = d_pos if not np.isnan(d_pos) else d_all
        print(f"ep {ep:3d}  loss {tot / len(tr):.4f}  val Dice(pneumonia imgs) {d_pos:.3f}  Dice(all) {d_all:.3f}")
        if score > best:
            best = score
            torch.save({"state": model.state_dict(), "size": a.size, "base": a.base,
                        "depth": a.depth, "threshold": 0.5, "val_dice": score}, a.out)

    print(f"best val Dice {best:.3f} -> {a.out}")
    if a.lungs:
        model, _, _ = load_model(a.out, dev)
        res = clinical_eval(model, va, a.lungs, a.size, dev, 0.5, images_root=images_root)
        print("clinical validation (per lung):", res)


if __name__ == "__main__":
    main()
