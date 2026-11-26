"""
Radiographic lung-zone localization and per-lung pneumonia severity grading
for frontal chest X-rays (SCA paediatric pneumonia project).

Implements the Meeting Report / Proposed Algorithm:
  * each lung field is split into 3 equal-height zones (upper / middle / lower)
        Right: RUZ, RMZ, RLZ      Left: LUZ, LMZ, LLZ
    (radiographic zones, NOT anatomical lobes)
  * opacity mask is restricted to each lung independently
  * P_R = A_PR / A_R * 100 ,  P_L = A_PL / A_L * 100  (each lung = its own 100%)
  * severity grade per lung:  0 | >0-25 | >25-50 | >50-75 | >75-100
  * outputs: status (absent / present / indeterminate), laterality,
             affected zones, per-lung %, per-lung grade, overlay image

Convention: standard frontal CXR display -> the patient's RIGHT lung appears on
the LEFT side of the image (smaller x). Use flip_lr=True if your images are mirrored.

Dependencies: numpy, opencv-python   (optional: torch + torchxrayvision)
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ZONE_NAMES = {
    "right": ["RUZ", "RMZ", "RLZ"],   # top -> bottom
    "left":  ["LUZ", "LMZ", "LLZ"],
}

# Proposed operational scale (needs radiologist validation).
# (upper bound inclusive in %, grade, label)
SEVERITY_SCALE = [
    (25.0,  1, "Mild"),
    (50.0,  2, "Moderate"),
    (75.0,  3, "Severe"),
    (100.0, 4, "Very severe"),
]


@dataclass
class Config:
    flip_lr: bool = False              # True if image is mirrored (right lung on image right)
    zone_min_overlap_px: int = 1       # zone "affected" if overlap >= this many pixels (>0 per report)
    zone_min_overlap_frac: float = 0.0 # ...and >= this fraction of the zone area (set e.g. 0.02 once agreed)
    present_min_frac: float = 0.005    # total opacity < this fraction of lung area -> "indeterminate"


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #
def severity_grade(percent: float) -> tuple[int, str]:
    """Grade 0 for exactly 0 %, otherwise per SEVERITY_SCALE (non-overlapping bounds)."""
    if percent <= 0:
        return 0, "No radiographic involvement"
    for upper, grade, label in SEVERITY_SCALE:
        if percent <= upper:
            return grade, label
    return 4, "Very severe"


# --------------------------------------------------------------------------- #
# Lung masks
# --------------------------------------------------------------------------- #
def split_left_right(lungs: np.ndarray, flip_lr: bool = False):
    """
    Split a combined binary lung mask into (right_mask, left_mask) (patient sides).
    Uses the two largest connected components; if the lungs are merged, cuts at the
    column-sum valley near the middle.
    """
    lungs = (lungs > 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(lungs, connectivity=8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1  # largest first
    h, w = lungs.shape

    if len(order) >= 2 and stats[order[1], cv2.CC_STAT_AREA] > 0.2 * stats[order[0], cv2.CC_STAT_AREA]:
        a, b = order[0], order[1]
        img_left, img_right = (a, b) if cents[a][0] < cents[b][0] else (b, a)
        img_left_m = (lab == img_left).astype(np.uint8)
        img_right_m = (lab == img_right).astype(np.uint8)
    else:  # merged -> cut at valley
        cols = lungs.sum(axis=0).astype(float)
        xs = np.where(cols > 0)[0]
        if len(xs) == 0:
            z = np.zeros_like(lungs)
            return z, z
        lo, hi = xs.min(), xs.max()
        c0, c1 = int(lo + 0.35 * (hi - lo)), int(lo + 0.65 * (hi - lo))
        cut = c0 + int(np.argmin(cols[c0:c1 + 1]))
        img_left_m, img_right_m = lungs.copy(), lungs.copy()
        img_left_m[:, cut:] = 0
        img_right_m[:, :cut] = 0

    # image-left == patient RIGHT in the standard display
    right, left = (img_left_m, img_right_m) if not flip_lr else (img_right_m, img_left_m)
    return right, left


def segment_lungs_classical(gray: np.ndarray) -> np.ndarray:
    """
    Crude, dependency-free lung-field estimate (Otsu on CLAHE image + morphology).
    ONLY a placeholder for testing - replace with a trained model (see
    segment_lungs_xrv) or radiologist lung masks for real work.
    """
    g = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    g = cv2.GaussianBlur(g, (0, 0), 3)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # drop components touching the border (air outside body) and tiny ones
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    out = np.zeros_like(m)
    keep = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area > 0.01 * h * w:
            keep.append((area, i))
    for _, i in sorted(keep, reverse=True)[:2]:
        out[lab == i] = 1
    return out


_XRV_LUNG_CACHE: dict[str, tuple] = {}
_XRV_OPACITY_CACHE: dict[str, tuple] = {}


def segment_lungs_xrv(gray: np.ndarray, flip_lr: bool = False):
    """
    Pre-trained lung segmentation via torchxrayvision (ChestX-Det PSPNet).
    pip install torch torchxrayvision.  Returns (right_mask, left_mask) at input size.
    NOTE: trained primarily on adult CXRs - verify visually on paediatric images.
    """
    import torch
    import torchxrayvision as xrv

    if "pspnet" not in _XRV_LUNG_CACHE:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = xrv.baseline_models.chestx_det.PSPNet().to(dev)
        model.eval()
        _XRV_LUNG_CACHE["pspnet"] = (model, dev)
    model, dev = _XRV_LUNG_CACHE["pspnet"]

    h, w = gray.shape
    img = cv2.resize(gray, (512, 512)).astype(np.float32)
    img = xrv.datasets.normalize(img, 255)[None, ...]
    with torch.no_grad():
        out = model(torch.from_numpy(img)[None, ...].to(dev))
    prob = torch.sigmoid(out)[0].cpu().numpy()
    t = model.targets
    r = (prob[t.index("Right Lung")] > 0.5).astype(np.uint8)
    l = (prob[t.index("Left Lung")] > 0.5).astype(np.uint8)
    r = cv2.resize(r, (w, h), interpolation=cv2.INTER_NEAREST)
    l = cv2.resize(l, (w, h), interpolation=cv2.INTER_NEAREST)
    return (l, r) if flip_lr else (r, l)


def predict_opacity_xrv(
    gray: np.ndarray,
    lung_mask: np.ndarray | None = None,
    prob_thresh: float = 0.5,
    cam_thresh: float = 0.45,
    weights: str = "densenet121-res224-rsna",
) -> np.ndarray:
    """
    Experimental pretrained opacity estimator using TorchXRayVision DenseNet + Grad-CAM.
    Produces an approximate binary mask inside the lung fields when predicted opacity
    probability >= prob_thresh; NOT calibrated for clinical pixel-area grading.
    """
    import torch
    import torch.nn.functional as F
    import torchxrayvision as xrv

    if weights not in _XRV_OPACITY_CACHE:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = xrv.models.DenseNet(weights=weights).to(dev)
        model.eval()
        _XRV_OPACITY_CACHE[weights] = (model, dev)
    model, dev = _XRV_OPACITY_CACHE[weights]

    h, w = gray.shape
    res = getattr(model, "input_resolution", 224)
    img = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = xrv.datasets.normalize(img, 255)[None, None, ...]
    x = torch.from_numpy(img).to(dev)

    target_name = "Lung Opacity" if "Lung Opacity" in model.pathologies else "Pneumonia"
    target_idx = model.pathologies.index(target_name)

    with torch.enable_grad():
        feat = F.relu(model.features(x), inplace=False)
        feat.retain_grad()
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        logits = model.classifier(pooled)
        probs = torch.sigmoid(logits)
        if getattr(model, "op_threshs", None) is not None:
            probs = xrv.models.op_norm(probs, model.op_threshs.to(dev))
        if float(probs[0, target_idx].detach().cpu()) < prob_thresh:
            return np.zeros((h, w), np.uint8)
        model.zero_grad(set_to_none=True)
        logits[0, target_idx].backward()
        grads = feat.grad[0]
        acts = feat.detach()[0]

    w_cam = grads.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((w_cam * acts).sum(dim=0)).cpu().numpy()
    if cam.max() <= 1e-8:
        return np.zeros((h, w), np.uint8)
    cam = cam / cam.max()
    cam_full = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if lung_mask is not None and np.any(lung_mask > 0):
        cam_max_in_lung = float(cam_full[lung_mask > 0].max())
        if cam_max_in_lung <= 1e-8:
            return np.zeros((h, w), np.uint8)
        cam_full = cam_full / cam_max_in_lung
        return ((cam_full >= cam_thresh) & (lung_mask > 0)).astype(np.uint8)
    return (cam_full >= cam_thresh).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #
def make_zones(lung_mask: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """
    Split one lung mask into three zones of equal projected HEIGHT, measured over the
    lung's own vertical extent (first->last row containing lung). Zones are clipped to
    the mask, and do not overlap.
    """
    rows = np.where(lung_mask.any(axis=1))[0]
    zones = {n: np.zeros_like(lung_mask, dtype=np.uint8) for n in names}
    if len(rows) == 0:
        return zones
    top, bottom = rows.min(), rows.max() + 1
    edges = np.linspace(top, bottom, 4).round().astype(int)  # 4 edges -> 3 bands
    for i, n in enumerate(names):
        band = np.zeros_like(lung_mask, dtype=np.uint8)
        band[edges[i]:edges[i + 1], :] = 1
        zones[n] = (band & (lung_mask > 0)).astype(np.uint8)
    return zones


# --------------------------------------------------------------------------- #
# Opacity masks (annotation input)
# --------------------------------------------------------------------------- #
def load_opacity_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """
    Opacity annotation as either
      * a mask image (.png/.jpg/.tif; non-zero = opacity), or
      * a JSON file {"polygons": [[[x,y],...], ...]} or LabelMe format
        ({"shapes":[{"points":[[x,y],...]}, ...]}).
    Polygons are rasterised (union -> pixels counted once).
    """
    path = Path(path)
    h, w = shape
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        polys = data.get("polygons") or [s["points"] for s in data.get("shapes", [])]
        mask = np.zeros((h, w), np.uint8)
        for p in polys:
            cv2.fillPoly(mask, [np.round(np.array(p)).astype(np.int32)], 1)
        return mask
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
@dataclass
class LungReport:
    lung_area_px: int
    opacity_area_px: int
    percent: float
    grade: int
    grade_label: str
    affected_zones: list[str]
    zone_percent_of_zone: dict[str, float]


@dataclass
class Report:
    status: str                  # absent | present | indeterminate
    laterality: str              # none | right | left | bilateral
    affected_zones: list[str]    # all, e.g. ["RLZ","LUZ","LMZ"]
    right: LungReport
    left: LungReport
    flags: list[str] = field(default_factory=list)


def analyze(right_mask, left_mask, opacity_mask, cfg: Config = Config(),
            force_indeterminate: bool = False):
    """Core algorithm (steps 5-15 of the pseudocode). All masks are HxW binary."""
    flags = []
    reports, zone_masks = {}, {}
    for side, lung in (("right", right_mask), ("left", left_mask)):
        lung = (lung > 0).astype(np.uint8)
        if lung.sum() == 0:
            flags.append(f"{side} lung segmentation empty - review")
        names = ZONE_NAMES[side]
        zones = make_zones(lung, names)
        op = (opacity_mask > 0).astype(np.uint8) & lung       # restrict to THIS lung
        a_lung, a_op = int(lung.sum()), int(op.sum())
        pct = 100.0 * a_op / a_lung if a_lung else 0.0
        g, lab = severity_grade(pct)

        affected, zpct = [], {}
        for n in names:
            za = int(zones[n].sum())
            ov = int((op & zones[n]).sum())       # each pixel belongs to exactly one zone
            zpct[n] = round(100.0 * ov / za, 2) if za else 0.0
            if ov >= cfg.zone_min_overlap_px and za and ov / za >= cfg.zone_min_overlap_frac and ov > 0:
                affected.append(n)
        reports[side] = LungReport(a_lung, a_op, round(pct, 2), g, lab, affected, zpct)
        zone_masks[side] = (lung, zones, op)

    r, l = reports["right"], reports["left"]
    total_op = r.opacity_area_px + l.opacity_area_px
    total_lung = r.lung_area_px + l.lung_area_px
    if total_op == 0:
        status = "absent"
    elif force_indeterminate or (total_lung and total_op / total_lung < cfg.present_min_frac):
        status = "indeterminate"
    else:
        status = "present"

    sides = [s for s, rep in (("right", r), ("left", l)) if rep.affected_zones]
    laterality = "bilateral" if len(sides) == 2 else (sides[0] if sides else "none")
    rep = Report(status, laterality, r.affected_zones + l.affected_zones, r, l, flags)
    return rep, zone_masks


# --------------------------------------------------------------------------- #
# Visual output
# --------------------------------------------------------------------------- #
def draw_overlay(gray, zone_masks, report: Report, alpha=0.4):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colours = {"right": (255, 160, 0), "left": (0, 200, 0)}
    for side, (lung, zones, op) in zone_masks.items():
        cnts, _ = cv2.findContours(lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, colours[side], 2)
        for n, z in zones.items():
            zc, _ = cv2.findContours(z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, zc, -1, (255, 255, 0), 1)
            ys, xs = np.where(z)
            if len(xs):
                txt = n + ("*" if n in report.affected_zones else "")
                cv2.putText(vis, txt, (int(xs.mean()) - 18, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # opacity in red
    red = np.zeros_like(vis)
    for _, (_, _, op) in zone_masks.items():
        red[op > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 1.0, red, alpha, 0)
    # text banner
    r, l = report.right, report.left
    lines = [f"Status: {report.status} | Laterality: {report.laterality}",
             f"Right lung: {r.percent:.1f}% - Grade {r.grade} ({r.grade_label})",
             f"Left lung:  {l.percent:.1f}% - Grade {l.grade} ({l.grade_label})",
             f"Zones: {', '.join(report.affected_zones) or '-'}"]
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, t, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------- #
# Batch labelling CLI
# --------------------------------------------------------------------------- #
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_IMAGES_DIR = Path(__file__).resolve().parent.parent / "BDCXR-3257"


def _find_matching_file(base_dir: str | Path | None, img_path: Path, images_root: Path) -> Path | None:
    """Locate a matching mask file in base_dir, checking mirrored subfolders then flat root."""
    if not base_dir:
        return None
    base = Path(base_dir)
    rel_parent = img_path.relative_to(images_root).parent
    search_dirs = [base / rel_parent, base] if rel_parent != Path(".") else [base]
    for d in search_dirs:
        for ext in (".png", ".json", ".jpg", ".tif"):
            cand = d / f"{img_path.stem}{ext}"
            if cand.exists():
                return cand
    return None


def iter_image_files(images_root: str | Path) -> list[Path]:
    """Return sorted image files under images_root, including nested class subfolders."""
    root = Path(images_root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)


def process_image(
    img_path,
    opacity_path,
    lung_path,
    cfg,
    out_dir,
    lung_backend,
    opacity_model=None,
    opacity_backend="none",
    xrv_prob_thresh=0.5,
    xrv_cam_thresh=0.45,
):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    h, w = gray.shape

    if lung_path:                                   # radiologist / model lung mask (both lungs)
        lungs = cv2.imread(str(lung_path), cv2.IMREAD_GRAYSCALE)
        lungs = cv2.resize(lungs, (w, h), interpolation=cv2.INTER_NEAREST)
        right, left = split_left_right(lungs, cfg.flip_lr)
    elif lung_backend == "xrv":
        right, left = segment_lungs_xrv(gray, cfg.flip_lr)
    else:
        right, left = split_left_right(segment_lungs_classical(gray), cfg.flip_lr)

    if opacity_path:                                # radiologist annotation wins
        opacity = load_opacity_mask(opacity_path, (h, w))
    elif opacity_model:                             # trained U-Net (train_opacity.py)
        from train_opacity import predict_opacity
        opacity = predict_opacity(gray, opacity_model)
    elif opacity_backend == "xrv-rsna":             # experimental pretrained RSNA Grad-CAM
        opacity = predict_opacity_xrv(
            gray,
            lung_mask=(right | left),
            prob_thresh=xrv_prob_thresh,
            cam_thresh=xrv_cam_thresh,
        )
    else:
        opacity = np.zeros((h, w), np.uint8)
    rep, zm = analyze(right, left, opacity, cfg)
    vis = draw_overlay(gray, zm, rep)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{Path(img_path).stem}_overlay.png"), vis)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="?", default=str(DEFAULT_IMAGES_DIR),
                    help=f"folder of chest X-rays (default: {DEFAULT_IMAGES_DIR})")
    ap.add_argument("--opacity-dir", help="folder with <stem>.png or <stem>.json opacity annotations")
    ap.add_argument("--lung-dir", help="folder with <stem>.png combined lung masks (optional)")
    ap.add_argument("--lung-backend", choices=["classical", "xrv"], default="classical")
    ap.add_argument("--opacity-model", help="checkpoint from train_opacity.py (used when no annotation exists)")
    ap.add_argument("--opacity-backend", choices=["none", "xrv-rsna"], default="none",
                    help="experimental pretrained opacity estimator when no annotation/model is given")
    ap.add_argument("--xrv-prob-thresh", type=float, default=0.5,
                    help="operating probability threshold for --opacity-backend xrv-rsna")
    ap.add_argument("--xrv-cam-thresh", type=float, default=0.45,
                    help="Grad-CAM activation threshold inside lung mask for xrv-rsna")
    ap.add_argument("--out", default="results")
    ap.add_argument("--flip-lr", action="store_true")
    ap.add_argument("--zone-min-frac", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="optional max number of images to process")
    a = ap.parse_args()

    images_root = Path(a.images)
    if not images_root.exists():
        raise FileNotFoundError(f"Images folder not found: {images_root}")

    cfg = Config(flip_lr=a.flip_lr, zone_min_overlap_frac=a.zone_min_frac)
    rows, all_json = [], {}
    image_paths = iter_image_files(images_root)
    if a.limit is not None:
        image_paths = image_paths[:a.limit]

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for p in image_paths:
        rel_path = p.relative_to(images_root)
        rel_key = rel_path.as_posix()
        overlay_dir = out_root / rel_path.parent if rel_path.parent != Path(".") else out_root
        rep = process_image(
            p,
            _find_matching_file(a.opacity_dir, p, images_root),
            _find_matching_file(a.lung_dir, p, images_root),
            cfg,
            overlay_dir,
            a.lung_backend,
            a.opacity_model,
            a.opacity_backend,
            a.xrv_prob_thresh,
            a.xrv_cam_thresh,
        )
        all_json[rel_key] = asdict(rep)
        rows.append({
            "image": rel_key, "status": rep.status, "laterality": rep.laterality,
            "affected_zones": ";".join(rep.affected_zones),
            "right_pct": rep.right.percent, "right_grade": rep.right.grade,
            "left_pct": rep.left.percent, "left_grade": rep.left.grade,
            "flags": ";".join(rep.flags),
        })
    (out_root / "labels.json").write_text(json.dumps(all_json, indent=2))
    if rows:
        with open(out_root / "labels.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
    print(f"Labelled {len(rows)} images -> {out_root}/labels.csv, labels.json, *_overlay.png")


if __name__ == "__main__":
    main()
