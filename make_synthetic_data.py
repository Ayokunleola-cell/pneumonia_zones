"""Tiny synthetic dataset to smoke-test the training pipeline (NOT medical data)."""
import sys, csv, numpy as np, cv2
from pathlib import Path
out = Path(sys.argv[1] if len(sys.argv) > 1 else "synth"); n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
for d in ("images", "masks", "lungs"): (out / d).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0); rows = []
for i in range(n):
    g = np.full((256, 256), 50, np.uint8); lung = np.zeros_like(g)
    cv2.ellipse(lung, (75, 130), (45, 90), 0, 0, 360, 1, -1); cv2.ellipse(lung, (180, 130), (45, 90), 0, 0, 360, 1, -1)
    g[lung > 0] = 90; m = np.zeros_like(g)
    if i % 3:                                         # 2/3 have pneumonia
        for _ in range(rng.integers(1, 3)):
            cx, cy = int(rng.choice([75, 180])), int(rng.integers(70, 190))
            cv2.circle(m, (cx + int(rng.integers(-15, 15)), cy), int(rng.integers(10, 25)), 1, -1)
        m &= lung
    g[m > 0] = 170
    g = np.clip(g + rng.normal(0, 8, g.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "images" / f"p{i}.png"), g); cv2.imwrite(str(out / "lungs" / f"p{i}.png"), lung * 255)
    if m.any(): cv2.imwrite(str(out / "masks" / f"p{i}.png"), m * 255)
    rows.append((f"p{i}.png", f"pt{i // 2}"))
with open(out / "patients.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image", "patient_id"]); w.writerows(rows)
