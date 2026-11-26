import numpy as np
from pneumonia_zones import *

def lungs():
    R = np.zeros((300, 400), np.uint8); L = R.copy()
    R[30:270, 40:180] = 1      # patient right = image left
    L[30:270, 220:360] = 1
    return R, L

def test_grades():
    assert severity_grade(0)[0] == 0
    assert severity_grade(21.8)[1] == "Mild"          # radiologist's sketch example
    assert [severity_grade(p)[0] for p in (25, 25.01, 50, 50.01, 75, 75.01, 100)] == [1, 2, 2, 3, 3, 4, 4]

def test_zones_partition():
    R, _ = lungs()
    z = make_zones(R, ZONE_NAMES["right"])
    assert sum(m.sum() for m in z.values()) == R.sum()
    assert all(abs(int(m.sum()) - R.sum() / 3) < 200 for m in z.values())

def test_unilateral_independence():
    R, L = lungs()
    op = np.zeros_like(R)
    op[30:110, 220:360] = 1                            # upper third of LEFT lung only
    rep, _ = analyze(R, L, op)
    assert rep.right.percent == 0 and rep.right.grade == 0
    assert rep.left.affected_zones == ["LUZ"] and rep.laterality == "left"
    assert abs(rep.left.percent - 100 * 80 / 240) < 0.1   # actual area, not auto 33.3%
    assert rep.left.grade == 2

def test_split_merged_and_bilateral():
    R, L = lungs()
    r, l = split_left_right(R | L)
    assert r[:, :200].sum() == r.sum() and l[:, 200:].sum() == l.sum()
    op = np.zeros_like(R); op[200:260, 60:160] = 1; op[100:140, 240:340] = 1
    rep, _ = analyze(r, l, op)
    assert rep.laterality == "bilateral" and "RLZ" in rep.affected_zones and "LMZ" in rep.affected_zones

def test_status():
    R, L = lungs()
    assert analyze(R, L, np.zeros_like(R))[0].status == "absent"
    tiny = np.zeros_like(R); tiny[100, 100] = 1
    assert analyze(R, L, tiny)[0].status == "indeterminate"

def test_nested_image_discovery(tmp_path):
    (tmp_path / "Normal").mkdir()
    (tmp_path / "Pneumonia_Bacterial").mkdir()
    (tmp_path / "Normal" / "a.jpg").write_bytes(b"")
    (tmp_path / "Pneumonia_Bacterial" / "b.png").write_bytes(b"")
    (tmp_path / "metadata.csv").write_text("Filename,Age,Gender,Class\n")
    found = [p.relative_to(tmp_path).as_posix() for p in iter_image_files(tmp_path)]
    assert found == ["Normal/a.jpg", "Pneumonia_Bacterial/b.png"]
