"""복원 메시 정리 -> 스케일 -> 정렬 -> 치수 오차·Chamfer 계산."""
import csv, os
from config import OBJECTS, CONDITIONS, RECON_DIR
from lib import make_reference, load_clean, scale_and_align, dim_errors, chamfer


def recon_path(name):
    """recon/<name>.obj가 파일이든, PhotoCatch가 만든 폴더든, 같은 이름의 폴더든 찾는다."""
    for c in (f"{RECON_DIR}/{name}.obj", f"{RECON_DIR}/{name}"):
        if os.path.exists(c):
            return c
    raise SystemExit(f"복원 메시를 찾지 못했습니다: {RECON_DIR}/{name}.obj "
                     "(또는 같은 이름의 폴더)")


os.makedirs("work", exist_ok=True)
os.makedirs("results", exist_ok=True)
rows = [["name", "scale", "err_x_%", "err_y_%", "err_z_%",
         "chamfer_recon_to_ref_mm", "chamfer_ref_to_recon_mm", "chamfer_mean_mm"]]
for obj, (kind, dims) in OBJECTS.items():
    ref = make_reference(kind, dims)
    for cond in CONDITIONS:
        name = f"{obj}_{cond}"
        r = load_clean(recon_path(name))
        r, s = scale_and_align(r, ref)
        r.export(f"work/{name}_aligned.ply")
        ex = dim_errors(r, ref)
        a, b, c = chamfer(r, ref)
        rows.append([name, f"{s:.4f}", *[f"{v:.2f}" for v in ex],
                     f"{a:.3f}", f"{b:.3f}", f"{c:.3f}"])
        print(rows[-1])

with open("results/geometry.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("저장: results/geometry.csv  (스케일 기준인 가장 긴 치수는 오차가 0에 가깝습니다)")
