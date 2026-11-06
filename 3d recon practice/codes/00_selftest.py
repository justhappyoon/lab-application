"""사진 없이 코드가 제대로 도는지 확인하는 자체 점검.
config.py의 두 물체(상자·캔) 각각에 대해 가짜 복원 메시를 만들어 파이프라인에 넣는다."""
import numpy as np
import trimesh
from config import OBJECTS
from lib import (make_reference, load_clean, scale_and_align, dim_errors, chamfer,
                 sample_grasps, validity_on_ref, add_noise)

rng = np.random.default_rng(0)
for obj, (kind, dims) in OBJECTS.items():
    print(f"===== {obj} {dims} =====")
    ref = make_reference(kind, dims)

    # (1) 기준 형상 자체를 '복원'으로 넣으면 유효 비율은 1.0이어야 한다
    p, q = sample_grasps(ref, n=5000)
    v = validity_on_ref(p, q, ref)
    print(f"[1] 기준 형상 자신: 후보 {len(p)}쌍, 유효 비율 {v:.3f}  (기대: 1.000)")

    # (2) 가짜 복원: 조밀화 + 노이즈 + 바닥면 제거 + 임의 회전·이동·스케일(단위가 다른 척)
    fake = ref.subdivide_to_size(3.0)
    fake = add_noise(fake, 0.3)
    keep = fake.face_normals[:, 2] > -0.9   # 아래를 향한 면(바닥)만 제거
    fake = trimesh.Trimesh(fake.vertices, fake.faces[keep], process=True)
    junk = trimesh.creation.icosphere(radius=2.0)
    junk.apply_translation([300, 0, 0])
    fake = trimesh.util.concatenate([fake, junk])   # 떠다니는 쓰레기 조각
    fake.apply_transform(trimesh.transformations.random_rotation_matrix(rng.random(3)))
    fake.apply_translation([30, -10, 5])
    fake.apply_scale(0.0042)  # 단위가 다른 척

    fake.export(f"selftest_fake_{obj}.obj")
    r = load_clean(f"selftest_fake_{obj}.obj")
    r, s = scale_and_align(r, ref)
    de = np.round(dim_errors(r, ref), 2)
    print(f"[2] 스케일 계수 {s:.1f}, 치수 오차(%) x,y,z = {de}")
    print(f"    Chamfer(복원->기준, 기준->복원, 평균) = {np.round(chamfer(r, ref), 3)} mm")

    # (3) 노이즈를 키우면 유효 비율이 떨어지는지
    for sg in (0.0, 0.5, 1.0, 2.0):
        m = add_noise(r, sg) if sg > 0 else r
        p, q = sample_grasps(m, n=5000)
        v = validity_on_ref(p, q, ref)
        print(f"[3] sigma={sg:>3} mm -> 후보 {len(p):4d}쌍, 유효 비율 {v:.3f}")
