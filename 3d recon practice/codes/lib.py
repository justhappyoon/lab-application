"""공용 함수. 단위는 모두 mm."""
from pathlib import Path
import numpy as np
import trimesh

MU = 0.5   # 마찰계수
ALPHA = np.arctan(MU)   # 마찰 원뿔 반각(rad) = arctan(0.5) = 약 26.6도
W_MIN, W_MAX = 5.0, 100.0   # 평행 그리퍼가 벌릴 수 있는 접촉점 간 거리 범위
FACES = 20000   # 모든 복원 메시를 같은 삼각형 수로 단순화


def make_reference(kind, dims):
    """기준 치수(YCB 논문 값)로 기준 형상을 만든다. 바닥면이 z=0, x·y는 중심이 원점.
    kind="box": dims=(x, y, z)   kind="can": dims=(지름, 높이)"""
    if kind == "box":
        m = trimesh.creation.box(extents=dims)
    else:
        m = trimesh.creation.cylinder(radius=dims[0] / 2, height=dims[1], sections=128)
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def find_mesh_file(path):
    """path가 파일이면 그대로 쓴다. 폴더이면 그 안의 가장 큰 .obj를 쓴다.
    (PhotoCatch는 'box_full.obj' 같은 이름의 *폴더*에
    model.obj, model.mtl, texgen_*.png를 내보낸다.)"""
    p = Path(path)
    if p.is_file():
        return str(p)
    found = sorted(p.rglob("*.obj"), key=lambda f: f.stat().st_size) if p.is_dir() else []
    if not found:
        raise FileNotFoundError(f"{path}: 파일도, .obj가 든 폴더도 아닙니다.")
    return str(found[-1])


def load_clean(path, faces=FACES, keep_frac=0.02):
    """메시를 읽고(재질·텍스처는 무시), 떠다니는 작은 쓰레기만 버리고, 삼각형 수를 맞춘다.
    사진측량 메시는 한 덩어리가 아니라 여러 조각으로 갈라져 나올 수 있다.
    그래서 '가장 큰 조각만' 남기지 않고, 가장 큰 조각 넓이의 keep_frac(2%) 이상인
    조각은 모두 합쳐서 쓴다."""
    m = trimesh.load(find_mesh_file(path), force="mesh", skip_materials=True)
    parts = m.split(only_watertight=False)
    amax = max(p.area for p in parts)
    parts = [p for p in parts if p.area >= keep_frac * amax]
    m = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return m


def orient_outward(m):
    """법선이 모두 바깥을 향하게 맞춘다. 상자·캔은 볼록 형상이므로
    '중심에서 멀어지는 방향'을 바깥으로 정의한다."""
    c = m.bounds.mean(axis=0)
    flip = np.einsum("ij,ij->i", m.face_normals, m.triangles_center - c) < 0
    m.faces[flip] = m.faces[flip][:, ::-1]
    return m


def pca_frame(pts):
    """점들의 주성분 축(SVD). 반환: 중심 c, 회전 R(행 = 주축)."""
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    if np.linalg.det(Vt) < 0:
        Vt[2] *= -1
    return c, Vt


def robust_extents(pts):
    """x·y·z 방향 크기: 극단값에 덜 민감하도록 1~99 백분위 사이 폭을 쓴다."""
    return np.percentile(pts, 99, axis=0) - np.percentile(pts, 1, axis=0)


def scale_and_align(recon, ref, n=5000, seed=0):
    """(1) 가장 긴 치수를 기준 치수에 맞춰 스케일 (2) PCA 축으로 초기 정렬
    (3) ICP 미세 정렬.
    상자·캔은 대칭이라 축 방향이 뒤집힐 수 있으므로 4가지 후보를 모두 시도해
    Chamfer 거리가 가장 작은 것을 고른다."""
    r = recon.copy()
    pts, _ = trimesh.sample.sample_surface(r, 50000, seed=seed)
    c, R = pca_frame(pts)
    s = robust_extents(trimesh.sample.sample_surface(ref, 50000, seed=seed)[0]).max() \
        / robust_extents((pts - c) @ R.T).max()
    r.apply_scale(s)
    pts, _ = trimesh.sample.sample_surface(r, 50000, seed=seed)
    c, R = pca_frame(pts)
    pr, _ = trimesh.sample.sample_surface(ref, 50000, seed=seed)
    cf, Rf = pca_frame(pr)
    best = None
    for flip in ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]):
        Rot = Rf.T @ np.diag(flip) @ R                         # 복원 주축 -> 기준 주축
        T0 = np.eye(4)
        T0[:3, :3] = Rot
        T0[:3, 3] = cf - Rot @ c
        T, _, _ = trimesh.registration.icp(pts[:n], ref, initial=T0, max_iterations=50,
                                           reflection=False, scale=False)
        cand = r.copy()
        cand.apply_transform(T)
        cost = chamfer(cand, ref, n=3000, seed=seed)[2]
        if best is None or cost < best[0]:
            best = (cost, cand)
    return orient_outward(best[1]), s


def dim_errors(r, ref, seed=0):
    """정렬된 복원 메시의 x·y·z 크기 오차(%). 기준 형상에도 같은 방법을 적용해 비교.
    스케일 기준으로 쓴 가장 긴 치수는 0에 가깝다."""
    er = robust_extents(trimesh.sample.sample_surface(r, 50000, seed=seed)[0])
    ef = robust_extents(trimesh.sample.sample_surface(ref, 50000, seed=seed)[0])
    return (er - ef) / ef * 100


def chamfer(r, ref, n=20000, seed=0):
    """Chamfer 거리(mm): (복원->기준, 기준->복원, 평균). 복원되지 않는 바닥면은 제외."""
    pr, _ = trimesh.sample.sample_surface(r, n, seed=seed)
    pf, _ = trimesh.sample.sample_surface(ref, n, seed=seed)
    pf = pf[pf[:, 2] > 2.0]
    d_rf = trimesh.proximity.closest_point(ref, pr)[1]       # 점 -> 기준 표면
    d_fr = trimesh.proximity.closest_point(r, pf)[1]         # 점 -> 복원 표면
    return d_rf.mean(), d_fr.mean(), 0.5 * (d_rf.mean() + d_fr.mean())


def cone_dirs(axis, alpha, rng):
    """axis(N,3) 둘레 반각 alpha 안의 무작위 단위벡터."""
    k = len(axis)
    cos_t = rng.uniform(np.cos(alpha), 1.0, k)
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = rng.uniform(0, 2 * np.pi, k)
    helper = np.where(np.abs(axis[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(axis, e1)
    return (axis * cos_t[:, None]
            + sin_t[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))


def sample_grasps(mesh, n=5000, seed=0):
    """antipodal 파지 후보(p, q) 샘플링.
    표면점 p에서 안쪽 법선(-n_p) 둘레 마찰 원뿔 안의 방향 d로 광선을 쏴 반대편 q를 찾고,
    q에서도 d가 마찰 원뿔 안이면(angle(d, n_q) <= alpha) 유효한 antipodal 쌍으로 본다."""
    rng = np.random.default_rng(seed)
    p, fid = trimesh.sample.sample_surface(mesh, n, seed=seed)
    n_p = mesh.face_normals[fid]
    d = cone_dirs(-n_p, ALPHA, rng)
    hits, ri, ti = mesh.ray.intersects_location(p - 0.05 * n_p, d, multiple_hits=True)
    dist = np.linalg.norm(hits - p[ri], axis=1)
    keep = dist > W_MIN                                   # 자기 자신 근처 교점 제외
    hits, ri, ti, dist = hits[keep], ri[keep], ti[keep], dist[keep]
    order = np.lexsort((dist, ri))                        # 광선별로 가까운 교점 순
    first = np.r_[True, ri[order][1:] != ri[order][:-1]]
    sel = order[first]
    q, ri, ti, w = hits[sel], ri[sel], ti[sel], dist[sel]
    ok = (w < W_MAX) & (np.einsum("ij,ij->i", d[ri], mesh.face_normals[ti]) > np.cos(ALPHA))
    return p[ri][ok], q[ok]


def validity_on_ref(p, q, ref):
    """복원 메시에서 뽑은 파지(p, q)를 기준 형상 표면으로 투영해 antipodal 조건을 다시 검사.
    반환: 기준 형상에서도 유효한 비율(0~1)."""
    if len(p) == 0:
        return float("nan")
    pr, _, tp = trimesh.proximity.closest_point(ref, p)
    qr, _, tq = trimesh.proximity.closest_point(ref, q)
    u = qr - pr
    w = np.linalg.norm(u, axis=1)
    u = u / np.maximum(w, 1e-9)[:, None]
    c = np.cos(ALPHA)
    ok = ((w > W_MIN) & (w < W_MAX)
          & (np.einsum("ij,ij->i", u, -ref.face_normals[tp]) > c)
          & (np.einsum("ij,ij->i", u, ref.face_normals[tq]) > c))
    return ok.mean()


def add_noise(mesh, sigma, seed=0):
    """각 꼭짓점을 법선 방향으로 N(0, sigma^2) mm만큼 이동."""
    m = mesh.copy()
    rng = np.random.default_rng(seed)
    m.vertices = m.vertices + m.vertex_normals * rng.normal(0, sigma, (len(m.vertices), 1))
    return m


def add_smooth(mesh, rms, wavelength=40.0, seed=0):
    """(선택 실험용) 파장 약 40mm의 저주파로 완만하게 휘는 변형. 변위의 RMS가 rms(mm)."""
    m = mesh.copy()
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(4, 3))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    ph = rng.uniform(0, 2 * np.pi, 4)
    f = sum(np.sin(2 * np.pi / wavelength * (m.vertices @ k[i]) + ph[i]) for i in range(4))
    m.vertices = m.vertices + m.vertex_normals * (rms * f / f.std())[:, None]
    return m
