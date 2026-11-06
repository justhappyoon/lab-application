# 치수 출처: YCB 논문(Calli et al., arXiv 1502.03143)의 물체 표(Table II).
OBJECTS = {
    "box": ("box", (60.0, 160.0, 230.0)),  # cracker box: (두께, 폭, 높이)
    "can": ("can", (66.0, 101.0)),  # tomato soup can: (지름, 높이)
}
CONDITIONS = ["full", "half"]   # full = 원본 해상도, half = 가로·세로 절반
RECON_DIR = "recon"  # recon/box_full.obj, recon/box_half.obj, ...
SIGMAS = [0.0, 0.5, 1.0, 2.0]  # 인공 노이즈 표준편차(mm)
REPEATS = 3   # 노이즈 시드 반복 횟수
