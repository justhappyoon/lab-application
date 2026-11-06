"""YCB(BigBIRD) 사진 폴더에서 사진을 고르고, 배경을 지우고, 절반 해상도 사본도 만든다.

실제 YCB(BigBIRD) 압축 파일(003_cracker_box, 005_tomato_soup_can)로 실행해 확인했습니다.
(사진: N<카메라 1~5>_<턴테이블 각도>.jpg / 마스크: .pbm 또는 .png)

1) 카메라 고르기 (높이가 뚜렷하게 다른 2대를 눈으로 고릅니다):
     python 00b_make_inputs.py --obj box --src ycb/003_cracker_box --list
   -> photos/_cameras_box.jpg 를 열어 봅니다.
2) 사진 만들기:
     python 00b_make_inputs.py --obj box --src ycb/003_cracker_box --cams 2 4
   -> photos/box_full/ (원본 해상도, 배경 제거)
      photos/box_half/ (가로·세로 절반)
      photos/_check_box.jpg (결과 확인용 미리보기. 꼭 열어 보세요)
"""
import argparse
import re
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# --obj 이름과 --src 폴더 이름이 짝인지 점검
EXPECT = {"box": "cracker_box", "can": "soup_can"}
IMG_RE = re.compile(r"^N(\d)_(\d+)\.(jpg|jpeg)$", re.I)
MASK_RE = re.compile(r"^(N\d_\d+)(?:_mask)?\.(pbm|png)$", re.I)


def scan(src):
    """src 아래의 사진과 마스크를 모두 찾는다."""
    imgs, masks = {}, {}
    for p in sorted(Path(src).rglob("*")):
        if not p.is_file():
            continue
        m = IMG_RE.match(p.name)
        if m:
            imgs[(int(m.group(1)), int(m.group(2)))] = p
            continue
        m = MASK_RE.match(p.name)
        if m:
            masks[m.group(1).upper()] = p
    return imgs, masks


def object_mask(path, size, invert, grow):
    """물체=255, 배경=0인 마스크.
    물체가 화면에서 더 작은 쪽이라고 가정해 극성을 자동 판단한다."""
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    m = m.point(lambda v: 255 if v > 127 else 0)
    frac = m.histogram()[255] / (size[0] * size[1])
    if frac > 0.5:
        m = m.point(lambda v: 255 - v)
        frac = 1 - frac
    if invert:
        m = m.point(lambda v: 255 - v)
        frac = 1 - frac
    if grow > 0:   # 가장자리가 잘리지 않게 약간 키움
        m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
    return m, frac


def half_exif(im0):
    """절반 크기 사진용 EXIF. 사진 크기와 초점면 해상도(px/인치)를 절반에 맞춘다.
    (그대로 두면 원본 크기 정보가 남아 도구가 초점거리를 2배로 잘못 읽을 수 있다.)"""
    ex = im0.getexif()
    sub = ex.get_ifd(0x8769)
    for tag in (0xA002, 0xA003):  # ExifImageWidth, ExifImageHeight
        if tag in sub:
            sub[tag] = int(sub[tag]) // 2
    for tag in (0xA20E, 0xA20F):  # FocalPlaneX/YResolution
        if tag in sub:
            sub[tag] = float(sub[tag]) / 2
    return ex.tobytes()


def contact_sheet(items, h=240, cols=4):
    thumbs = []
    for label, im in items:
        im = im.convert("RGB")
        t = im.resize((max(1, int(im.width * h / im.height)), h))
        ImageDraw.Draw(t).text((4, 4), label, fill=(255, 0, 0))
        thumbs.append(t)
    cw = max(t.width for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, h * rows), (255, 255, 255))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * cw, (i // cols) * h))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True, help="config.py의 물체 이름 (box 또는 can)")
    ap.add_argument("--src", required=True, help="압축을 푼 YCB 폴더")
    ap.add_argument("--cams", type=int, nargs="+", help="사용할 카메라 번호들 (예: 2 4)")
    ap.add_argument("--every", type=int, default=18, help="각도 간격(도). 기본 18 -> 20곳")
    ap.add_argument("--list", action="store_true", help="카메라별 미리보기만 만들고 끝")
    ap.add_argument("--no-mask", action="store_true", help="배경을 안 지움(권하지 않음)")
    ap.add_argument("--invert", action="store_true", help="마스크가 반대로 적용될 때")
    ap.add_argument("--grow", type=int, default=2, help="마스크를 키우는 픽셀 수(기본 2)")
    ap.add_argument("--gray", type=int, default=127, help="배경 회색 값 0~255(기본 127)")
    ap.add_argument("--out", default="photos")
    a = ap.parse_args()

    expect = EXPECT.get(a.obj)
    if expect and expect not in str(a.src):
        sys.exit(f"--obj {a.obj} 는 이름에 '{expect}'가 들어 있는 YCB 폴더와 짝입니다. "
                 f"--src 를 확인하세요 (지금: {a.src}).")
    imgs, masks = scan(a.src)
    if not imgs:
        sys.exit("N<카메라>_<각도>.jpg 사진을 찾지 못했습니다. "
                 "--src 경로와 압축 해제 위치를 확인하세요.")
    cams = sorted({c for c, _ in imgs})
    angles = sorted({g for _, g in imgs})
    print(f"찾은 사진 {len(imgs)}장 / 카메라 {cams} / 각도 {len(angles)}개 "
          f"({angles[0]}~{angles[-1]}도) / 마스크 {len(masks)}개")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.list:
        items = [(f"N{c}", Image.open(imgs[(c, angles[0])]))
                 for c in cams if (c, angles[0]) in imgs]
        path = out / f"_cameras_{a.obj}.jpg"
        contact_sheet(items, h=300, cols=len(items)).save(path)
        print(f"저장: {path}")
        print("-> 보는 높이가 뚜렷하게 다른 카메라 2대를 고르세요.")
        return

    if not a.cams:
        sys.exit("--cams 로 카메라 번호를 주세요. 먼저 --list 로 확인하세요.")
    pick = [g for g in angles if g % a.every == 0]
    chosen = [(c, g) for c in a.cams for g in pick if (c, g) in imgs]
    absent = [(c, g) for c in a.cams for g in pick if (c, g) not in imgs]
    print(f"각도 {len(pick)}곳 x 카메라 {len(a.cams)}대 -> 사진 {len(chosen)}장 선택")
    if absent:
        print(f"경고: 없는 사진 {len(absent)}장 (예: {absent[:3]})")
    if not chosen:
        sys.exit("선택된 사진이 없습니다. --cams 번호를 확인하세요.")

    if not a.no_mask:
        lost = [f"N{c}_{g}" for c, g in chosen if f"N{c}_{g}" not in masks]
        if lost:
            sys.exit(f"마스크를 못 찾은 사진 {len(lost)}장 (예: {lost[:3]}). "
                     "이 물체는 마스크가 없을 수 있습니다. 부록 B를 보세요.")

    d_full, d_half = out / f"{a.obj}_full", out / f"{a.obj}_half"
    d_full.mkdir(parents=True, exist_ok=True)
    d_half.mkdir(parents=True, exist_ok=True)
    samples, fracs = [], []
    for k, (c, g) in enumerate(chosen):
        name = f"N{c}_{g}"
        im0 = Image.open(imgs[(c, g)])
        exif = im0.info.get("exif")
        im = im0.convert("RGB")
        if not a.no_mask:
            m, frac = object_mask(masks[name], im.size, a.invert, a.grow)
            fracs.append(frac)
            bg = Image.new("RGB", im.size, (a.gray,) * 3)
            im = Image.composite(im, bg, m)
        kw = dict(quality=95, subsampling=0)
        if exif:
            kw["exif"] = exif
        im.save(d_full / f"{name}.jpg", **kw)
        half = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        kh = dict(kw)
        if exif:
            kh["exif"] = half_exif(im0)
        half.save(d_half / f"{name}.jpg", **kh)
        if k % max(1, len(chosen) // 8) == 0 and len(samples) < 8:
            samples.append((name, im))
    if fracs:
        print(f"물체로 판정한 면적 비율: 평균 {sum(fracs) / len(fracs):.1%} "
              f"(최소 {min(fracs):.1%}, 최대 {max(fracs):.1%})")
        if min(fracs) < 0.003 or max(fracs) > 0.6:
            print("경고: 비율이 이상합니다. 미리보기를 보고 --invert 를 시도하세요.")
    check = out / f"_check_{a.obj}.jpg"
    contact_sheet(samples).save(check)
    W, H = Image.open(next(d_full.glob("*.jpg"))).size
    print(f"저장: {d_full} ({len(chosen)}장, {W}x{H})")
    print(f"저장: {d_half} ({len(chosen)}장, {W // 2}x{H // 2})")
    print(f"미리보기: {check}  -> 물체만 남고 배경이 회색인지 눈으로 확인하세요.")


if __name__ == "__main__":
    main()
