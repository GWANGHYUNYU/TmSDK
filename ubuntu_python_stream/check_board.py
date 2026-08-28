#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""체커보드 코너 검출 확인 — 현장에서 촬영 직후 바로 돌려 보는 도구.

촬영만 하고 돌아오면 실패했을 때 다시 가야 합니다. 한 장 찍을 때마다,
적어도 자세를 바꿀 때마다 이 스크립트로 검출되는지 확인하십시오.

    # 열화상 원본 (슬롯 평균 자동)
    python3 check_board.py 192_168_0_151_20260812_140000.y16raw

    # RGB 사진
    python3 check_board.py board_rgb.png

    # 영상도 됩니다 (가장 선명한 프레임을 자동으로 골라 씁니다)
    python3 check_board.py pose01.mp4

    # 폴더 전체 일괄 확인
    python3 check_board.py calib/

    # 세로로 세워 잡았다면
    python3 check_board.py board.png --pattern 4x7

기본 패턴은 7x4 (가로로 눕혀 잡은 경우). 판을 세우면 4x7.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
RAW_EXT = {".y16raw"}
# 녹화기가 같이 만드는 부속 파일 — 짝 맞추기에서 제외해야 한다
SKIP_EXT = {".y16meta", ".csv", ".json", ".txt", ".log"}
ALL_EXT = IMG_EXT | VID_EXT | RAW_EXT


# ── 입력 로딩 ─────────────────────────────────────────────────────
# ★ 초점거리 — 사양서의 42°(f=208.4 px)는 실측과 맞지 않는 것으로 확인됐다.
#   2026-08-28 줄자 실측(보드~렌즈 400 mm·450 mm 두 지점)으로 f=147.4 px
#   (화각 57.2° x 44.4°) 확정. 카메라 상수는 1000/147.4 = 6.78 mm/px/m.
F_PX = 147.4
SQ_TARGET = 15.0        # 이보다 작으면 더 가까이 대라고 경고 (권장 17 px)

NFRAMES = 12          # 평균낼 프레임 수 (--frames 로 변경)
PROBE = 15            # 영상에서 훑어볼 후보 프레임 수



def short(s, w=22):
    """긴 파일명을 꼬리부터 남긴다.

    현장 파일명은 192_168_0_151_20260828_154701 처럼 앞부분이 공통이라
    앞에서 자르면 서로 구분이 안 된다. 꼬리를 남겨야 카메라 번호와 시각이
    보인다.
    """
    s = str(s)
    return s if len(s) <= w else "…" + s[-(w-1):]


def sharpness(g):
    """라플라시안 분산 — 클수록 선명. 흔들린 프레임을 걸러내는 데 쓴다."""
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def load_thermal(path):
    """y16raw → 후보 목록 [(섭씨영상, 8bit)]. FFC 프레임은 제외.

    --frames 1 이면 '가장 선명한 한 장'을 고른다. 손으로 들고 찍어 흔들린
    프레임이 섞여 있을 때 이 편이 안전하다. 2 이상이면 균등 간격 평균.
    """
    import inspect
    from read_y16 import (load_meta, load_raw, temperature_converter,
                          frame_mean_celsius, detect_ffc, bad_frame_mask)
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)

    # FFC 프레임 제외는 '있으면 좋은' 정제 단계다. 체커보드 검출에는 필수가
    # 아니므로, read_y16.py 버전이 달라 실패해도 전체 프레임으로 진행한다.
    #
    # 구버전 read_y16 의 detect_ffc 는 (tmean, threshold) 시그니처라
    # meta 를 threshold 자리로 받아 max(float, dict) 비교에서 터진다.
    idx = None
    try:
        tm = frame_mean_celsius(arr, conv)
        if "meta" in inspect.signature(detect_ffc).parameters:
            ffc = detect_ffc(tm, meta)
        else:
            ffc = detect_ffc(tm)
        keep = bad_frame_mask(arr.shape[0], ffc, 3)
        idx = np.where(keep)[0]
    except Exception as e:
        print(f"    ⚠ FFC 프레임 판별을 건너뜁니다 ({type(e).__name__}: {e})",
              file=sys.stderr)
        print(f"      read_y16.py 가 구버전일 수 있습니다. 검출에는 영향 없습니다.",
              file=sys.stderr)
    if idx is None or len(idx) == 0:
        idx = np.arange(arr.shape[0])

    if NFRAMES <= 1:
        # ★ 가장 선명한 한 장만 쓰면 안 된다.
        #   현장 데이터에서 검출에 성공한 프레임이 선명도 순 2·9·10번째였다.
        #   선명한 것과 '보드가 제대로 보이는 것'은 다르다. 영상 경로와 똑같이
        #   여러 장을 선명한 순으로 모두 넘기고 검출기가 고르게 한다.
        probe = idx[np.linspace(0, len(idx)-1, min(PROBE, len(idx))).astype(int)]
        out = []
        for i in probe:
            c = conv(np.asarray(arr[i])).astype(np.float64)
            out.append((c, to8(c)))
        out.sort(key=lambda t: sharpness(t[1]), reverse=True)
        return out

    k = min(NFRAMES, len(idx))
    sel = idx[np.linspace(0, len(idx)-1, k).astype(int)]
    c = conv(np.asarray(arr[sel])).astype(np.float64).mean(axis=0)
    return [(c, to8(c))]


def load_video(path):
    """영상 → 균등 간격 후보 프레임들. 선명한 순으로 정렬해서 돌려준다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if n <= 0:                                   # 프레임 수를 못 읽는 컨테이너
        while len(out) < PROBE:
            ok, fr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    else:
        for i in np.linspace(0, n-1, min(PROBE, n)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not out:
        raise RuntimeError(f"영상에서 프레임을 못 읽었습니다: {path}")
    out.sort(key=sharpness, reverse=True)        # 선명한 것부터
    return [(None, g) for g in out]


def load_candidates(path):
    """→ [(섭씨영상 또는 None, 8bit 그레이)] · 검출을 시도할 후보 목록"""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        return load_thermal(path)
    if ext in VID_EXT:
        return load_video(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"읽을 수 없습니다: {path}")
    return [(None, img)]


def load_any(path):
    """첫 후보만 (호환용)."""
    c, g = load_candidates(path)[0]
    return c, g, os.path.splitext(path)[1].lower() in RAW_EXT


def to8(img):
    lo, hi = np.percentile(img, [1, 99])
    if hi - lo < 1e-6:
        return np.zeros(img.shape, np.uint8)
    return np.clip((img-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)


# ── 검출 ──────────────────────────────────────────────────────────
def detect(gray, patterns, upscale=1):
    """SB → 고전 순, 여러 명암 변형으로 시도.

    변형에 CLAHE(국소 히스토그램 평활화)를 넣는다. 같은 녹화라도 .avi 는
    프레임마다 자동이득이 걸려 약한 대비가 늘어나 검출되는데, y16raw 는
    전역 백분위로만 정규화해 그대로 묻히는 일이 있었다. CLAHE 가 그 역할을
    대신한다.
    """
    im0 = gray
    if upscale > 1:
        im0 = cv2.resize(gray, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC)
    try:
        eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(im0)
    except cv2.error:
        eq = im0
    variants = (("정", im0), ("반전", 255-im0),
                ("CLAHE", eq), ("CLAHE반전", 255-eq))
    has_sb = hasattr(cv2, "findChessboardCornersSB")

    for pat in patterns:
        if has_sb:
            for pol, im in variants:
                try:
                    ok, c = cv2.findChessboardCornersSB(
                        im, pat,
                        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
                except cv2.error:
                    ok = False
                if ok:
                    return True, c/upscale, pat, f"SB · {pol}"
        for pol, im in variants:
            ok, c = cv2.findChessboardCorners(
                im, pat,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                c = cv2.cornerSubPix(
                    im, c, (3, 3), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
                return True, c/upscale, pat, f"classic · {pol}"
    return False, None, None, "-"


def detect_file(path, patterns, upscale=1):
    """파일(이미지·영상·y16raw) 하나에서 최선의 검출을 찾는다.

    영상이면 선명한 프레임부터 훑어 처음 성공한 것을 쓴다. 흔들린 프레임이
    섞여 있어도 자동으로 쓸 만한 프레임이 골라진다.

    → (ok, corners, pat, how, cels, gray, 시도한 후보 수)
    """
    cands = load_candidates(path)
    for k, (cels, gray) in enumerate(cands, 1):
        ok, c, pat, how = detect(gray, patterns, upscale)
        if ok:
            if len(cands) > 1:
                how = f"{how} · {k}/{len(cands)}번째 프레임"
            return True, c, pat, how, cels, gray, k

    # 실패 시에는 '가장 대비가 좋았던' 후보를 진단용으로 돌려준다.
    # 첫 후보(=가장 선명한 장)가 보드를 못 담고 있을 수 있어, 그것만 보고
    # "대비 없음" 이라고 판단하면 오진이 된다.
    best_i = 0
    if len(cands) > 1:
        loc = [scene_stats(c)[1] for c, _ in cands]
        if any(v is not None for v in loc):
            best_i = int(np.argmax([-1 if v is None else v for v in loc]))
    return False, None, None, "-", cands[best_i][0], cands[best_i][1], len(cands)


def inplane_deg(corners, pat):
    """판을 화면 안에서 얼마나 돌려 들었는가 (긴 변과 수평선의 각, 0~90)."""
    g = corners.reshape(pat[1], pat[0], 2)
    v = g[:, -1].mean(0) - g[:, 0].mean(0)
    a = abs(float(np.degrees(np.arctan2(v[1], v[0]))))
    return min(a, 180-a)


def square_px(corners, pat):
    """이웃 코너 간 거리의 중앙값 = 칸 크기(화소)."""
    c = corners.reshape(pat[1], pat[0], 2)
    d = []
    if pat[0] > 1:
        d.append(np.linalg.norm(np.diff(c, axis=1), axis=2).ravel())
    if pat[1] > 1:
        d.append(np.linalg.norm(np.diff(c, axis=0), axis=2).ravel())
    return float(np.median(np.concatenate(d)))


def scene_stats(cels):
    """코너 검출 없이도 구할 수 있는 진단값.

    대비(contrast_c)는 코너를 찾은 뒤에만 계산되는데, 정작 필요한 것은
    검출이 실패했을 때다. 그래서 코너와 무관한 두 값을 따로 낸다.

      · 범위   화면 전체 온도 폭 (p1~p99). 1℃ 미만이면 판이 안 데워진 것
      · 국소대비 15x15 창 표준편차의 상위값. 강한 패턴이 화면에 있는지.
                판이 데워져 보드가 보이면 1℃ 이상 나온다.
    """
    if cels is None:
        return None, None
    lo, hi = np.percentile(cels, [1, 99])
    f = cels.astype(np.float32)
    k = 15
    mean = cv2.blur(f, (k, k))
    sq = cv2.blur(f*f, (k, k))
    std = np.sqrt(np.maximum(sq - mean*mean, 0))
    return float(hi-lo), float(np.percentile(std, 99))


def contrast_c(cels, corners, pat):
    """검정 칸과 금속 칸의 겉보기 온도차 추정."""
    if cels is None:
        return None
    c = corners.reshape(pat[1], pat[0], 2)
    vals = {0: [], 1: []}
    for j in range(pat[1]-1):
        for i in range(pat[0]-1):
            cx, cy = c[j:j+2, i:i+2].reshape(4, 2).mean(axis=0)
            x, y = int(round(cx)), int(round(cy))
            if 1 <= x < cels.shape[1]-1 and 1 <= y < cels.shape[0]-1:
                vals[(i+j) % 2].append(cels[y-1:y+2, x-1:x+2].mean())
    if not vals[0] or not vals[1]:
        return None
    return abs(float(np.mean(vals[0]) - np.mean(vals[1])))


# ── 실행 ──────────────────────────────────────────────────────────
def gather(target):
    if os.path.isfile(target):
        return [target]
    out = []
    for f in sorted(os.listdir(target)):
        if os.path.splitext(f)[1].lower() in ALL_EXT:
            out.append(os.path.join(target, f))
    return out


def main():
    ap = argparse.ArgumentParser(description="체커보드 코너 검출 확인")
    ap.add_argument("target", help="파일 또는 폴더")
    ap.add_argument("--pattern", default="7x4",
                    help="내부 코너 수. 기본 7x4 (판을 눕힌 경우). 세우면 4x7")
    ap.add_argument("--both", action="store_true",
                    help="7x4 와 4x7 을 모두 시도")
    ap.add_argument("--upscale", type=int, default=1,
                    help="검출 전 확대 배율 (열화상에서 실패하면 3 을 시도)")
    ap.add_argument("--save", metavar="DIR", help="검출 결과 오버레이 저장 폴더")
    ap.add_argument("--dump", metavar="DIR",
                    help="검출 성공·실패와 무관하게 실제로 본 프레임을 PNG 로 저장. "
                         "실패 원인을 눈으로 확인할 때 쓰십시오")
    ap.add_argument("--frames", type=int, default=12,
                    help="열화상 평균 프레임 수. 손으로 들고 찍었으면 3 (기본 12)")
    ap.add_argument("--min-dt", type=float, default=1.0,
                    help="열화상 최소 대비(℃). 미만이면 경고 (기본 1.0)")
    args = ap.parse_args()

    global NFRAMES
    NFRAMES = args.frames

    w, h = (int(v) for v in args.pattern.lower().split("x"))
    pats = [(w, h)] + ([(h, w)] if args.both else [])

    files = gather(args.target)
    if not files:
        print("대상 파일이 없습니다.")
        return 1
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)

    print("=" * 82)
    print(f"체커보드 검출 확인 · 패턴 {'/'.join(f'{a}x{b}' for a, b in pats)}"
          f" · OpenCV {cv2.__version__}")
    if not hasattr(cv2, "findChessboardCornersSB"):
        print("  ⚠ findChessboardCornersSB 없음 (OpenCV 4.0 미만) — 검출률이 떨어집니다")
    print("=" * 82)
    print(f"  {'파일':<40}{'검출':>6}{'칸(px)':>9}{'대비':>9}   방법")

    nok = 0
    warn = []
    nofail = []
    for p in files:
        name = os.path.basename(p)
        try:
            ok, corners, pat, how, cels, gray, _ = detect_file(
                p, pats, args.upscale)
        except Exception as e:
            print(f"  {name[:38]:<40}{'오류':>6}   {e}")
            continue

        if args.dump:
            stem = os.path.splitext(name)[0]
            try:
                for k, (cc, gg) in enumerate(load_candidates(p)[:4], 1):
                    big = cv2.resize(gg, None, fx=5, fy=5,
                                     interpolation=cv2.INTER_NEAREST)                         if gg.shape[1] < 400 else gg
                    cv2.imwrite(os.path.join(
                        args.dump, f"{stem}_f{k}.png"), big)
                    if cc is not None and k == 1:
                        # 온도 범위를 파일명에 남겨 대비를 바로 보게
                        os.rename(
                            os.path.join(args.dump, f"{stem}_f{k}.png"),
                            os.path.join(args.dump,
                                         f"{stem}_f{k}_{cc.min():.1f}-"
                                         f"{cc.max():.1f}C.png"))
            except Exception as e:
                print(f"    ⚠ dump 실패: {e}")

        span, loc = scene_stats(cels)
        if not ok:
            diag = ""
            if span is not None:
                diag = f"프레임 온도 범위 {span:.1f}℃"
            else:
                diag = "영상이라 온도 진단 불가 — .y16raw 를 쓰십시오"
            print(f"  {name[:38]:<40}{'실패':>6}   {diag}")
            nofail.append((name, span, loc))
            continue

        nok += 1
        sq = square_px(corners, pat)
        dt = contrast_c(cels, corners, pat)
        dts = f"{dt:.2f}℃" if dt is not None else "-"
        print(f"  {name[:38]:<40}{'OK':>6}{sq:>9.1f}{dts:>9}   {how} · {pat[0]}x{pat[1]}")

        if sq < SQ_TARGET:
            dist = F_PX*30.0/sq
            # 보드 실제 면적 기준 점유. 축정렬 상자로 재면 판을 돌려 들었을 때
            # 오히려 커져서 손실을 놓친다.
            occ = (8*sq)*(5*sq)/(gray.shape[1]*gray.shape[0])*100
            warn.append(
                f"{name}: 칸 {sq:.1f} px (거리 약 {dist:.0f} mm, 보드가 화면의 "
                f"{occ:.0f} %) — {F_PX*30.0/SQ_TARGET:.0f} mm 로 당기면 "
                f"{SQ_TARGET:.0f} px / {(8*SQ_TARGET)*(5*SQ_TARGET)/(gray.shape[1]*gray.shape[0])*100:.0f} % 가 됩니다")
        rot = inplane_deg(corners, pat)
        if rot > 20:
            warn.append(
                f"{name}: 판을 {rot:.0f}° 돌려 들었습니다 — 눕혀 드십시오. "
                f"45° 로 들면 대각선이 화면 세로에 먼저 걸려 칸 12.6 px 가 "
                f"한계입니다 (눕히면 19.5 px)")
        if dt is not None and dt < args.min_dt:
            warn.append(f"{name}: 대비가 {dt:.2f}℃ 뿐 — 판을 더 데우십시오")

        if args.save:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if gray.shape[1] < 400:
                s = 5
                vis = cv2.resize(vis, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_NEAREST)
                cv2.drawChessboardCorners(vis, pat, corners*s, ok)
            else:
                cv2.drawChessboardCorners(vis, pat, corners, ok)
            cv2.imwrite(os.path.join(args.save,
                        os.path.splitext(name)[0] + "_corners.png"), vis)

    print()
    print(f"  {nok}/{len(files)} 검출 성공")
    for w_ in warn:
        print(f"  ⚠ {w_}")
    if nofail:
        have = [x for x in nofail if x[2] is not None]
        cold = [x for x in have if x[2] < 1.0]
        if have:
            print()
            print(f"  ⚠ 숫자만으로는 보드 유무를 판단할 수 없습니다.")
            print(f"     실측 확인 결과, 잎이 무성한 캐노피는 그 자체의 국소대비가")
            print(f"     보드가 있을 때와 사실상 같았습니다 (70.6 vs 70.4).")
            print(f"     프레임 온도 범위가 3℃ 이상이면 대비는 대체로 충분합니다.")
            print(f"     **실패 원인은 --dump 그림으로만 확정하십시오.**")
        flat = [x for x in have if x[1] is not None and x[1] < 2.0]
        if flat:
            print()
            print(f"  ★ {len(flat)}개 파일의 온도 범위가 2℃ 미만입니다. 이 경우는")
            print(f"     대비 부족이 거의 확실하니 판을 더 데우십시오.")
    if nok < len(files):
        print()
        print("  실패했을 때 순서대로 시도")
        print("   1. 판을 다시 데우기 (뒷면에서. 대비가 1℃ 미만이면 이게 원인)")
        print("   2. --both 로 방향 바꿔 시도")
        print("   3. --upscale 3")
        print("   4. 보드 전체가 프레임 안에 있고 가장자리에 여백이 있는지 확인")
        print("   5. --dump chk 로 실제 프레임을 PNG 로 뽑아 눈으로 확인")
        print("      (열화상은 파일명에 온도 범위가 붙습니다. 범위가 좁으면 대비 부족)")
    if nok >= 15:
        print("\n  자세 15개 이상 확보 — 내부 파라미터 산출에 충분합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
