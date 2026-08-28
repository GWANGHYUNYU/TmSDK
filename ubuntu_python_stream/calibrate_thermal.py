#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""열화상 단독 캘리브레이션 검증 — 현장에서 돌아오기 전에 판정.

RGB 는 사무실에서 처리하고 현장에서는 열화상만 저장하는 경우를 위한 도구입니다.
열화상만으로 판정할 수 있는 것을 전부 판정하고, **무엇을 더 찍어야 하는지**
구체적으로 알려 줍니다.

    python3 calibrate_thermal.py calib/th/ --frames 1

    # 줄자로 잰 값이 있으면 함께 검증 (강력 권장)
    python3 calibrate_thermal.py calib/th/ --frames 1 \
        --measured pose07=absolute420 --canopy 470

열화상만으로 판정 가능한 것
    자세 수 · 코너 검출 · 내부 파라미터(K, 왜곡) · 재투영 RMS
    화면 커버리지 · 기울임 다양성 · 깊이 분포 · 보드 대비(℃)
    보드 기하로 역산한 자세별 거리 ↔ 줄자 대조

RGB 가 있어야 판정 가능한 것 (사무실에서 calibrate_pair.py)
    스테레오 외부 파라미터 R·T · 베이스라인 · 평면 호모그래피
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import check_board                                            # noqa: E402
from check_board import (detect_file, square_px, contrast_c,   # noqa: E402
                         load_candidates, detect,
                         scene_stats, short)

CALIB_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
GRID = 3          # 화면을 3x3 으로 나눠 커버리지 확인


def object_points(pat, cell_mm):
    o = np.zeros((pat[0]*pat[1], 3), np.float32)
    o[:, :2] = np.mgrid[0:pat[0], 0:pat[1]].T.reshape(-1, 2)
    return o * cell_mm


def gather(d):
    """이름 → 경로. 부속 파일 제외, y16raw 우선."""
    rank = {}
    for f in sorted(os.listdir(d)):
        stem, ext = os.path.splitext(f)
        ext = ext.lower()
        if ext in check_board.SKIP_EXT or ext not in check_board.ALL_EXT:
            continue
        pr = (0 if ext in check_board.RAW_EXT else
              1 if ext in check_board.IMG_EXT else 2)
        if stem not in rank or pr < rank[stem][0]:
            rank[stem] = (pr, os.path.join(d, f))
    return {k: v[1] for k, v in sorted(rank.items())}


def coverage_map(pts, size):
    """검출된 코너 점들이 3x3 어느 칸에 닿았는지 → 채워진 칸 집합.

    보드 중심이 아니라 **코너 점** 기준이어야 한다. 보드가 화면의 68 %x57 %
    를 채우므로 중심은 좌우·상하 1/3 칸에 들어갈 수 없고, 중심 기준으로 잡으면
    달성 불가능한 목표가 된다. 왜곡 계수에 필요한 것은 '주변부 코너 관측'이므로
    코너가 그 칸에 닿았는지가 옳은 척도다.
    """
    W, H = size
    seen = set()
    for cx, cy in pts:
        gx = min(GRID-1, max(0, int(cx/W*GRID)))
        gy = min(GRID-1, max(0, int(cy/H*GRID)))
        seen.add((gx, gy))
    return seen


def spread(pts, size):
    """코너 점 전체가 차지한 폭·높이 비율, 네 모서리까지 최근접 거리."""
    W, H = size
    P = np.asarray(pts, np.float64)
    fw = (P[:, 0].max()-P[:, 0].min())/W
    fh = (P[:, 1].max()-P[:, 1].min())/H
    diag = float(np.hypot(W, H))
    cor = {}
    for nm, (cx, cy) in (("좌상", (0, 0)), ("우상", (W, 0)),
                         ("좌하", (0, H)), ("우하", (W, H))):
        cor[nm] = float(np.min(np.hypot(P[:, 0]-cx, P[:, 1]-cy)))/diag
    return fw, fh, cor


CELL_NAME = [["좌상", "상", "우상"], ["좌", "중앙", "우"], ["좌하", "하", "우하"]]


def draw_coverage(path, size, polys, centers, seen):
    """자세 배치도 — 어디를 못 찍었는지 눈으로 보게."""
    W, H = size
    S = max(3, int(700/W))
    vis = np.full((H*S, W*S, 3), 250, np.uint8)
    for gy in range(GRID):
        for gx in range(GRID):
            x0, y0 = int(gx*W*S/GRID), int(gy*H*S/GRID)
            x1, y1 = int((gx+1)*W*S/GRID), int((gy+1)*H*S/GRID)
            if (gx, gy) not in seen:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (215, 230, 255), -1)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 200), 1)
            if (gx, gy) not in seen:
                cv2.putText(vis, "MISSING", (x0+8, (y0+y1)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5*S/4,
                            (60, 60, 200), max(1, S//3), cv2.LINE_AA)
    for p in polys:
        cv2.polylines(vis, [np.int32(p*S)], True, (150, 110, 60), max(1, S//3),
                      cv2.LINE_AA)
    for cx, cy in centers:
        cv2.drawMarker(vis, (int(cx*S), int(cy*S)), (30, 120, 30),
                       cv2.MARKER_CROSS, 8*S//3, max(1, S//3))
    cv2.imwrite(path, vis)


def main():
    ap = argparse.ArgumentParser(description="열화상 단독 캘리브레이션 검증")
    ap.add_argument("th_dir", help="열화상 파일 폴더 (calib/th/)")
    ap.add_argument("--pattern", default="7x4", help="내부 코너 수 (기본 7x4)")
    ap.add_argument("--cell", type=float, default=30.0, help="칸 크기 mm")
    ap.add_argument("--focal", type=float, default=None,
                    help="초점거리를 px 로 고정. 줄자로 확정했으면 넣으십시오. "
                         "기울임·역산거리가 f 에 의존하므로 결과가 달라집니다 "
                         "(TMC160F 실측 147.4)")
    ap.add_argument("--no-refine", action="store_true",
                    help="프레임 재선택을 끕니다 (기본 켜짐). 가장 선명한 "
                         "프레임이 코너가 가장 정확한 프레임은 아닙니다")
    ap.add_argument("--frames", type=int, default=12,
                    help="평균 프레임 수. 손으로 들고 찍었으면 1 (가장 선명한 장)")
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--canopy", type=float, default=None,
                    help="★ 열화상 렌즈 ~ 잎 표면 실측 거리 mm. 이 스크립트는 "
                         "열화상 좌표계로 계산하므로 열화상 기준값입니다. "
                         "(calibrate_pair.py 의 --depth 는 RGB 기준)")
    ap.add_argument("--measured", action="append", default=[],
                    metavar="POSE=MM",
                    help="줄자로 잰 보드 거리. 예: --measured pose07=420 "
                         "(여러 번 지정 가능)")
    ap.add_argument("--min-dt", type=float, default=1.0,
                    help="최소 대비 ℃ (기본 1.0)")
    ap.add_argument("--out", default="thermal_calib.json")
    ap.add_argument("--coverage", default="coverage.png")
    args = ap.parse_args()

    check_board.NFRAMES = args.frames
    w, h = (int(v) for v in args.pattern.lower().split("x"))
    pat = (w, h)
    obj = object_points(pat, args.cell)
    measured = {}
    for m in args.measured:
        k, v = m.split("=")
        measured[k.strip()] = float(v)

    files = gather(args.th_dir)
    if not files:
        raise SystemExit(f"열화상 파일이 없습니다: {args.th_dir}")

    print("=" * 84)
    print(f"열화상 단독 검증 · 패턴 {w}x{h} · 칸 {args.cell:g} mm · "
          f"OpenCV {cv2.__version__}")
    print("=" * 84)
    print(f"  {'자세':<24}{'검출':>6}{'칸(px)':>9}{'대비':>9}{'중심(x,y)':>14}")

    names, PTS, centers, polys, dts, ALLP = [], [], [], [], [], []
    size = None
    for n, p in files.items():
        try:
            ok, c, pp, how, cels, gray, _ = detect_file(p, [pat], args.upscale)
        except Exception as e:
            print(f"  {short(n):<24}{'오류':>6}   {e}")
            continue
        size = gray.shape[::-1]
        if not ok:
            print(f"  {short(n):<24}{'실패':>6}")
            continue
        pts = c.reshape(-1, 2)
        ctr = pts.mean(axis=0)
        sq = square_px(c, pat)
        dt = contrast_c(cels, c, pat)
        g = pts.reshape(pat[1], pat[0], 2)
        polys.append(np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]]))
        names.append(n); PTS.append(c.astype(np.float32))
        centers.append(ctr); dts.append(dt); ALLP.append(pts)
        print(f"  {short(n):<24}{'OK':>6}{sq:>9.1f}"
              f"{(f'{dt:.2f}℃' if dt is not None else '-'):>9}"
              f"{f'{ctr[0]:.0f}, {ctr[1]:.0f}':>14}")

    n = len(names)
    print(f"\n  검출 성공 {n}/{len(files)}")
    if n < 4:
        print("  ✗ 내부 파라미터를 구할 수 없습니다. 최소 8자세, 권장 20자세.")
        return 1

    def calib(pts_list):
        if args.focal:
            Kg = np.array([[args.focal, 0, size[0]/2],
                           [0, args.focal, size[1]/2], [0, 0, 1]])
            return cv2.calibrateCamera(
                [obj]*len(pts_list), pts_list, size, Kg, None,
                flags=(CALIB_FLAGS | cv2.CALIB_FIX_FOCAL_LENGTH
                       | cv2.CALIB_USE_INTRINSIC_GUESS))
        return cv2.calibrateCamera([obj]*len(pts_list), pts_list, size,
                                   None, None, flags=CALIB_FLAGS)

    # ── 내부 파라미터 ──
    print("\n" + "=" * 84)
    print("내부 파라미터 (열화상 단독으로 확정 가능)")
    print("=" * 84)
    rms, K, D, rv, tv = calib(PTS)

    # ★ 프레임 재선택 — '가장 선명한' 프레임이 '코너가 가장 정확한' 프레임은
    #   아니다. 1차 데이터에서 한 자세가 RMS 1.34 → 0.21 로 좋아졌고 역산
    #   거리도 523 → 455 mm 로 줄자에 맞았다. 1차 모델로 후보 프레임들을
    #   다시 평가해서 재투영 오차가 가장 작은 것을 고른다.
    if not args.no_refine:
        swapped, before = 0, rms
        for i, nm in enumerate(names):
            try:
                cands = load_candidates(files[nm])
            except Exception:
                continue
            best = (float("inf"), None)
            for _, g8 in cands:
                ok2, c2, _, _ = detect(g8, [pat], args.upscale)
                if not ok2:
                    continue
                c2 = c2.astype(np.float64)
                okp, rvi, tvi = cv2.solvePnP(obj, c2, K, D)
                if not okp:
                    continue
                pr, _ = cv2.projectPoints(obj, rvi, tvi, K, D)
                e = float(np.sqrt(((c2.reshape(-1, 2)-pr.reshape(-1, 2))**2)
                                  .sum(axis=1).mean()))
                if e < best[0]:
                    best = (e, c2.astype(np.float32))
            if best[1] is not None:
                cur, _, _ = cv2.solvePnP(obj, PTS[i].astype(np.float64), K, D)
                pr, _ = cv2.projectPoints(obj, *cv2.solvePnP(
                    obj, PTS[i].astype(np.float64), K, D)[1:], K, D)
                e0 = float(np.sqrt(((PTS[i].reshape(-1, 2)-pr.reshape(-1, 2))**2)
                                   .sum(axis=1).mean()))
                if best[0] < e0*0.9:
                    PTS[i] = best[1]
                    ALLP[i] = best[1].reshape(-1, 2)
                    swapped += 1
        if swapped:
            rms, K, D, rv, tv = calib(PTS)
            print(f"  프레임 재선택: {swapped}자세를 더 좋은 프레임으로 교체 "
                  f"→ RMS {before:.3f} → {rms:.3f} px")

    if args.focal:
        print(f"  ★ 초점거리를 {args.focal:g} px 로 고정했습니다 "
              f"(줄자 실측으로 확정된 값)")
    fx, fy = K[0, 0], K[1, 1]
    fov_h = 2*np.degrees(np.arctan(size[0]/(2*fx)))
    fov_v = 2*np.degrees(np.arctan(size[1]/(2*fy)))
    print(f"  RMS 재투영 오차  {rms:.3f} px")
    print(f"  f = ({fx:.1f}, {fy:.1f})   주점 = ({K[0,2]:.1f}, {K[1,2]:.1f})")
    print(f"  왜곡 k1={D[0,0]:+.4f}  k2={D[0,1]:+.4f}")
    print(f"  역산 화각 {fov_h:.1f}° × {fov_v:.1f}°   (사양 42° × 32°)")
    if args.focal:
        print(f"    (사양 42° 는 실측과 맞지 않는 것으로 확인됐습니다. "
              f"줄자 실측값을 신뢰하십시오.)")
    elif abs(fov_h-42) > 4:
        print(f"    ⚠ 사양과 {abs(fov_h-42):.1f}° 차이 — 칸 크기(--cell)나 "
              f"패턴을 확인하십시오. 줄자 실측이 있으면 --focal 로 고정하는 "
              f"편이 정확합니다")

    # ── 초점거리 결정력 ──
    # ★ RMS 가 낮아도 f 가 결정되지 않을 수 있다. 정면 자세만 있거나 보드가
    #   화면 중앙에만 머물면 f 와 왜곡계수 k1 이 서로를 상쇄해서, f 를 반이나
    #   두 배로 틀리게 고정해도 RMS 가 거의 그대로다. 그러면 역산 거리와
    #   깊이별 호모그래피 H(d) 가 통째로 틀린다. RMS 만 보면 절대 못 잡는다.
    if args.focal:
        # f 를 줄자로 외부 확정했으면 이 지표는 의미가 없다. 사진이 f 를
        # 결정하지 못해도 상관없다 — 이미 알고 있으므로.
        fpow = 1.0
        print(f"  초점거리 결정력  해당 없음 — 줄자 실측으로 외부 확정 "
              f"(사진이 f 를 못 정해도 무방합니다)")
        print(f"    대신 아래 «줄자 대조»에서 이 f 가 맞는지 검증됩니다.")
    else:
        fpow = None
    worse = []
    for probe in (fx*0.7, fx*1.4) if fpow is None else ():
        Kg = np.array([[probe, 0, size[0]/2], [0, probe, size[1]/2], [0, 0, 1]])
        worse.append(cv2.calibrateCamera(
            [obj]*n, PTS, size, Kg, None,
            flags=(CALIB_FLAGS | cv2.CALIB_FIX_FOCAL_LENGTH
                   | cv2.CALIB_USE_INTRINSIC_GUESS))[0])
    if fpow is None:
        fpow = (min(worse)-rms)/max(rms, 1e-9)
        print(f"  초점거리 결정력  {fpow*100:.0f} %  "
              f"(f 를 -30 % / +40 % 로 고정 → RMS {worse[0]:.3f} / "
              f"{worse[1]:.3f})")
        print(f"    이 데이터가 f 를 얼마나 확실히 결정하는지입니다. "
              f"50 % 이상이면 신뢰할 수 있습니다.")
        if fpow < 0.5:
            print(f"    ✗ f 가 사실상 결정되지 않았습니다. 역산 거리와 H(d) 를 "
                  f"쓰면 안 됩니다.")
            print(f"      원인은 (1) 정면 자세만 있음 (2) 보드가 화면 중앙에만 "
                  f"머묾 — 둘 다 f 와 왜곡을 분리하지 못하게 합니다.")
            print(f"      줄자로 서로 다른 거리 3점을 재 오면 --focal 로 "
                  f"외부에서 확정할 수 있습니다.")

    # ── 자세 다양성 ──
    depths, tilts = [], []
    for i in range(n):
        R = cv2.Rodrigues(rv[i])[0]
        depths.append(float((R @ obj.T + tv[i])[2].mean()))
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))))
    depths, tilts = np.array(depths), np.array(tilts)
    allp = np.vstack(ALLP)
    seen = coverage_map(allp, size)
    fw, fh, cor = spread(allp, size)
    missing = [CELL_NAME[gy][gx] for gy in range(GRID) for gx in range(GRID)
               if (gx, gy) not in seen]

    print("\n" + "=" * 84)
    print("자세 다양성 — 내부 파라미터 품질을 좌우합니다")
    print("=" * 84)
    print(f"  코너가 닿은 칸  {len(seen)}/9")
    if missing:
        print(f"    ✗ 코너가 못 닿은 칸: {', '.join(missing)}")
        print(f"      → 왜곡 계수는 주변부 코너 관측이 있어야 추정됩니다. "
              f"보드를 그 방향으로 밀어서 더 찍으십시오.")
    print(f"  코너 분포 폭   가로 {fw*100:.0f} % · 세로 {fh*100:.0f} % "
          f"(각 80 % 이상 권장)")
    far = [k for k, v in cor.items() if v > 0.25]
    print(f"  모서리 접근    " + " · ".join(
        f"{k} {v*100:.0f}%" for k, v in cor.items()) + "  (화면 대각선 대비)")
    if far:
        print(f"    ⚠ {', '.join(far)} 쪽이 비어 있습니다. 그 모서리로 보드를 "
              f"밀어 넣은 자세를 추가하십시오.")
    print(f"  기울임         {tilts.min():.0f}° ~ {tilts.max():.0f}°  "
          f"(20° 이상 자세 {int((tilts >= 20).sum())}개)")
    if (tilts >= 20).sum() < 4:
        print(f"    ✗ 기울인 자세가 부족합니다. 정면만 찍으면 초점거리가 "
              f"수학적으로 결정되지 않습니다.")
    print(f"  보드 거리      {depths.min():.0f} ~ {depths.max():.0f} mm "
          f"(폭 {depths.max()-depths.min():.0f} mm)")
    if depths.max()-depths.min() < 60:
        print(f"    ⚠ 거리 변화가 작습니다. 2~3개 거리에서 찍으면 깊이 모델을 "
              f"검증할 수 있습니다.")

    # 자세별 재투영 오차 — 전체 RMS 는 나쁜 한두 장에 통째로 끌려간다.
    # 어느 장을 다시 찍어야 하는지 알려면 자세별로 봐야 한다.
    # ★ 점유는 축정렬 외접상자가 아니라 **보드 자신의 면적**으로 재야 한다.
    #   판을 45° 로 돌려 들면 외접상자는 오히려 커져서, 정작 보드가 작게
    #   담긴 것을 놓친다 (1차에서 실제로 놓쳤다).
    perr, occ, rot = [], [], []
    for i in range(n):
        pr, _ = cv2.projectPoints(obj, rv[i], tv[i], K, D)
        d2 = ((PTS[i].reshape(-1, 2)-pr.reshape(-1, 2))**2).sum(axis=1)
        perr.append(float(np.sqrt(d2.mean())))
        g = PTS[i].reshape(pat[1], pat[0], 2)
        q = np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]])   # 내부코너 네 귀
        a = 0.5*abs(np.dot(q[:, 0], np.roll(q[:, 1], -1))
                    - np.dot(q[:, 1], np.roll(q[:, 0], -1)))     # 신발끈 공식
        # 내부코너는 (칸수-1) 칸이므로 판 전체 면적으로 환산
        full = a*(pat[0]+1)*(pat[1]+1)/((pat[0]-1)*(pat[1]-1))
        occ.append(float(full/(size[0]*size[1])))
        v = g[:, -1].mean(0)-g[:, 0].mean(0)                     # 긴 변 방향
        ang = abs(float(np.degrees(np.arctan2(v[1], v[0]))))
        rot.append(min(ang, 180-ang))
    perr, occ, rot = np.array(perr), np.array(occ), np.array(rot)

    print(f"\n  {'자세':<24}{'거리(mm)':>10}{'기울임':>8}{'면내회전':>9}"
          f"{'점유':>7}{'자세RMS':>9}{'대비':>9}")
    for i in range(n):
        mk = ""
        if names[i] in measured:
            d = measured[names[i]]
            err = depths[i]-d
            mk = (f"   줄자 {d:.0f} mm · 차이 {err:+.0f} mm "
                  f"({abs(err)/d*100:.1f} %)")
        elif perr[i] > max(0.5, 2.5*np.median(perr)):
            mk = "   ← 이 장이 전체 RMS 를 끌어올립니다. 다시 찍으십시오"
        elif rot[i] > 20:
            mk = "   ← 판을 돌려 들어 작게 담겼습니다. 눕혀 드십시오"
        print(f"  {short(names[i]):<24}{depths[i]:>10.0f}{tilts[i]:>7.0f}°"
              f"{rot[i]:>8.0f}°{occ[i]*100:>6.0f}%{perr[i]:>9.3f}"
              f"{(f'{dts[i]:.2f}℃' if dts[i] is not None else '-'):>9}{mk}")
    if (rot > 20).sum():
        print(f"\n  ⚠ {int((rot > 20).sum())}자세가 판을 20° 이상 돌려 "
              f"들었습니다 (면내회전).")
        print(f"    보정 정확도에는 무해하지만, 45° 로 들면 보드 대각선이 화면 "
              f"세로에 먼저 걸려")
        print(f"    더 가까이 갈 수 없습니다. 눕혀 들 때 칸 19.5 px 까지 되는 "
              f"것이 45° 에서는 12.6 px 입니다.")
    bad = [names[i] for i in range(n)
           if perr[i] > max(0.5, 2.5*np.median(perr))]
    if bad:
        keep = [i for i in range(n) if names[i] not in bad]
        if len(keep) >= 4:
            r2 = cv2.calibrateCamera([obj]*len(keep), [PTS[i] for i in keep],
                                     size, None, None, flags=CALIB_FLAGS)[0]
            print(f"\n  불량 {len(bad)}장을 빼면 RMS {rms:.3f} → {r2:.3f} px. "
                  f"코너 검출 자체는 정상이라는 뜻입니다.")
    if occ.mean() < 0.40:
        print(f"\n  ⚠ 보드가 화면의 평균 {occ.mean()*100:.0f} % 만 차지합니다 "
              f"(40~70 % 권장).")
        print(f"    더 가까이 대면 코너 위치 정밀도와 왜곡 추정이 함께 "
              f"좋아집니다.")

    # ── 줄자 대조 ──
    # ★ 실측 1개로는 아무것도 못 가린다. 계산 거리와 줄자는
    #       d_계산 = s·d_줄자 + c
    #   관계인데, s≠1 은 초점거리가 틀린 것(거리에 비례)이고 c≠0 은 줄자를
    #   투영중심이 아닌 곳(렌즈 앞면 등)에서 잰 것(거리와 무관한 고정 오차)이다.
    #   보통 c 는 10~20 mm 로, 350 mm 에서 3~6 % 라 s 와 크기가 비슷하다.
    #   미지수 2개이므로 서로 다른 거리에서 최소 2점, 권장 3점이 필요하다.
    scale_ok = None
    if measured:
        pair = [(v, depths[names.index(k)], k)
                for k, v in measured.items() if k in names]
        pair.sort()
        errs = [abs(e-t)/t*100 for t, e, _ in pair]
        print(f"\n  줄자 대조 {len(pair)}개")
        print(f"  {'자세':<24}{'줄자':>9}{'계산':>9}{'차이':>9}{'비율':>8}")
        for t, e, k in pair:
            print(f"  {short(k):<24}{t:>8.0f}mm{e:>8.0f}mm"
                  f"{e-t:>+8.0f}mm{e/t:>8.3f}")

        # ── ① 비례 모형 (c=0) — 이것이 초점거리를 직접 준다 ──
        # 자세별 거리는 f 에 정비례하므로 f_참값 = f0 x (줄자/계산).
        # 줄자 기준점이 투영중심에 가까우면 이 비가 자세마다 같아야 한다.
        rat = np.array([t/e for t, e, _ in pair])
        mrat = float(np.median(rat))
        out = [k for (t, e, k), r in zip(pair, rat)
               if abs(r/mrat-1) > 0.05]
        print(f"\n  ① 비례 모형 — f_참값 = f x (줄자/계산)")
        print(f"     자세별 비  " + " ".join(f"{r:.3f}" for r in rat))
        print(f"     중앙값 {mrat:.3f} · 산포 {rat.std()/mrat*100:.1f} %"
              f"  →  f = {fx:.1f} × {mrat:.3f} = {fx*mrat:.1f}"
              f"  (화각 {2*np.degrees(np.arctan(size[0]/(2*fx*mrat))):.1f}°"
              f" × {2*np.degrees(np.arctan(size[1]/(2*fx*mrat))):.1f}°)")
        if out:
            keep = np.array([r for (t, e, k), r in zip(pair, rat)
                             if k not in out])
            print(f"     ⚠ 중앙값에서 5 % 넘게 벗어난 자세: "
                  f"{', '.join(short(k) for k in out)}")
            print(f"       그 자세의 줄자를 잘못 읽었거나, 재고 나서 보드를 "
                  f"옮겼을 수 있습니다.")
            if len(keep) >= 2:
                print(f"       이들을 빼면 비 {np.median(keep):.3f} ± "
                      f"{keep.std()/np.median(keep)*100:.1f} %  →  "
                      f"f = {fx*np.median(keep):.1f}")
        else:
            print(f"     ✓ 모든 자세가 5 % 이내로 일치합니다 — f 가 "
                  f"확정되었습니다.")

        span = pair[-1][0]-pair[0][0] if len(pair) >= 2 else 0.0
        if len(pair) >= 2:
            print(f"\n  ② 선형 모형 (기준점 어긋남 c 까지 추정)")
            # d_계산 = s·d_줄자 + c  최소제곱
            td = np.array([t for t, _, _ in pair], float)
            ed = np.array([e for _, e, _ in pair], float)
            A = np.column_stack([td, np.ones(len(td))])
            (s, c), *_ = np.linalg.lstsq(A, ed, rcond=None)
            resid = np.abs(A @ [s, c] - ed)
            # ★ 불확도를 반드시 같이 봐야 한다. 측정 거리가 몰려 있으면 s 와 c 가
            #   서로를 상쇄해서, 맞춤 결과는 그럴듯한데 실제로는 아무것도 결정되지
            #   않는다. 줄자 오차는 ±5 mm 로 가정한다.
            SIG = 5.0
            sxx = float(((td-td.mean())**2).sum())
            se_s = SIG/np.sqrt(sxx) if sxx > 0 else np.inf
            se_c = (SIG*np.sqrt(1/len(td) + td.mean()**2/sxx)
                    if sxx > 0 else np.inf)
            print(f"\n  d_계산 = {s:.4f} × d_줄자 {c:+.1f} mm   "
                  f"(측정 폭 {span:.0f} mm · 잔차 최대 {resid.max():.1f} mm)")
            print(f"    기울기 s = {s:.4f} ± {se_s:.4f}  →  초점거리가 실제보다 "
                  f"{(s-1)*100:+.1f} % ± {se_s*100:.1f} % 추정")
            print(f"       보정하면 f = {fx:.1f} → {fx/s:.1f} "
                  f"(화각 {2*np.degrees(np.arctan(size[0]*s/(2*fx))):.1f}°)")
            print(f"    절편 c = {c:+.1f} ± {se_c:.1f} mm  →  줄자 기준점이 "
                  f"투영중심보다 {'앞' if c > 0 else '뒤'}에 있음")

            # 결정력 판정 — s 를 5 % 안으로 묶지 못하면 쓸 수 없다.
            # 합격 여부는 ①비례 모형의 산포로 본다. ②는 보조 지표.
            usable = se_s <= 0.05 and not out
            scale_ok = (not out) and rat.std()/mrat <= 0.05
            if not usable:
                # √Σ(d-d̄)² 는 측정 폭에 비례하므로, 같은 배치를 유지한 채
                # 폭만 이만큼 늘리면 5 % 에 닿는다.
                print(f"\n    ✗ s 의 불확도 {se_s*100:.1f} % 가 판정 기준 5 % 보다 "
                      f"큽니다 — 이 맞춤은 쓸 수 없습니다.")
                print(f"      측정 거리가 몰려 있어 s 와 c 가 서로 상쇄됩니다. "
                      f"위 숫자를 믿지 마십시오.")
                print(f"      같은 배치로 폭을 {span*se_s/0.05:.0f} mm 까지 "
                      f"넓히거나, 실측 자세를 더 늘리십시오.")
            if resid.max() > 15:
                print(f"    ⚠ 잔차가 큽니다. 줄자 기준점이 매번 달랐거나 "
                      f"자세 이름이 어긋났을 수 있습니다.")
            if usable and abs(c) > 25:
                print(f"    ⚠ 절편이 큽니다 (정상 ±25 mm). 줄자를 어디에 "
                      f"댔는지 확인하십시오.")
        else:
            print(f"\n  ⚠ 실측이 1개뿐입니다. 차이 {errs[0]:.1f} % 가 초점거리 "
                  f"오차인지, 줄자 기준점이")
            print(f"    투영중심과 어긋난 것인지 **구분할 수 없습니다.** "
                  f"다른 거리에서 1~2개 더 재십시오.")
            scale_ok = None
    else:
        print(f"\n  ⚠ 줄자 실측값이 없습니다. 서로 다른 거리 3자세에서 재고 "
              f"--measured pose16=320\n     --measured pose17=430 --measured "
              f"pose18=540 처럼 넣으면 초점거리를 독립 검증할 수 있습니다.")

    # ── 캐노피 외삽 ──
    if args.canopy:
        print("\n" + "=" * 84)
        print(f"정합 기준 깊이 {args.canopy:g} mm (렌즈~잎 표면 실측)")
        print("=" * 84)
        lo, hi = depths.min(), depths.max()
        if lo <= args.canopy <= hi:
            print(f"  ✓ 보드 거리 범위 {lo:.0f}~{hi:.0f} mm 안에 있습니다 "
                  f"(내삽) — 가장 안전합니다.")
        else:
            gap = (args.canopy-hi) if args.canopy > hi else (lo-args.canopy)
            print(f"  보드 거리 범위 {lo:.0f}~{hi:.0f} mm 밖입니다. "
                  f"외삽 거리 {gap:.0f} mm")
            if gap <= 50:
                print(f"  ○ 50 mm 이내 외삽이면 실용상 문제없습니다.")
            else:
                print(f"  ⚠ 외삽이 50 mm 를 넘습니다. 잎 표면 거리에 더 가까운 "
                      f"자세를 몇 장 추가하십시오.")
                print(f"    식물을 건드릴 수 없으면, **같은 거리에서 옆으로 비켜** "
                      f"식물 없는 곳에 두고 찍으십시오.")

    # ── 판정 ──
    print("\n" + "=" * 84)
    print("현장 판정 — 열화상 단독")
    print("=" * 84)
    have_dt = [d for d in dts if d is not None]
    lowdt = [names[i] for i in range(n)
             if dts[i] is not None and dts[i] < args.min_dt]
    checks = [
        ("자세 수 ≥ 15", n >= 15, f"{n}개"),
        ("재투영 RMS < 0.5 px", rms < 0.5, f"{rms:.3f} px (목표 0.3)"),
        ("초점거리 결정력 ≥ 50 %", fpow >= 0.5,
         "줄자로 외부 확정" if args.focal else f"{fpow*100:.0f} %"),
        ("보드 화면 점유 ≥ 40 %", occ.mean() >= 0.40,
         f"평균 {occ.mean()*100:.0f} %"),
        ("면내회전 20° 초과 자세 없음", int((rot > 20).sum()) == 0,
         f"{int((rot > 20).sum())}개"),
        ("코너가 9칸 모두 닿음", not missing, f"{len(seen)}/9"),
        ("코너 분포 폭 ≥ 80 %", fw >= 0.8 and fh >= 0.8,
         f"가로 {fw*100:.0f} % · 세로 {fh*100:.0f} %"),
        ("20° 이상 기울임 ≥ 4개", int((tilts >= 20).sum()) >= 4,
         f"{int((tilts >= 20).sum())}개"),
        ("거리 폭 ≥ 60 mm", depths.max()-depths.min() >= 60,
         f"{depths.max()-depths.min():.0f} mm"),
    ]
    if have_dt:
        checks.append((f"대비 ≥ {args.min_dt:g}℃", not lowdt,
                       f"{min(have_dt):.2f} ~ {max(have_dt):.2f}℃"
                       + ("" if not lowdt else f" · {len(lowdt)}개 미달")))
    if scale_ok is not None:
        checks.append(("줄자 대조 스케일 오차 ≤ 5 %", scale_ok, ""))
    for label, ok, val in checks:
        print(f"  {'✓' if ok else '✗'}  {label:<26}{val}")
    allok = all(c[1] for c in checks)
    if lowdt:
        print(f"     대비 미달: {', '.join(lowdt[:6])} — 판을 더 데워 재촬영")
    if not have_dt:
        # 영상(avi/mp4)에는 온도값이 없어 대비를 숫자로 확인할 수 없다.
        # 검사를 통과로 표시하면 검증하지 않은 것을 통과로 오독하게 된다.
        print(f"  ―  대비 ≥ {args.min_dt:g}℃{'':<14}"
              f"판정 불가 (영상에는 온도값이 없음)")
        print(f"     검출이 성공했다는 것 자체가 대비가 충분했다는 간접 증거입니다.")
        print(f"     다만 여유가 1.2℃ 인지 8℃ 인지는 알 수 없습니다.")
        print(f"     .y16raw 를 함께 저장하면 자세별 대비가 ℃ 로 표시됩니다.")

    draw_coverage(args.coverage, size, polys, centers, seen)
    json.dump({
        "pattern": [w, h], "cell_mm": args.cell, "poses": names,
        "size": list(size), "K": K.tolist(), "dist": D.ravel().tolist(),
        "rms_px": rms, "fov_deg": [fov_h, fov_v],
        "pose_depth_mm": depths.tolist(), "tilt_deg": tilts.tolist(),
        "contrast_c": [None if d is None else d for d in dts],
        "coverage_cells": len(seen), "spread": [fw, fh],
        "corner_gap": cor, "canopy_mm": args.canopy,
        "measured_mm": measured, "pass": bool(allok),
    }, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n  저장: {args.out} · 자세 배치도: {args.coverage}")
    print("\n  " + ("현장 작업 완료 — 돌아가셔도 됩니다."
                    if allok else "위 ✗ 항목을 보완하고 재촬영하십시오."))
    print("\n  사무실에서 RGB 를 붙인 뒤")
    print(f"    python3 pair_rgb.py session.mp4 {args.th_dir} --out ../rgb/")
    print(f"    python3 calibrate_pair.py <calib> --baseline <실측> "
          f"--depth {args.canopy or '<잎표면거리>'} --frames {args.frames}")
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())
