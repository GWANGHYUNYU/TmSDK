#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연속 녹화된 RGB 영상에서 자세별 프레임을 뽑아 열화상과 짝지운다.

열화상은 자세마다 짧게 끊어 녹화하고, RGB 는 계속 돌려놓는 촬영 방식을 위한
도구입니다.

    python3 pair_rgb.py session.mp4 calib/th/ --out calib/rgb/

★ 시각(타임스탬프)으로 맞추지 않고 **순서**로 맞춥니다.
   실측 fps 가 30.55 인데 30 으로 가정하면 30분 뒤 32초가 어긋납니다.
   대신 영상에서 '체커보드가 보이는 구간'을 찾아 앞에서부터 열화상 파일과
   1:1 로 대응시킵니다. 시계 동기가 전혀 필요 없습니다.

촬영할 때 지킬 것 하나
    자세를 바꿀 때 **보드를 화각에서 완전히 빼거나 손으로 가려** 2초쯤 두십시오.
    그 공백이 자세 사이의 경계가 됩니다. 계속 보이면 두 자세가 한 구간으로
    붙어버립니다 (그 경우에도 움직임으로 쪼개려 시도하지만 확실하지 않습니다).
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

import check_board                                        # noqa: E402
from check_board import detect, sharpness, short          # noqa: E402


def scan(video, pat, step_s, coarse_w, move_px, verbose):
    """영상을 훑어 보드가 보이는 구간들을 찾는다.

    → [(구간 시작 프레임, 끝 프레임, [(프레임번호, 코너중심)...])]
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(fps*step_s)))
    print(f"  영상 {W}x{H} · {fps:.2f} fps · {total} 프레임 "
          f"({total/fps/60:.1f} 분) · {step_s:g} 초 간격으로 훑음")

    scale = coarse_w/W if W > coarse_w else 1.0
    hits = []
    i = 0
    while i < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(g, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA) if scale < 1 else g
        ok2, c, _, _ = detect(gs, [pat], 1)
        if ok2:
            ctr = c.reshape(-1, 2).mean(axis=0)/scale
            hits.append((i, ctr))
            if verbose:
                print(f"      {i/fps:7.1f}s  검출  중심 ({ctr[0]:.0f}, {ctr[1]:.0f})")
        i += stride
    cap.release()

    if not hits:
        raise SystemExit(
            "영상 어디에서도 보드를 찾지 못했습니다.\n"
            "  · --pattern 을 확인하십시오 (눕혀 잡으면 7x4, 세우면 4x7)\n"
            "  · --step 0.25 로 더 촘촘히 훑어 보십시오\n"
            "  · check_board.py 로 영상 자체가 검출되는지 먼저 확인하십시오")

    # 공백(검출 실패) 또는 큰 움직임에서 구간을 끊는다
    segs, cur = [], [hits[0]]
    for prev, now in zip(hits, hits[1:]):
        gap = now[0] - prev[0] > stride*1.5
        moved = np.linalg.norm(now[1] - prev[1]) > move_px
        if gap or moved:
            segs.append(cur)
            cur = [now]
        else:
            cur.append(now)
    segs.append(cur)
    return segs, fps, total


def best_frame(video, frames, pat):
    """구간 안에서 원본 해상도로 검출되는 가장 선명한 프레임을 고른다."""
    cap = cv2.VideoCapture(video)
    pick = None
    cand = frames if len(frames) <= 6 else \
        [frames[k] for k in np.linspace(0, len(frames)-1, 6).astype(int)]
    for f in cand:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ok2, _, _, _ = detect(g, [pat], 1)
        if not ok2:
            continue
        s = sharpness(g)
        if pick is None or s > pick[0]:
            pick = (s, int(f), fr)
    cap.release()
    return pick


def main():
    ap = argparse.ArgumentParser(
        description="연속 RGB 영상 → 자세별 프레임 (열화상과 순서로 짝지음)")
    ap.add_argument("video", help="연속 녹화된 RGB 영상")
    ap.add_argument("th_dir", help="열화상 파일이 있는 폴더 (calib/th/)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 <th_dir>/../rgb)")
    ap.add_argument("--pattern", default="7x4", help="내부 코너 수 (기본 7x4)")
    ap.add_argument("--step", type=float, default=0.5,
                    help="훑는 간격 초 (기본 0.5)")
    ap.add_argument("--coarse-width", type=int, default=640,
                    help="탐색용 축소 폭. 빠르게 훑기 위함 (기본 640)")
    ap.add_argument("--move-px", type=float, default=40.0,
                    help="이만큼 움직이면 다른 자세로 간주 (원본 화소, 기본 40)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    w, h = (int(v) for v in args.pattern.lower().split("x"))
    pat = (w, h)
    out_dir = args.out or os.path.join(os.path.dirname(
        os.path.abspath(args.th_dir.rstrip("/\\"))), "rgb")
    os.makedirs(out_dir, exist_ok=True)

    # 열화상 파일 목록 (짝지을 이름)
    stems = []
    for f in sorted(os.listdir(args.th_dir)):
        st, ext = os.path.splitext(f)
        if ext.lower() in check_board.ALL_EXT and st not in stems:
            stems.append(st)
    if not stems:
        raise SystemExit(f"열화상 파일이 없습니다: {args.th_dir}")

    print("=" * 80)
    print(f"RGB 자세 추출 · 패턴 {w}x{h} · 열화상 {len(stems)}자세")
    print("=" * 80)
    segs, fps, total = scan(args.video, pat, args.step,
                            args.coarse_width, args.move_px, args.verbose)

    print(f"\n  보드가 보이는 구간 {len(segs)}개")
    print(f"  {'#':>3}{'시작':>9}{'끝':>9}{'길이':>8}{'검출':>7}")
    for k, sg in enumerate(segs, 1):
        t0, t1 = sg[0][0]/fps, sg[-1][0]/fps
        print(f"  {k:>3}{t0:>8.1f}s{t1:>8.1f}s{t1-t0:>7.1f}s{len(sg):>7}")

    # 개수 비교
    print()
    if len(segs) == len(stems):
        print(f"  ✓ 구간 {len(segs)}개 = 열화상 {len(stems)}자세 — 순서대로 짝지웁니다")
    else:
        print(f"  ⚠ 구간 {len(segs)}개 ≠ 열화상 {len(stems)}자세")
        print("    구간이 더 많으면: 한 자세에서 보드가 잠깐 안 잡혀 둘로 쪼개진 것.")
        print("                     --move-px 를 키우거나 --step 을 줄여 보십시오.")
        print("    구간이 더 적으면: 자세 사이에 보드를 안 뺐거나 검출이 실패한 것.")
        print("                     위 표의 '길이'가 유독 긴 구간을 확인하십시오.")
        print("    ※ 짝이 확실한 앞쪽부터만 저장합니다. 나머지는 직접 확인하십시오.")

    n = min(len(segs), len(stems))
    print()
    print(f"  {'자세':<24}{'구간':>6}{'프레임':>9}{'시각':>9}   결과")
    saved = 0
    for k in range(n):
        frames = [f for f, _ in segs[k]]
        pick = best_frame(args.video, frames, pat)
        if pick is None:
            print(f"  {short(stems[k]):<24}{k+1:>6}{'-':>9}{'-':>9}   "
                  f"원본 해상도에서 검출 실패 — 건너뜀")
            continue
        _, fi, fr = pick
        path = os.path.join(out_dir, stems[k] + ".png")
        cv2.imwrite(path, fr)
        saved += 1
        print(f"  {short(stems[k]):<24}{k+1:>6}{fi:>9}{fi/fps:>8.1f}s   "
              f"{os.path.basename(path)}")

    print()
    print(f"  {saved}/{len(stems)} 자세 저장 → {out_dir}")
    if saved:
        print("\n  다음 단계")
        print(f"    python3 calibrate_pair.py {os.path.dirname(out_dir)} "
              f"--baseline <실측> --depth <실측> --frames 1")
    return 0 if saved == len(stems) else 2


if __name__ == "__main__":
    sys.exit(main())
