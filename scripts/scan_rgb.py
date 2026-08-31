#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연속 RGB 영상을 훑어 보드가 보이는 프레임을 전부 기록한다.

    python3 scripts/scan_rgb.py calib/rgb/xxx.mp4 --out output/rgb_scan.npz

시각 동기가 없을 때 열화상과 짝짓기 위한 1단계입니다. 결과는 npz 로 저장되어
2단계(내부 파라미터 + 기하 매칭)에서 재사용됩니다.

탐색은 축소본에서 하고, 검출된 프레임만 원본 해상도로 다시 정밀 검출합니다.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import detect, inplane_deg, square_px       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RGB 영상 보드 검출 스캔")
    ap.add_argument("video")
    ap.add_argument("--out", default="output/rgb_scan.npz")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--step", type=float, default=1.0, help="훑는 간격 초")
    ap.add_argument("--coarse-width", type=int, default=640)
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = args.coarse_width/W if W > args.coarse_width else 1.0
    stride = max(1, int(round(fps*args.step)))
    print(f"  {W}x{H} · {fps:.2f} fps · {n} 프레임 ({n/fps:.0f} 초)")
    print(f"  {args.step:g} 초 간격 · 탐색 축소 {sc:.2f}")

    idx, corners, rots, sqs, ctrs = [], [], [], [], []
    i = 0
    while i < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gs = (cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
              if sc < 1 else g)
        ok2, _, _, _ = detect(gs, [pat], 1)
        if ok2:
            # 축소본에서 잡혔으면 원본 해상도로 다시 — 코너 정밀도가 중요하다
            ok3, c, _, _ = detect(g, [pat], 1)
            if ok3:
                idx.append(i)
                corners.append(c.reshape(-1, 2))
                rots.append(inplane_deg(c, pat))
                sqs.append(square_px(c, pat))
                ctrs.append(c.reshape(-1, 2).mean(0))
                print(f"    {i/fps:7.1f}s  칸 {sqs[-1]:5.1f}px · "
                      f"면내회전 {rots[-1]:4.0f}° · "
                      f"중심 ({ctrs[-1][0]:.0f}, {ctrs[-1][1]:.0f})")
        i += stride
    cap.release()

    if not idx:
        raise SystemExit("보드를 찾지 못했습니다.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, idx=np.array(idx), corners=np.array(corners),
             rots=np.array(rots), sqs=np.array(sqs), ctrs=np.array(ctrs),
             fps=fps, size=np.array([W, H]), video=args.video)

    # 구간 나누기 — 검출 공백 또는 큰 움직임
    t = np.array(idx)/fps
    seg, cur = [], [0]
    for k in range(1, len(idx)):
        gap = t[k]-t[k-1] > args.step*2.5
        moved = np.linalg.norm(ctrs[k]-ctrs[k-1]) > 120
        if gap or moved:
            seg.append(cur)
            cur = [k]
        else:
            cur.append(k)
    seg.append(cur)

    print(f"\n  검출 {len(idx)}프레임 · 구간 {len(seg)}개")
    print(f"  {'#':>3}{'시작':>8}{'끝':>8}{'길이':>7}{'장':>5}"
          f"{'면내회전':>12}{'칸px':>11}")
    for k, s in enumerate(seg, 1):
        r = [rots[j] for j in s]
        q = [sqs[j] for j in s]
        print(f"  {k:>3}{t[s[0]]:>7.0f}s{t[s[-1]]:>7.0f}s"
              f"{t[s[-1]]-t[s[0]]:>6.0f}s{len(s):>5}"
              f"{min(r):>6.0f}~{max(r):<5.0f}{min(q):>5.0f}~{max(q):<5.0f}")
    print(f"\n  저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
