#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""매칭된 짝의 RGB 프레임을 원본 프레임 단위로 다시 고른다.

    python3 scripts/refine_match.py output/pair_match/match.npz \
        calib/rgb/xxx.mp4 --window 2.0

왜 필요한가
    1단계 스캔은 1초 간격이라 판이 움직이는 동안 최대 0.5초 어긋난 프레임이
    잡힙니다. 초당 2° 씩 돌아가면 그것만으로 1° 오차가 생기고, 스테레오
    외부 파라미터가 그만큼 틀어집니다.

    각 짝 주변 ±window 초를 **모든 프레임**에서 훑어, 상대 회전·베이스라인이
    전체 평균과 가장 잘 맞는 프레임으로 교체합니다. 두 번 반복합니다.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import detect                                 # noqa: E402
from calibrate_thermal import object_points                    # noqa: E402


def pose(obj, pts, K, D):
    ok, rv, tv = cv2.solvePnP(obj, pts.reshape(-1, 1, 2).astype(np.float32),
                              K, D)
    if not ok:
        return None, None
    return cv2.Rodrigues(rv)[0], tv.reshape(3)


def ang(A, B):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(A @ B.T)-1)/2, -1, 1))))


def main():
    ap = argparse.ArgumentParser(description="짝의 RGB 프레임 정밀 재선택")
    ap.add_argument("match")
    ap.add_argument("video")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=2.0, help="탐색 반경 초")
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    obj = object_points(pat, args.cell)
    z = dict(np.load(args.match, allow_pickle=True))
    names = z["names"]
    TH, RG, flip = z["th_corners"], z["rgb_corners"], z["flip"]
    Kt, Dt, Kr, Dr = z["Kt"], z["Dt"], z["Kr"], z["Dr"]
    times = z["rgb_time"].astype(float)
    n = len(names)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 열화상 자세 (순서 정렬 후)
    Rth, Tth = [], []
    for i in range(n):
        c = TH[i]
        if flip[i]:
            c = c.reshape(pat[1], pat[0], 2)[::-1, ::-1].reshape(-1, 2)
        R, t = pose(obj, c, Kt, Dt)
        Rth.append(R)
        Tth.append(t)

    print("=" * 80)
    print(f"짝 {n}개 · RGB 프레임 정밀 재선택 (±{args.window:g}초, "
          f"{fps:.0f} fps 전부)")
    print("=" * 80)

    RGb = [RG[i].copy() for i in range(n)]
    cur = times.copy()
    for it in range(args.iters):
        # 현재 짝으로 기준 R_rel, T 추정
        Rs, Ts = [], []
        for i in range(n):
            R, t = pose(obj, RGb[i], Kr, Dr)
            Rr = Rth[i] @ R.T
            Rs.append(Rr)
            Ts.append(Tth[i] - Rr @ t)
        # 회전 평균은 가장 가운데 있는 것으로 (사원수 평균 대신 medoid)
        med = int(np.argmin([sum(ang(a, b) for b in Rs) for a in Rs]))
        Rref, Tref = Rs[med], np.median(np.array(Ts), axis=0)
        sc = [ang(r, Rref) for r in Rs]
        st = [float(np.linalg.norm(t-Tref)) for t in Ts]
        print(f"\n  [{it+1}회] 기준 R_rel · T = "
              f"({Tref[0]:+.1f}, {Tref[1]:+.1f}, {Tref[2]:+.1f}) "
              f"|T| {np.linalg.norm(Tref):.1f} mm")
        print(f"        현재 산포 — 회전 {np.mean(sc):.2f}° · "
              f"위치 {np.mean(st):.1f} mm")

        moved = 0
        for i in range(n):
            f0 = int(round(cur[i]*fps))
            lo = max(0, f0-int(args.window*fps))
            hi = min(total-1, f0+int(args.window*fps))
            best = (1e9, None, None)
            for fi in range(lo, hi+1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok:
                    continue
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                ok2, c, _, _ = detect(g, [pat], 1)
                if not ok2:
                    continue
                R, t = pose(obj, c.reshape(-1, 2), Kr, Dr)
                if R is None:
                    continue
                Rr = Rth[i] @ R.T
                cost = ang(Rr, Rref) + np.linalg.norm(
                    (Tth[i] - Rr @ t) - Tref)/10.0
                if cost < best[0]:
                    best = (cost, c.reshape(-1, 2), fi/fps)
            if best[1] is not None and abs(best[2]-cur[i]) > 1e-6:
                moved += 1
            if best[1] is not None:
                RGb[i], cur[i] = best[1], best[2]
        print(f"        {moved}/{n} 짝의 프레임을 교체")

    cap.release()
    Rs, Ts = [], []
    for i in range(n):
        R, t = pose(obj, RGb[i], Kr, Dr)
        Rr = Rth[i] @ R.T
        Rs.append(Rr)
        Ts.append(Tth[i] - Rr @ t)
    med = int(np.argmin([sum(ang(a, b) for b in Rs) for a in Rs]))
    Rref, Tref = Rs[med], np.median(np.array(Ts), axis=0)
    print(f"\n  {'자세':<12}{'RGB 시각':>10}{'이동':>9}{'회전차':>9}{'T차':>9}")
    for i in range(n):
        print(f"  {str(names[i])[:12]:<12}{cur[i]:>9.2f}s"
              f"{cur[i]-times[i]:>+8.2f}s{ang(Rs[i], Rref):>8.2f}°"
              f"{np.linalg.norm(Ts[i]-Tref):>8.1f}mm")

    z["rgb_corners"] = np.array(RGb)
    z["rgb_time"] = cur
    np.savez(args.match, **z)
    print(f"\n  갱신: {args.match}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
