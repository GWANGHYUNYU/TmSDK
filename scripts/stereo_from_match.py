#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""매칭된 짝으로 스테레오 보정 + 깊이별 평면 호모그래피.

    python3 scripts/stereo_from_match.py output/pair_match/match.npz \
        --depth 550 --baseline 47.5
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from calibrate_thermal import object_points                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="매칭 결과로 스테레오 보정")
    ap.add_argument("match", default="output/pair_match/match.npz", nargs="?")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--depth", type=float, default=None,
                    help="정합 기준 깊이 mm (RGB 좌표계, 렌즈~잎 표면)")
    ap.add_argument("--baseline", type=float, default=None,
                    help="실측 베이스라인 mm (교차 검증용)")
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    obj = object_points(pat, args.cell)
    z = np.load(args.match, allow_pickle=True)
    names = z["names"]
    TH = z["th_corners"].astype(np.float32)
    RG = z["rgb_corners"].astype(np.float32)
    flip = z["flip"]
    Kt, Dt, Kr, Dr = z["Kt"], z["Dt"], z["Kr"], z["Dr"]
    Wr, Hr = (int(v) for v in z["size_rgb"])
    n = len(names)

    # 코너 순서 정렬 — 열화상이 RGB 대비 180° 돌아가 있어 순서가 뒤집힌다
    P_th = []
    for i in range(n):
        c = TH[i]
        if flip[i]:
            c = c.reshape(pat[1], pat[0], 2)[::-1, ::-1].reshape(-1, 2)
        P_th.append(c.astype(np.float32).reshape(-1, 1, 2))
    P_rg = [RG[i].reshape(-1, 1, 2) for i in range(n)]

    print("=" * 80)
    print(f"스테레오 보정 · 짝 {n}개")
    print("=" * 80)
    print(f"  열화상 160x120 · f {Kt[0,0]:.1f} (줄자로 고정)")
    print(f"  RGB {Wr}x{Hr} · f {Kr[0,0]:.1f} · k1 {Dr[0,0]:+.3f}")

    rms, Kt2, Dt2, Kr2, Dr2, R, T, E, F = cv2.stereoCalibrate(
        [obj]*n, P_th, P_rg, Kt, Dt, Kr, Dr, (160, 120),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                  200, 1e-7))
    # ★ stereoCalibrate 는 X_2 = R·X_1 + T 를 돌려준다. 열화상을 1번으로
    #   넘겼으므로 반환값은 X_rgb = R·X_th + T 다. 우리가 쓰는 호모그래피는
    #   RGB → 열화상 방향(X_th = R'·X_rgb + T')이므로 뒤집어야 한다.
    R, T = R.T, -R.T @ T
    b = float(np.linalg.norm(T))
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))
    print(f"\n  스테레오 RMS   {rms:.3f} px")
    print(f"  베이스라인      {b:.1f} mm   T = "
          f"({T[0,0]:+.1f}, {T[1,0]:+.1f}, {T[2,0]:+.1f})")
    print(f"  상대 회전       {rot:.2f}°")
    if args.baseline:
        d = abs(b-args.baseline)
        print(f"  실측 {args.baseline:g} mm 와 차이 {d:.1f} mm "
              f"({d/args.baseline*100:.0f} %)"
              + ("  ✓" if d < 0.15*args.baseline else "  ✗"))

    # ── 깊이별 평면 호모그래피 ──
    # H(d) = K_t (R + T·nᵀ/d) K_r⁻¹   — 부호는 plus (평면 nᵀX = d)
    if args.depth:
        nv = np.array([[0.], [0.], [1.]])
        H = Kt @ (R + (T @ nv.T)/args.depth) @ np.linalg.inv(Kr)
        print(f"\n  기준 깊이 {args.depth:g} mm 에서의 호모그래피")
        # 그 깊이에 있던 자세로 실측 오차 확인
        errs = []
        for i in range(n):
            okp, rv, tv = cv2.solvePnP(obj, P_rg[i], Kr, Dr)
            dz = float(((cv2.Rodrigues(rv)[0] @ obj.T).T+tv.reshape(3))[:, 2]
                       .mean())
            und_r = cv2.undistortPoints(P_rg[i], Kr, Dr, P=Kr)
            proj = cv2.perspectiveTransform(und_r, H)
            und_t = cv2.undistortPoints(P_th[i], Kt, Dt, P=Kt)
            e = float(np.sqrt(((und_t.reshape(-1, 2)-proj.reshape(-1, 2))**2)
                              .sum(axis=1).mean()))
            errs.append((dz, e, names[i]))
        # ★ 모델 자체를 평가하려면 **그 자세의 실제 보드 평면**을 써야 한다.
        #   보드가 기울어져 있으므로 정면 평면(n=(0,0,1), z=d)으로 재면 기울임
        #   때문에 생기는 오차가 통째로 섞여 들어와 모델을 평가할 수 없다.
        #   평면은 nᵀX = d, n = 보드 z축(RGB 좌표계), d = n·t.
        own = []
        for i in range(n):
            okp, rv, tv = cv2.solvePnP(obj, P_rg[i], Kr, Dr)
            Rb = cv2.Rodrigues(rv)[0]
            ni = Rb[:, 2].reshape(3, 1)
            di = float(ni.ravel() @ tv.reshape(3))
            if di < 0:                      # 법선이 카메라 반대편을 보면 뒤집기
                ni, di = -ni, -di
            Hi = Kt @ (R + (T @ ni.T)/di) @ np.linalg.inv(Kr)
            und_r = cv2.undistortPoints(P_rg[i], Kr, Dr, P=Kr)
            proj = cv2.perspectiveTransform(und_r, Hi)
            und_t = cv2.undistortPoints(P_th[i], Kt, Dt, P=Kt)
            own.append(float(np.sqrt(
                ((und_t.reshape(-1, 2)-proj.reshape(-1, 2))**2)
                .sum(axis=1).mean())))
        print(f"  {'자세':<12}{'RGB 깊이':>10}{"실평면 오차":>13}"
              f"{'기준깊이 오차':>14}{'예상 시차':>11}")
        for i in range(n):
            dz, e, nm = errs[i]
            par = Kt[0, 0]*b*abs(1/dz - 1/args.depth)
            print(f"  {str(nm)[:12]:<12}{dz:>9.0f}mm{own[i]:>13.2f}px"
                  f"{e:>13.2f}px{par:>10.1f}px")
        print(f"\n  실제 보드 평면 평균 오차 {np.mean(own):.2f} px "
              f"— 이것이 **모델 자체의 정확도**입니다 (목표 < 1.5)")
        print(f"  기준 깊이 오차는 깊이 차이에서 오는 시차를 포함합니다.")
        np.savez(os.path.join(os.path.dirname(args.match), "stereo.npz"),
                 R=R, T=T, Kt=Kt, Dt=Dt, Kr=Kr, Dr=Dr, H=H,
                 depth=args.depth, rms=rms)
        print(f"  저장: {os.path.dirname(args.match)}/stereo.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
