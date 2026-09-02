#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시각 창 후보 중 기하가 가장 잘 맞는 RGB 프레임을 골라 스테레오까지.

    python3 scripts/solve_pairs.py output/pair_time/candidates.npz \
        --baseline 47.5 --depth 550

pair_by_time.py 가 시각으로 후보를 좁혀 놓았으므로, 여기서는 각 자세마다
후보 중 상대 회전·베이스라인이 전체와 가장 잘 맞는 하나를 고르면 됩니다.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from calibrate_thermal import object_points, CALIB_FLAGS       # noqa: E402


def ang(A, B):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(A @ B.T)-1)/2, -1, 1))))


def flip(pts, pat):
    return pts.reshape(pat[1], pat[0], 2)[::-1, ::-1].reshape(-1, 2)


def pose(obj, pts, K, D):
    ok, rv, tv = cv2.solvePnP(obj, pts.reshape(-1, 1, 2).astype(np.float32),
                              K, D)
    return (cv2.Rodrigues(rv)[0], tv.reshape(3)) if ok else (None, None)


def main():
    ap = argparse.ArgumentParser(description="후보 선택 + 스테레오")
    ap.add_argument("cands", nargs="?",
                    default="output/pair_time/candidates.npz")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--baseline", type=float, default=None)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--ransac", type=int, default=4,
                    help="RGB 내부 파라미터 강건 적합 시도 횟수")
    ap.add_argument("--rgb-f0", type=float, default=1350.0,
                    help="RGB 초점거리 초기값 px (1920 폭 기준)")
    ap.add_argument("--min-sep", type=float, default=15.0,
                    help="코너가 평균 이만큼(px) 달라야 다른 자세로 본다")
    ap.add_argument("--max-intr", type=int, default=90,
                    help="RGB 내부 파라미터에 쓸 최대 장수")
    ap.add_argument("--tol", type=float, default=3.0, help="회전 허용 도")
    ap.add_argument("--btol", type=float, default=25.0, help="T 허용 mm")
    ap.add_argument("--min-pairs", type=int, default=6)
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    obj = object_points(pat, args.cell)
    z = np.load(args.cands, allow_pickle=True)
    names, TH = z["names"], z["th"]
    CT, CC = z["cand_t"], z["cand_c"]
    Kt = z["Kt"]
    W, H = (int(v) for v in z["size"])
    n = len(names)
    Dt = np.zeros(5)

    print("=" * 88)
    print(f"후보 선택 + 스테레오 · 열화상 자세 {n}개 · RGB {W}x{H}")
    print("=" * 88)

    # ── RGB 내부 파라미터 — 모든 후보에서 다양하게 골라 ──
    allc, allctr = [], []
    for i in range(n):
        for c in CC[i]:
            allc.append(np.asarray(c, np.float32))
            allctr.append(np.asarray(c).mean(0))
    allctr = np.array(allctr)
    # ★ 후보 728장을 그냥 넣거나 '가장 편심된 것부터' 고르면 발산한다
    #   (f 가 4000 을 넘고 주점이 화면 밖으로 나간다). 오검출 몇 장이 왜곡
    #   계수를 끌고 가기 때문이다. RANSAC 식으로 — 무작위 씨앗으로 맞춘 뒤
    #   전수 잔차로 좋은 것만 남겨 다시 맞추기를 반복한다. 이렇게 하면 씨앗을
    #   바꿔도 같은 답으로 수렴한다.
    FL = CALIB_FLAGS | cv2.CALIB_USE_INTRINSIC_GUESS
    K0 = np.array([[args.rgb_f0, 0, W/2.], [0, args.rgb_f0, H/2.],
                   [0, 0, 1.]])
    pts = [c.reshape(-1, 1, 2) for c in allc]

    def resid(K, D):
        out = []
        for p in pts:
            ok, rv, tv = cv2.solvePnP(obj, p, K, D)
            pr, _ = cv2.projectPoints(obj, rv, tv, K, D)
            out.append(float(np.sqrt(
                ((p.reshape(-1, 2)-pr.reshape(-1, 2))**2).sum(axis=1).mean())))
        return np.array(out)

    rng = np.random.default_rng(7)
    best = None
    for _ in range(args.ransac):
        seed = rng.choice(len(pts), min(35, len(pts)), replace=False)
        try:
            r, K, D, _, _ = cv2.calibrateCamera(
                [obj]*len(seed), [pts[k] for k in seed], (W, H),
                K0.copy(), None, flags=FL)
        except cv2.error:
            continue
        if r > 3 or not (40 < 2*np.degrees(np.arctan(W/(2*K[0, 0]))) < 100):
            continue
        sel = []
        for _ in range(3):
            e = resid(K, D)
            gi = np.where(e < max(1.0, 2.5*np.median(e)))[0]
            sel = []
            for k in gi[np.argsort(e[gi])]:
                if all(np.abs(allc[k]-allc[q]).mean() > args.min_sep
                       for q in sel):
                    sel.append(k)
                if len(sel) >= args.max_intr:
                    break
            r, K, D, _, _ = cv2.calibrateCamera(
                [obj]*len(sel), [pts[k] for k in sel], (W, H),
                K0.copy(), None, flags=FL)
        inl = int((resid(K, D) < 1.0).sum())
        if best is None or inl > best[0]:
            best = (inl, r, K, D, len(sel))
    if best is None:
        print("  ✗ RGB 내부 파라미터를 구하지 못했습니다.")
        return 2
    inl, rms_r, Kr, Dr, nsel = best
    fov = 2*np.degrees(np.arctan(W/(2*Kr[0, 0])))
    print(f"  RGB 내부 파라미터 — 후보 {len(allc)}장 → 채택 {nsel}장 "
          f"(전수 inlier {inl}/{len(allc)})")
    print(f"    RMS {rms_r:.3f} px · f {Kr[0,0]:.1f} · 화각 {fov:.1f}° · "
          f"k1 {Dr[0,0]:+.3f} · k2 {Dr[0,1]:+.3f}")
    print(f"    주점 ({Kr[0,2]:.0f}, {Kr[1,2]:.0f})  "
          f"— 화면 중심 ({W/2:.0f}, {H/2:.0f})")
    if rms_r > 2.0 or not (40 < fov < 100):
        print(f"    ✗ RGB 보정이 발산했습니다.")
        return 2

    # ── 자세별 후보의 (R_rel, T) ──
    cand = []
    for i in range(n):
        row = []
        for f in (0, 1):
            cth = TH[i] if f == 0 else flip(TH[i], pat)
            Rt, Tt = pose(obj, cth, Kt, Dt)
            if Rt is None:
                continue
            for j, c in enumerate(CC[i]):
                Rr, Tr = pose(obj, np.asarray(c), Kr, Dr)
                if Rr is None:
                    continue
                Rrel = Rt @ Rr.T
                row.append((j, f, Rrel, Tt - Rrel @ Tr))
        cand.append(row)

    # ── 가장 지지가 많은 가설 ──
    best = (-1, None, None, None)
    for i in range(n):
        for (_, _, Rh, Th) in cand[i]:
            pick, tot = {}, 0.0
            for i2 in range(n):
                b = None
                for (j2, f2, R2, T2) in cand[i2]:
                    dr = ang(R2, Rh)
                    dtv = float(np.linalg.norm(T2-Th))
                    if dr < args.tol and dtv < args.btol:
                        sc = dr + dtv/10
                        if b is None or sc < b[0]:
                            b = (sc, j2, f2, dr, dtv)
                if b:
                    pick[i2] = b[1:]
                    tot += b[0]
            if len(pick) > best[0] or (len(pick) == best[0] and
                                       tot < (best[3] or 1e9)):
                best = (len(pick), Rh, pick, tot)
    cnt, Rh, pick, _ = best
    print(f"\n  하나의 상대 회전으로 설명되는 자세 {cnt}/{n}")
    if cnt < args.min_pairs:
        print("  ✗ 짝이 너무 적습니다.")
        return 2

    idx = sorted(pick)
    P_th, P_rg, used = [], [], []
    for i in idx:
        j, f, dr, dtv = pick[i]
        cth = TH[i] if f == 0 else flip(TH[i], pat)
        P_th.append(np.asarray(cth, np.float32).reshape(-1, 1, 2))
        P_rg.append(np.asarray(CC[i][j], np.float32).reshape(-1, 1, 2))
        used.append((str(names[i]), float(CT[i][j]), dr, dtv))

    rms, *_ , R, T, E, F = cv2.stereoCalibrate(
        [obj]*len(idx), P_th, P_rg, Kt, Dt, Kr, Dr, (160, 120),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                  300, 1e-8))
    # stereoCalibrate 는 X_2 = R·X_1 + T. 열화상이 1번이므로 뒤집는다.
    R, T = R.T, -R.T @ T
    b = float(np.linalg.norm(T))
    print(f"\n  스테레오 RMS {rms:.3f} px · 베이스라인 {b:.1f} mm · "
          f"상대 회전 {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))):.2f}°")
    print(f"  T = ({T[0,0]:+.1f}, {T[1,0]:+.1f}, {T[2,0]:+.1f}) mm")
    if args.baseline:
        print(f"  기구 실측 {args.baseline:g} mm + 간격 → 차이 "
              f"{b-args.baseline:+.1f} mm")

    # ── 자세별 정합 오차 (그 자세의 실제 보드 평면) ──
    print(f"\n  {'자세':<28}{'RGB 시각':>10}{'회전차':>8}{'T차':>8}"
          f"{'정합오차':>10}")
    errs = []
    for k, i in enumerate(idx):
        Rb, tb = pose(obj, np.asarray(CC[i][pick[i][0]]), Kr, Dr)
        nv = Rb[:, 2].reshape(3, 1)
        d = float(nv.ravel() @ tb)
        if d < 0:
            nv, d = -nv, -d
        Hm = Kt @ (R + (T @ nv.T)/d) @ np.linalg.inv(Kr)
        und = cv2.undistortPoints(P_rg[k], Kr, Dr, P=Kr)
        proj = cv2.perspectiveTransform(und, Hm).reshape(-1, 2)
        undt = cv2.undistortPoints(P_th[k], Kt, Dt, P=Kt).reshape(-1, 2)
        e = float(np.sqrt(((undt-proj)**2).sum(axis=1).mean()))
        errs.append(e)
        nm, tt, dr, dtv = used[k]
        print(f"  {nm[:28]:<28}{tt%3600/60:>7.2f}분{dr:>7.2f}°{dtv:>7.1f}mm"
              f"{e:>9.2f}px")
    print(f"\n  정합 오차 평균 {np.mean(errs):.2f} px · "
          f"중앙값 {np.median(errs):.2f} px  (목표 < 1.5)")

    out = os.path.dirname(args.cands)
    np.savez(os.path.join(out, "stereo_final.npz"), R=R, T=T, Kt=Kt, Dt=Dt,
             Kr=Kr, Dr=Dr, rms=rms, errs=np.array(errs),
             names=np.array([u[0] for u in used]),
             rgb_t=np.array([u[1] for u in used]),
             th_corners=np.array([p.reshape(-1, 2) for p in P_th]),
             rgb_corners=np.array([p.reshape(-1, 2) for p in P_rg]),
             size_rgb=np.array([W, H]))
    if args.depth:
        nv = np.array([[0.], [0.], [1.]])
        Hd = Kt @ (R + (T @ nv.T)/args.depth) @ np.linalg.inv(Kr)
        print(f"\n  기준 깊이 {args.depth:g} mm 호모그래피 저장")
        np.savez(os.path.join(out, "homography.npz"), H=Hd,
                 depth=args.depth, Kt=Kt, Dt=Dt, Kr=Kr, Dr=Dr, R=R, T=T)
    print(f"  저장: {out}/stereo_final.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
