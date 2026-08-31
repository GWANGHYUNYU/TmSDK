#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시각 동기 없이 열화상 자세 ↔ RGB 프레임을 기하로 짝짓는다.

    python3 scripts/match_rgb_thermal.py calib/_all output/rgb_scan.npz \
        --focal 147.4 --out output/pair_match

원리
    두 카메라가 강체로 붙어 있으면 **상대 회전 R_rel = R_th · R_rgbᵀ 가 모든
    자세에서 같습니다.** 올바른 짝에서는 R_rel 이 한 점에 모이고, 틀린 짝에서는
    흩어집니다. 이 성질로 타임스탬프 없이 짝을 찾고 동시에 검증합니다.

    체커보드는 평면이라 180° 뒤집힌 코너 순서도 똑같이 잘 맞습니다. 그래서
    자세마다 두 순서를 모두 후보로 두고, 가장 많은 지지를 받는 R_rel 을
    고릅니다 (calibrate_pair.py 의 resolve_ordering 과 같은 논리).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

import check_board as CB                                          # noqa: E402
from check_board import (detect_file, detect, load_candidates,     # noqa: E402
                         short)
from calibrate_thermal import object_points, CALIB_FLAGS, gather   # noqa: E402

RZ180 = np.array([[-1., 0, 0], [0, -1., 0], [0, 0, 1.]])


def ang(A, B):
    """두 회전 사이 각도 (도)."""
    c = (np.trace(A @ B.T)-1)/2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def flip_order(pts, pat):
    """코너 순서를 180° 뒤집는다."""
    return pts.reshape(pat[1], pat[0], 2)[::-1, ::-1].reshape(-1, 2)


def thermal_poses(src, pat, cell, focal):
    """열화상 자세별 최적 프레임과 R, t."""
    CB.NFRAMES = 1
    obj = object_points(pat, cell)
    files = gather(src)
    names, P = [], []
    for n, p in files.items():
        ok, c = detect_file(p, [pat])[:2]
        if ok:
            names.append(n)
            P.append(c.astype(np.float32))
    Kg = np.array([[focal, 0, 80.], [0, focal, 60.], [0, 0, 1.]])
    rms, K, D, rv, tv = cv2.calibrateCamera(
        [obj]*len(P), P, (160, 120), Kg, None,
        flags=(CALIB_FLAGS | cv2.CALIB_FIX_FOCAL_LENGTH
               | cv2.CALIB_USE_INTRINSIC_GUESS))
    # 프레임 재선택 — 가장 선명한 것이 코너가 가장 정확한 것은 아니다
    for i, n in enumerate(names):
        keep, CB.PROBE = CB.PROBE, 40
        try:
            cands = load_candidates(files[n])
        except Exception:
            CB.PROBE = keep
            continue
        CB.PROBE = keep
        best = (float("inf"), None)
        for _, g8 in cands:
            ok2, c2, _, _ = detect(g8, [pat])
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
            P[i] = best[1]
    rms, K, D, rv, tv = cv2.calibrateCamera(
        [obj]*len(P), P, (160, 120), Kg, None,
        flags=(CALIB_FLAGS | cv2.CALIB_FIX_FOCAL_LENGTH
               | cv2.CALIB_USE_INTRINSIC_GUESS))
    return names, P, K, D, rv, tv, rms


def main():
    ap = argparse.ArgumentParser(description="열화상 ↔ RGB 기하 매칭")
    ap.add_argument("th_dir")
    ap.add_argument("rgb_scan", help="scan_rgb.py 가 만든 npz")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--focal", type=float, default=147.4)
    ap.add_argument("--tol", type=float, default=6.0,
                    help="상대 회전 일치 허용 각 (도)")
    ap.add_argument("--bmin", type=float, default=15.0,
                    help="베이스라인 하한 mm (물리적으로 가능한 범위)")
    ap.add_argument("--bmax", type=float, default=200.0,
                    help="베이스라인 상한 mm")
    ap.add_argument("--btol", type=float, default=40.0,
                    help="베이스라인 벡터 일치 허용 mm")
    ap.add_argument("--out", default="output/pair_match")
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    obj = object_points(pat, args.cell)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 84)
    print("열화상 ↔ RGB 기하 매칭 — 시각 동기 없이")
    print("=" * 84)

    # ── 열화상 ──
    names, PT, Kt, Dt, rvt, tvt, rms_t = thermal_poses(
        args.th_dir, pat, args.cell, args.focal)
    print(f"  열화상 {len(names)}자세 · RMS {rms_t:.3f} px · "
          f"f {Kt[0,0]:.1f} (고정)")

    # ── RGB 내부 파라미터 ──
    z = np.load(args.rgb_scan, allow_pickle=True)
    RC = z["corners"].astype(np.float32)
    RT = z["idx"]/float(z["fps"])
    W, H = (int(v) for v in z["size"])
    # 자세가 몰려 있으면 과적합하므로 중심이 흩어진 것부터 고른다
    ctr = z["ctrs"]
    order = np.argsort(-np.linalg.norm(ctr-ctr.mean(0), axis=1))
    sel = order[:min(40, len(order))]
    rms_r, Kr, Dr, _, _ = cv2.calibrateCamera(
        [obj]*len(sel), [RC[i].reshape(-1, 1, 2) for i in sel], (W, H),
        None, None, flags=CALIB_FLAGS)
    fov = 2*np.degrees(np.arctan(W/(2*Kr[0, 0])))
    print(f"  RGB {len(RC)}프레임 중 {len(sel)}장으로 보정 · RMS {rms_r:.3f} px")
    print(f"    {W}x{H} · f {Kr[0,0]:.1f} · 화각 {fov:.1f}° · k1 {Dr[0,0]:+.3f}")

    # ── 자세별 회전 + 위치 ──
    # ★ 회전만으로는 제약이 약하다. 사람이 각 구간에서 비슷한 각도로 판을
    #   쓸어 지나가므로, 엉뚱한 프레임도 R_rel 을 만족시킨다. 위치까지 써야
    #   한다: 같은 순간이면 T = t_th − R_rel·t_rgb 가 베이스라인(≈50 mm)이다.
    #   틀린 짝은 수백 mm 가 나온다.
    Rth, Tth = [], []
    for i in range(len(names)):
        row_R, row_T = [], []
        for f in (0, 1):
            pts = (PT[i] if f == 0 else flip_order(PT[i], pat)
                   ).astype(np.float32).reshape(-1, 1, 2)
            okp, rv, tv = cv2.solvePnP(obj, pts, Kt, Dt)
            row_R.append(cv2.Rodrigues(rv)[0])
            row_T.append(tv.reshape(3))
        Rth.append(row_R)
        Tth.append(row_T)
    Rrgb, Trgb = [], []
    for j in range(len(RC)):
        okp, rv, tv = cv2.solvePnP(obj, RC[j].reshape(-1, 1, 2), Kr, Dr)
        Rrgb.append(cv2.Rodrigues(rv)[0] if okp else None)
        Trgb.append(tv.reshape(3) if okp else None)

    # ── 지지도가 가장 높은 상대 회전 찾기 ──
    # 후보가 (자세 x 2순서 x RGB프레임) 이라 순진하게 짜면 수백만 회 루프가
    # 된다. trace(A·Bᵀ) = Σ(A∘B) 이므로 (N,9) 행렬곱 하나로 전부 계산한다.
    print(f"\n  상대 회전 후보를 투표로 고릅니다 (허용 {args.tol:g}°)")
    cand, tcand, owner = [], [], []
    for i in range(len(names)):
        for f in (0, 1):
            for j in range(len(RC)):
                if Rrgb[j] is None:
                    continue
                Rr = Rth[i][f] @ Rrgb[j].T
                cand.append(Rr.ravel())
                tcand.append(Tth[i][f] - Rr @ Trgb[j])
                owner.append((i, j, f))
    Cm, Tm = np.array(cand), np.array(tcand)
    owner = np.array(owner)
    N = len(Cm)

    # 1차 걸러내기 — 베이스라인 크기가 물리적으로 말이 되는가
    nb = np.linalg.norm(Tm, axis=1)
    plaus = (nb > args.bmin) & (nb < args.bmax)
    print(f"    후보 {N}개 → 베이스라인 {args.bmin:g}~{args.bmax:g} mm 인 것 "
          f"{int(plaus.sum())}개")
    if plaus.sum() < 4:
        print("  ✗ 물리적으로 가능한 후보가 거의 없습니다.")
        return 2

    tr = Cm @ Cm.T
    dR = np.degrees(np.arccos(np.clip((tr-1)/2, -1, 1)))
    dT = np.linalg.norm(Tm[:, None, :] - Tm[None, :, :], axis=2)
    ti = owner[:, 0]

    # 가설 h: R_rel 과 T 가 동시에 맞고, 시간 순서가 단조인 조합
    order_th = np.argsort([n for n in names])      # 파일명 = 시각 순
    rank_th = {int(v): k for k, v in enumerate(order_th)}
    best = (-1, None, None, None)
    for h in np.where(plaus)[0]:
        ok = plaus & (dR[h] < args.tol) & (dT[h] < args.btol)
        if not ok.any():
            continue
        pick = {}
        for i2 in np.unique(ti[ok]):
            m = np.where(ok & (ti == i2))[0]
            b = m[np.argmin(dR[h][m] + dT[h][m]/10.0)]
            pick[int(i2)] = (int(owner[b, 1]), int(owner[b, 2]),
                             float(dR[h][b]), float(dT[h][b]))
        # 시간 순서 단조성 — 열화상 순서와 RGB 시각 순서가 같아야 한다
        seq = sorted(pick, key=lambda i: rank_th[i])
        ts = [RT[pick[i][0]] for i in seq]
        mono = all(ts[k] <= ts[k+1] + 1e-9 for k in range(len(ts)-1))
        score = len(pick) + (0.5 if mono else -5)
        if score > best[0]:
            best = (score, Cm[h].reshape(3, 3), pick, (mono, Tm[h]))
    _, hyp, pick, (mono, Tbase) = best
    cnt = len(pick)
    print(f"    채택 가설 — 베이스라인 {np.linalg.norm(Tbase):.1f} mm · "
          f"시간 순서 {'단조 ✓' if mono else '어긋남 ✗'}")
    print(f"  → {cnt}/{len(names)} 자세가 하나의 상대 회전으로 설명됩니다")

    if cnt < 4:
        print("\n  ✗ 짝을 찾지 못했습니다. 두 영상이 같은 촬영이 아니거나,")
        print("    RGB 가 열화상 구간을 담고 있지 않습니다.")
        return 2

    print(f"\n  {'열화상 자세':<12}{'RGB 시각':>10}{'회전 불일치':>12}"
          f"{'RGB 칸px':>10}")
    for i in sorted(pick, key=lambda k: names[k]):
        j, f, d, dt = pick[i]
        print(f"  {short(names[i], 12):<12}{RT[j]:>9.1f}s{d:>8.2f}°"
              f"{dt:>8.1f}mm{z['sqs'][j]:>10.1f}"
              + ("   [순서뒤집힘]" if f else ""))
    unmatched = [names[i] for i in range(len(names)) if i not in pick]
    if unmatched:
        print(f"\n  짝을 못 찾은 자세 {len(unmatched)}개: "
              f"{', '.join(short(u, 10) for u in unmatched)}")

    np.savez(os.path.join(args.out, "match.npz"),
             names=np.array([names[i] for i in sorted(pick)]),
             th_corners=np.array([PT[i].reshape(-1, 2) for i in sorted(pick)]),
             rgb_corners=np.array([RC[pick[i][0]].reshape(-1, 2)
                                   for i in sorted(pick)]),
             rgb_time=np.array([RT[pick[i][0]] for i in sorted(pick)]),
             flip=np.array([pick[i][1] for i in sorted(pick)]),
             baseline=Tbase,
             Kt=Kt, Dt=Dt, Kr=Kr, Dr=Dr, size_rgb=np.array([W, H]))
    print(f"\n  저장: {args.out}/match.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
