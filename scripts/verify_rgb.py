#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""새 RGB 내부 파라미터를 열화상과 맞대어 검증한다.

    python3 scripts/verify_rgb.py output/rgb_intrinsics/rgb_intrinsics.npz \
        calib/2026-08-28 --out output/rgb_verify

검증의 근거는 **열화상 f = 147.4 px 가 줄자 실측으로 확정된 값**이라는
것입니다. 같은 순간 같은 판을 두 카메라로 재면 거리가 같아야 합니다.
그 조건만으로 RGB 초점거리가 독립적으로 결정됩니다.

이어서
    · 스테레오를 풀어 ‖T‖ 가 캘리퍼스 실측 51.9 mm 와 맞는지
    · 그 결과로 만든 H(d) 를 남겨 둔 쌍에 적용했을 때 몇 화소 어긋나는지
를 봅니다.

**열화상은 RGB 대비 180° 회전 장착이므로 먼저 돌립니다.**
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import detect, to8                        # noqa: E402

PAT = (7, 4)
CELL = 30.0
F_TH = 147.4
BASELINE_MM = 51.9


def obj_pts():
    o = np.zeros((PAT[0]*PAT[1], 3), np.float32)
    o[:, :2] = np.mgrid[0:PAT[0], 0:PAT[1]].T.reshape(-1, 2)*CELL
    return o


def secs(s):
    return (int(s[11:13])*3600 + int(s[14:16])*60 + float(s[17:])
            if len(s) > 10 else None)


def thermal_poses(path):
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    ts = json.load(open(os.path.splitext(path)[0]+".y16meta",
                        encoding="utf-8")).get("timestamps") or []
    out = []
    for i in range(arr.shape[0]):
        if i >= len(ts):
            break
        cels = conv(np.asarray(arr[i])).astype(np.float64)[::-1, ::-1]
        ok, c, pt, _ = detect(to8(cels), [PAT])
        if ok:
            out.append((secs(ts[i]), c.reshape(-1, 2).astype(np.float32)))
    return out


def rgb_at(video, times, s0, span=2):
    """필요한 시각 근처만 본다.

    구간 전체를 훑으면 12개 녹화에 1만 프레임이 넘어 몇십 분이 걸린다.
    짝을 맞출 시각은 이미 알고 있으므로 그 앞뒤 ±span 프레임만 본다.
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, done = [], set()
    for t in times:
        c0 = int(round((t-s0)*fps))
        for fi in range(max(0, c0-span), min(n-1, c0+span)+1):
            if fi in done:
                continue
            done.add(fi)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gs = cv2.resize(g, None, fx=1/3, fy=1/3,
                            interpolation=cv2.INTER_AREA)
            if not detect(gs, [PAT], 1)[0]:
                continue
            ok2, c, _, _ = detect(g, [PAT], 1)
            if ok2:
                out.append((s0+fi/fps, c.reshape(-1, 2).astype(np.float32)))
    cap.release()
    return sorted(out)


def dist(c, K, D=None):
    ok, rv, tv = cv2.solvePnP(obj_pts(), c.reshape(-1, 1, 2), K,
                              D if D is not None else None)
    return float(np.linalg.norm(tv)), rv, tv


def orient(ct, cr, Kt, K, D):
    """RGB 코너 순서의 180° 모호성을 푼다.

    7×4 격자는 점 집합이 중심대칭이라 180° 돌려 대응시켜도 **호모그래피가
    똑같이 잘 맞습니다.** 그래서 재투영 잔차로는 가를 수 없습니다.

    대신 판의 자세를 봅니다. 두 카메라는 52 mm 떨어져 거의 나란하므로
    판의 회전행렬이 양쪽에서 거의 같아야 합니다. 순서가 뒤집히면 판 법선을
    축으로 180° 돌아간 자세가 나오므로 금방 갈립니다.
    """
    _, rvt, _ = dist(ct, Kt)
    Rt = cv2.Rodrigues(rvt)[0]
    best = None
    for flip, c in ((False, cr), (True, cr[::-1])):
        _, rvr, _ = dist(c, K, D)
        Rr = cv2.Rodrigues(rvr)[0]
        ang = float(np.degrees(np.arccos(np.clip(
            (np.trace(Rt @ Rr.T)-1)/2, -1, 1))))
        if best is None or ang < best[0]:
            best = (ang, c, flip)
    return best[1], best[0], best[2]


def main():
    ap = argparse.ArgumentParser(description="RGB 내부 파라미터 검증")
    ap.add_argument("npz")
    ap.add_argument("session", help="calib/2026-08-28 처럼 thermal/ rgb/ 가 있는 폴더")
    ap.add_argument("--max-dt", type=float, default=0.05, help="허용 시각차 초")
    ap.add_argument("--per-rec", type=int, default=8,
                    help="녹화 하나에서 쓸 자세 수")
    ap.add_argument("--max-pose-diff", type=float, default=25.0,
                    help="두 카메라가 본 판 자세의 허용 차이(도)")
    ap.add_argument("--max-blur", type=float, default=2.0,
                    help="시각차 동안 판이 움직인 양의 상한 (RGB 화소)")
    ap.add_argument("--out", default="output/rgb_verify")
    ap.add_argument("--cache", action="store_true",
                    help="짝을 저장/재사용 — 변형을 빨리 시험할 때")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.npz)
    K, D, size = z["K"], z["D"], tuple(int(v) for v in z["size"])
    Kt = np.array([[F_TH, 0, 79.5], [0, F_TH, 59.5], [0, 0, 1.]])
    print("=" * 88)
    print(f"RGB 내부 파라미터 검증 · fx {K[0,0]:.1f} fy {K[1,1]:.1f} "
          f"k1 {D.ravel()[0]:+.4f} k2 {D.ravel()[1]:+.4f}")
    print("=" * 88)

    vids = []
    for f in sorted(os.listdir(os.path.join(args.session, "rgb"))):
        m = re.match(r"(\d{2})-(\d{2})-(\d{2})", f)
        if not m:
            continue
        p = os.path.join(args.session, "rgb", f)
        cap = cv2.VideoCapture(p)
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or n <= 0:
            continue
        s0 = sum(int(v)*k for v, k in zip(m.groups(), (3600, 60, 1)))
        vids.append((p, s0, n/fps))

    cpath = os.path.join(args.out, 'pairs.npz')
    if args.cache and os.path.exists(cpath):
        cz = np.load(cpath, allow_pickle=True)
        pairs = [(a, b, c) for a, b, c in cz['pairs']]
        nflip, speeds = int(cz['nflip']), list(cz['speeds'])
        print(f'  (저장된 짝 {len(pairs)}개 사용)')
    else:
        pairs, nflip, speeds = _collect(args, vids, K, D, Kt)
        if args.cache:
            np.savez(cpath, pairs=np.array(pairs, dtype=object),
                     nflip=nflip, speeds=speeds, allow_pickle=True)
    _rest(args, pairs, nflip, speeds, K, D, Kt, size)
    return 0


def _collect(args, vids, K, D, Kt):
    pairs, nflip, speeds = [], 0, []
    for tp in sorted(glob.glob(os.path.join(args.session, "thermal",
                                            "*.y16raw"))):
        nm = os.path.basename(tp)[:-7]
        tposes = thermal_poses(tp)
        if not tposes:
            continue
        # 한 녹화에서 자세는 몇 개면 충분하다. 고르게 솎는다.
        if len(tposes) > args.per_rec:
            idx = np.linspace(0, len(tposes)-1, args.per_rec).astype(int)
            tposes = [tposes[i] for i in idx]
        t0 = tposes[0][0]
        vid = next(((p, s0) for p, s0, d in vids if s0 <= t0 <= s0+d), None)
        if vid is None:
            print(f"  {nm:<28}겹치는 RGB 없음")
            continue
        rposes = rgb_at(vid[0], [t for t, _ in tposes], vid[1])
        got = 0
        for tt, ct in tposes:
            if not rposes:
                break
            j = int(np.argmin([abs(r[0]-tt) for r in rposes]))
            if abs(rposes[j][0]-tt) > args.max_dt:
                continue
            cr, ang, flip = orient(ct, rposes[j][1], Kt, K, D)
            if ang > args.max_pose_diff:
                continue               # 자세가 이만큼 다르면 짝이 틀린 것
            # ★ 판이 움직이는 중이면 시각차 몇 십 ms 가 몇 화소로 번진다.
            #   앞뒤 RGB 검출과 비교해 이동 속도를 재고, 빠른 것은 버린다.
            spd = 0.0
            for k in (j-1, j+1):
                if 0 <= k < len(rposes):
                    dtt = abs(rposes[k][0]-rposes[j][0])
                    if dtt > 1e-6:
                        spd = max(spd, float(np.linalg.norm(
                            rposes[k][1]-rposes[j][1], axis=1).mean())/dtt)
            if spd*abs(rposes[j][0]-tt) > args.max_blur:
                continue
            pairs.append((nm, ct, cr))
            speeds.append(spd)
            nflip += flip
            got += 1
        print(f"  {nm:<28}열화상 {len(tposes):>3} · RGB {len(rposes):>3}"
              f"  →  짝 {got}")
    if len(pairs) < 4:
        raise SystemExit(f"짝이 {len(pairs)}개뿐입니다.")
    return pairs, nflip, speeds


def _rest(args, pairs, nflip, speeds, K, D, Kt, size):
    print(f"\n  짝 {len(pairs)}개")

    # ── ① 같은 판의 거리가 두 카메라에서 같은가 ──
    print("\n  ① 같은 판까지의 거리 — 열화상(f=147.4, 줄자 검증) 대비")
    rows = {}
    for nm, ct, cr in pairs:
        dt_, _, _ = dist(ct, Kt)
        dr, _, _ = dist(cr, K, D)
        rows.setdefault(nm, []).append((dt_, dr))
    rat = []
    for nm in sorted(rows):
        a = np.array(rows[nm])
        r = float(np.median(a[:, 1]/a[:, 0]))
        rat.append(r)
        print(f"    {nm:<28}열화상 {np.median(a[:,0]):5.0f} mm · "
              f"RGB {np.median(a[:,1]):5.0f} mm   비 {r:6.3f}")
    rat = np.array(rat)
    print(f"\n    비 중앙 {np.median(rat):.4f}  (1.000 이면 완전 일치)")
    # 거리는 초점거리에 비례한다. RGB 거리가 비만큼 멀게 나왔다면 쓴 초점거리가
    # 그만큼 큰 것이므로 **나눈다**.
    print(f"    → 거리를 맞추는 f_rgb = {K[0,0]/np.median(rat):.1f} px "
          f"(지금 값 {K[0,0]:.1f})")

    # ── ② 스테레오 ──
    obj = [obj_pts() for _ in pairs]
    ith = [p[1].reshape(-1, 1, 2) for p in pairs]
    irg = [p[2].reshape(-1, 1, 2) for p in pairs]
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj, irg, ith, K, D, Kt, np.zeros(5), size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                  200, 1e-8))
    print(f"\n  ② 스테레오 — RGB 를 1번으로 넣었으므로 X_th = R X_rgb + T")
    print(f"    RMS {rms:.3f} px")
    print(f"    ‖T‖ {np.linalg.norm(T):.1f} mm   "
          f"T = ({T[0,0]:+.1f}, {T[1,0]:+.1f}, {T[2,0]:+.1f})")
    print(f"    캘리퍼스 실측 {BASELINE_MM} mm "
          f"→ 차이 {np.linalg.norm(T)-BASELINE_MM:+.1f} mm")
    ang = np.degrees(cv2.Rodrigues(R)[0].ravel())
    print(f"    회전 (pitch, yaw, roll) = "
          f"({ang[0]:+.2f}, {ang[1]:+.2f}, {ang[2]:+.2f}) 도")

    # ── ③ H(d) 를 실제 쌍에 적용했을 때 ──
    print("\n  ③ H(d) 정합 오차 — 각 쌍의 실제 판 깊이로")
    errs = []
    nv = np.array([[0.], [0.], [1.]])
    for nm, ct, cr in pairs:
        d_, rv, tv = dist(cr, K, D)
        Rb = cv2.Rodrigues(rv)[0]
        n = Rb[:, 2:3]
        dd = float((n.T @ tv).ravel()[0])
        if dd < 0:
            n, dd = -n, -dd
        H = Kt @ (R + (T @ n.T)/dd) @ np.linalg.inv(K)
        und = cv2.undistortPoints(cr.reshape(-1, 1, 2), K, D, P=K)
        pr = cv2.perspectiveTransform(und, H).reshape(-1, 2)
        errs.append(float(np.sqrt(((pr-ct)**2).sum(1)).mean()))
    errs = np.array(errs)
    print(f"    중앙 {np.median(errs):.2f} px · 평균 {errs.mean():.2f} · "
          f"최대 {errs.max():.2f}  (n={len(errs)})")
    np.savez(os.path.join(args.out, "stereo.npz"), R=R, T=T, K=K, D=D, Kt=Kt,
             rms=rms, err=errs)
    print(f"\n  저장 {args.out}/stereo.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
