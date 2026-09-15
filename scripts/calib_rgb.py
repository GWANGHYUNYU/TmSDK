#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB 카메라 내부 파라미터 재산출 (왜곡 포함).

    python3 scripts/calib_rgb.py calib/2026-08-28/rgb --out output/rgb_intrinsics

이전 값 `f = 1299.5` 는 왜곡을 넣지 않고 푼 것이고, 같은 판을 재면 열화상
(f = 147.4, 줄자 검증)보다 9.8 % 멀게 나옵니다. 여기서는

    · 영상 전체를 훑어 코너를 모으고
    · 화면을 고르게 덮도록 자세를 고른 뒤
    · k1 k2 p1 p2 까지 포함해 풀고
    · 한 자세씩 잔차를 보며 나쁜 것을 떨어뜨립니다

마지막에 **화면 귀퉁이가 실제로 덮였는지**를 함께 냅니다. 귀퉁이가 비면
왜곡 계수는 외삽이라 믿을 수 없습니다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import detect                             # noqa: E402

PAT = (7, 4)
CELL = 30.0


def object_points(pat, cell):
    o = np.zeros((pat[0]*pat[1], 3), np.float32)
    o[:, :2] = np.mgrid[0:pat[0], 0:pat[1]].T.reshape(-1, 2)*cell
    return o


def scan(folder, step_sec, coarse_w, cache):
    """영상들을 훑어 코너를 모은다 → [(영상, 프레임, 코너)]."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"  (이전 결과 사용: {cache})")
        return list(z["items"]), tuple(z["size"])
    items, size = [], None
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        p = os.path.join(folder, f)
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or n <= 0:
            cap.release()
            print(f"  {f:<18}열 수 없음 — 건너뜁니다")
            continue
        size = (W, H)
        sc = coarse_w/W
        stride = max(1, int(round(fps*step_sec)))
        hit = 0
        for fi in range(0, n, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gs = cv2.resize(g, None, fx=sc, fy=sc,
                            interpolation=cv2.INTER_AREA)
            if not detect(gs, [PAT], 1)[0]:
                continue
            ok2, c, _, _ = detect(g, [PAT], 1)
            if ok2:
                items.append((f, fi, c.reshape(-1, 2).astype(np.float32)))
                hit += 1
        cap.release()
        print(f"  {f:<18}{n} 프레임 · {n/fps:.0f}초  →  코너 {hit}개")
    np.savez(cache, items=np.array(items, dtype=object),
             size=np.array(size), allow_pickle=True)
    return items, size


def pick(items, size, want, gx=4, gy=3):
    """화면을 고르게 덮도록 고른다 — 보드 중심 격자 × 보드 크기 3단계."""
    W, H = size
    buckets = {}
    for it in items:
        c = it[2]
        cx, cy = c.mean(0)
        area = cv2.contourArea(cv2.convexHull(c))
        s = min(2, int(np.sqrt(area/(W*H))*6))        # 크기 3단계
        k = (int(cx/W*gx), int(cy/H*gy), s)
        buckets.setdefault(k, []).append(it)
    out, rr = [], 0
    while len(out) < want:
        added = False
        for k in sorted(buckets):
            if len(buckets[k]) > rr:
                out.append(buckets[k][rr])
                added = True
                if len(out) >= want:
                    break
        if not added:
            break
        rr += 1
    return out, len(buckets)


def solve(sel, size, flags, guess):
    obj = [object_points(PAT, CELL) for _ in sel]
    img = [it[2].reshape(-1, 1, 2) for it in sel]
    K = np.array([[guess, 0, size[0]/2], [0, guess, size[1]/2], [0, 0, 1.]])
    D = np.zeros(5)
    rms, K, D, rv, tv = cv2.calibrateCamera(
        obj, img, size, K, D, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                  100, 1e-7))
    per = []
    for i in range(len(sel)):
        pr, _ = cv2.projectPoints(obj[i], rv[i], tv[i], K, D)
        per.append(float(np.sqrt(((pr.reshape(-1, 2)-sel[i][2])**2)
                                 .sum(1)).mean()))
    return rms, K, D, np.array(per), rv, tv


def plausible(c):
    """격자가 기하적으로 말이 되는가 — 오검출을 미리 떨군다.

    투시 때문에 칸 간격이 달라지긴 하지만 3배를 넘지는 않는다. 넘으면
    체커보드가 아닌 것을 잡았거나 코너 순서가 엉킨 것이다.
    """
    g = c.reshape(PAT[1], PAT[0], 2)
    d = np.concatenate([
        np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
        np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])
    return d.min() > 1.0 and d.max()/d.min() < 3.0


def refine(sel, size, flags, guess, tag, keep_min=15, rounds=8):
    """중앙값 기준으로 나쁜 자세를 떨구며 다시 푼다 (평균+2σ 는 이상치에 끌린다)."""
    cur = list(sel)
    rms = K = D = None
    for it in range(rounds):
        rms, K, D, per, _, _ = solve(cur, size, flags, guess)
        guess = K[0, 0]
        med = np.median(per)
        mad = np.median(np.abs(per-med))*1.4826 + 1e-9
        nk = [cur[i] for i in range(len(cur)) if per[i] <= med + 3*mad]
        print(f"    {tag} {it+1}회차  자세 {len(cur):>3}개  RMS {rms:6.3f} px  "
              f"f {K[0,0]:7.1f}  (중앙잔차 {med:.3f})")
        if len(nk) < keep_min or len(nk) == len(cur):
            break
        cur = nk
    return rms, K, D, cur


def coverage(sel, size, gx=8, gy=6):
    W, H = size
    seen = np.zeros((gy, gx), bool)
    for it in sel:
        for x, y in it[2]:
            seen[min(gy-1, int(y/H*gy)), min(gx-1, int(x/W*gx))] = True
    return seen


def main():
    ap = argparse.ArgumentParser(description="RGB 내부 파라미터")
    ap.add_argument("folder")
    ap.add_argument("--step", type=float, default=0.5, help="훑는 간격 초")
    ap.add_argument("--coarse-width", type=int, default=640)
    ap.add_argument("--views", type=int, default=40)
    ap.add_argument("--guess", type=float, default=1200.0)
    ap.add_argument("--out", default="output/rgb_intrinsics")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 88)
    print(f"RGB 내부 파라미터 재산출 · {args.folder}")
    print("=" * 88)
    items, size = scan(args.folder, args.step, args.coarse_width,
                       os.path.join(args.out, "corners.npz"))
    if len(items) < 12:
        raise SystemExit(f"코너가 {len(items)}개뿐입니다 — 너무 적습니다.")
    print(f"\n  코너 총 {len(items)}자세 · 해상도 {size[0]}x{size[1]}")

    items = [it for it in items if plausible(it[2])]
    print(f"  격자가 기하적으로 말이 되는 것 {len(items)}자세")
    sel, nb = pick(items, size, args.views)
    print(f"  화면을 덮도록 {len(sel)}자세 선정 (채워진 구획 {nb}개)")

    models = [
        ("f 하나 · 왜곡 없음",
         cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
         | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3),
        ("f 하나 · k1",
         cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
         | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3),
        ("f 하나 · k1 k2",
         cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
         | cv2.CALIB_FIX_K3),
        ("fx fy · k1 k2 p1 p2",
         cv2.CALIB_FIX_K3),
    ]
    # ① 먼저 단순한 모델(f + k1)로 나쁜 자세를 떨군다. 이상치가 섞인 채로
    #    모델을 고르면 가장 자유도 높은 모델이 이상치를 흡수해 버린다.
    print("\n  ① 이상치 제거 (f + k1)")
    simple = (cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
              | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    _, K0, _, sel = refine(sel, size, simple, args.guess, "")
    print(f"    남은 자세 {len(sel)}개")

    # ② 깨끗해진 자세로 모델을 고른다
    print(f"\n  ② 모델 비교 (같은 {len(sel)}자세)")
    print(f"  {'모델':<22}{'RMS':>8}{'fx':>9}{'fy':>9}{'cx':>8}{'cy':>8}"
          f"{'k1':>9}{'k2':>9}{'귀퉁이보정':>11}")
    W, H = size
    pts = np.array([[[0., 0.]], [[W-1., 0.]], [[0., H-1.]], [[W-1., H-1.]]],
                   np.float32)
    best = None
    for nm, fl in models:
        try:
            r, Km, Dm, _, _, _ = solve(sel, size, fl, K0[0, 0])
        except cv2.error as e:
            print(f"  {nm:<22}실패 {e}")
            continue
        d_ = Dm.ravel()
        und = cv2.undistortPoints(pts, Km, Dm, P=Km).reshape(-1, 2)
        cor = float(np.linalg.norm(und-pts.reshape(-1, 2), axis=1).mean())
        print(f"  {nm:<22}{r:8.3f}{Km[0,0]:9.1f}{Km[1,1]:9.1f}"
              f"{Km[0,2]:8.1f}{Km[1,2]:8.1f}{d_[0]:9.4f}{d_[1]:9.4f}"
              f"{cor:9.0f}px")
        # 자유도를 늘려 RMS 가 3 % 넘게 줄 때만 갈아탄다
        if best is None or r < best[0]*0.97:
            best = (r, Km, Dm, nm, fl)
    rms, K, D, name, fl = best
    print(f"\n  채택: {name}  (RMS 3 % 이상 개선될 때만 자유도를 늘림)")
    rms, K, D, sel = refine(sel, size, fl, K[0, 0], "  최종")

    seen = coverage(sel, size)
    W, H = size
    fov_x = 2*np.degrees(np.arctan(W/2/K[0, 0]))
    fov_y = 2*np.degrees(np.arctan(H/2/K[1, 1]))
    print("\n" + "=" * 88)
    print(f"  자세 {len(sel)}개 · 재투영 RMS {rms:.3f} px")
    print(f"  fx {K[0,0]:.1f}  fy {K[1,1]:.1f}  cx {K[0,2]:.1f}  cy {K[1,2]:.1f}")
    d_ = D.ravel()
    print(f"  왜곡 k1 {d_[0]:+.4f}  k2 {d_[1]:+.4f}  "
          f"p1 {d_[2]:+.5f}  p2 {d_[3]:+.5f}")
    print(f"  화각 가로 {fov_x:.1f}도 · 세로 {fov_y:.1f}도")
    print(f"  화면 덮임 {int(seen.sum())}/{seen.size} 구획")
    for r in seen:
        print("    " + "".join("#" if v else "." for v in r))
    # 귀퉁이가 비면 왜곡은 외삽이다
    corners_seen = int(seen[0, 0]) + int(seen[0, -1]) + int(seen[-1, 0]) + int(seen[-1, -1])
    if corners_seen < 4:
        print(f"  ⚠ 네 귀퉁이 중 {corners_seen}곳만 덮였습니다 — "
              f"왜곡 계수는 그만큼 덜 믿을 값입니다")
    # 화면 가장자리에서 왜곡이 만드는 이동량
    pts = np.array([[[0., 0.]], [[W-1., 0.]], [[0., H-1.]], [[W-1., H-1.]],
                    [[W/2, 0.]], [[W/2, H-1.]]], np.float32)
    und = cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)
    d = np.linalg.norm(und - pts.reshape(-1, 2), axis=1)
    print(f"  왜곡 보정량  귀퉁이 {d[:4].mean():.1f} px · "
          f"위아래 변 중앙 {d[4:].mean():.1f} px")

    np.savez(os.path.join(args.out, "rgb_intrinsics.npz"),
             K=K, D=D, size=np.array(size, int), rms=rms,
             nview=len(sel))
    json.dump(dict(fx=float(K[0, 0]), fy=float(K[1, 1]),
                   cx=float(K[0, 2]), cy=float(K[1, 2]),
                   dist=[float(v) for v in D.ravel()],
                   width=int(W), height=int(H), rms=float(rms), views=int(len(sel)),
                   fov_x_deg=round(fov_x, 2), fov_y_deg=round(fov_y, 2),
                   model=name),
              open(os.path.join(args.out, "rgb_intrinsics.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  저장 {args.out}/rgb_intrinsics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
