#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""열화상 ↔ RGB 캘리브레이션 — 내부 파라미터 + 평면 호모그래피.

폴더 구조 (같은 이름끼리 짝을 맞춘다):

    calib/
      th/    pose01.y16raw   pose02.y16raw  ...
      rgb/   pose01.png      pose02.png     ...

실행:

    python3 calibrate_pair.py calib/ --baseline 50 --depth 457

산출:
    calib_result.json   K·왜곡계수·호모그래피·오차
    overlay/            자세별 코너 검출 확인 이미지

두 카메라를 각각 왜곡 보정한 뒤 호모그래피를 구합니다. 열화상을 보정하지 않고
호모그래피만 걸면 열화상 렌즈 왜곡이 그대로 오차로 남습니다.
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

import check_board                              # noqa: E402
from check_board import (detect, detect_file, load_any, to8,  # noqa: E402
                         short)


# ── 대응점 수집 ───────────────────────────────────────────────────
def object_points(pat, cell_mm):
    o = np.zeros((pat[0]*pat[1], 3), np.float32)
    o[:, :2] = np.mgrid[0:pat[0], 0:pat[1]].T.reshape(-1, 2)
    return o * cell_mm


CALIB_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
RZ180 = np.diag([-1.0, -1.0, 1.0])     # 보드 프레임 Z축 180° 회전


def _ang(Ra, Rb):
    """두 회전 사이 각도 (도)."""
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(Ra @ Rb.T)[0])))


def resolve_ordering(obj, pt, pr, sh_th, sh_rgb, names, tol=10.0):
    """열화상 코너 순서를 RGB 와 맞추고, 짝이 잘못된 자세를 걸러낸다.

    ★ 왜 필요한가
      판을 데워 찍으면 열화상은 검정 칸이 '밝게', RGB 는 '어둡게' 나온다.
      체커보드 검출기는 명암 극성으로 시작 코너를 정하므로 두 카메라의 코너
      순서가 180° 어긋난다. 그대로 스테레오에 넣으면 상대 회전이 179° 로
      나오면서 결과가 통째로 망가진다.

    ★ 어떻게 판별하는가
      한 자세만 놓고 보면 판별이 불가능하다. 평면 격자는 180° 돌려 대응시켜도
      호모그래피가 성립하므로 잔차로는 구분되지 않는다.

      대신 기하를 쓴다. 두 카메라는 단단히 붙어 있으므로 **카메라간 상대 회전은
      모든 자세에서 같아야 한다.** 자세마다 '뒤집지 않음 / 뒤집음' 두 후보의
      상대 회전을 구하고, 가장 많은 자세가 모이는 군집을 기준으로 삼는다.
      실측에서 일치도가 0.5° 수준으로 떨어져 판별이 명확하다.

      부수 효과로 **파일 짝이 뒤바뀐 자세도 잡힌다** — 다른 자세의 RGB 와
      짝지어지면 상대 회전이 군집에서 크게 벗어난다.
    """
    n = len(obj)
    _, _, _, rv_t, _ = cv2.calibrateCamera(obj, pt, sh_th, None, None,
                                           flags=CALIB_FLAGS)
    _, _, _, rv_r, _ = cv2.calibrateCamera(obj, pr, sh_rgb, None, None,
                                           flags=CALIB_FLAGS)
    cand = []
    for i in range(n):
        Rt = cv2.Rodrigues(rv_t[i])[0]
        Rr = cv2.Rodrigues(rv_r[i])[0]
        cand.append([Rt @ Rr.T, (Rt @ RZ180) @ Rr.T])

    best_cnt, ref = -1, None
    for i in range(n):
        for f in (0, 1):
            c = cand[i][f]
            cnt = sum(1 for j in range(n)
                      if min(_ang(c, cand[j][0]), _ang(c, cand[j][1])) < tol)
            if cnt > best_cnt:
                best_cnt, ref = cnt, c

    out, flips, bad, devs = [], [], [], []
    for j in range(n):
        a0, a1 = _ang(ref, cand[j][0]), _ang(ref, cand[j][1])
        use_flip = a1 < a0
        dev = min(a0, a1)
        devs.append(dev)
        if dev > tol:
            bad.append(names[j])
            out.append(None)
            continue
        out.append(pt[j][::-1].copy() if use_flip else pt[j])
        if use_flip:
            flips.append(names[j])

    print(f"\n  코너 순서 정렬 — 카메라간 상대 회전의 일치도로 판별")
    print(f"    기준 군집에 든 자세 {best_cnt}/{n} · "
          f"일치 편차 중앙값 {np.median(devs):.2f}° (최대 "
          f"{max(d for d in devs if d <= tol) if any(d <= tol for d in devs) else 0:.2f}°)")
    if flips:
        print(f"    순서를 뒤집은 자세 {len(flips)}개 — 명암 극성이 반대라 정상입니다")
    if bad:
        print(f"    ✗ 군집에서 벗어난 자세 {len(bad)}개 — 제외합니다: "
              f"{', '.join(bad[:6])}{' …' if len(bad) > 6 else ''}")
        print(f"      th/ 와 rgb/ 의 짝이 뒤바뀌었을 가능성이 큽니다. 확인하십시오.")
    return out, bad


def collect(root, pat, cell, upscale, save_dir):
    th_dir, rgb_dir = os.path.join(root, "th"), os.path.join(root, "rgb")
    for d in (th_dir, rgb_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"폴더가 없습니다: {d}")

    def index(d):
        """이름(확장자 제외) → 파일 경로.

        같은 이름에 여러 확장자가 있으면 y16raw > 이미지 > 영상 순으로 고른다.
        녹화기는 pose01.y16raw 와 함께 .y16meta/.avi/.csv 를 만드는데,
        .y16meta·.csv 는 제외하고 .avi 보다 .y16raw 를 쓴다.
        """
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
        return {k: v[1] for k, v in rank.items()}

    th, rgb = index(th_dir), index(rgb_dir)
    names = sorted(set(th) & set(rgb))
    only = (set(th) ^ set(rgb))
    if only:
        print(f"  ⚠ 짝이 없는 파일 {len(only)}개는 건너뜁니다: "
              f"{', '.join(sorted(only)[:5])}{' …' if len(only) > 5 else ''}")
    if not names:
        raise SystemExit(
            "짝이 맞는 자세가 없습니다.\n"
            "  th/ 와 rgb/ 의 파일명(확장자 제외)을 같게 하십시오.\n"
            "  예: th/pose01.y16raw  ↔  rgb/pose01.png")

    obj = object_points(pat, cell)
    O, PT, PR, used = [], [], [], []
    sh_th = sh_rgb = None
    print(f"  {'자세':<24}{'열화상':>8}{'RGB':>8}{'칸(px)':>9}{'대비':>9}")
    for n in names:
        ok_t, c_t, p_t, _, cel_t, g_t, _ = detect_file(th[n], [pat], upscale)
        ok_r, c_r, p_r, _, _, g_r, _ = detect_file(rgb[n], [pat], 1)
        sh_th, sh_rgb = g_t.shape[::-1], g_r.shape[::-1]
        mark_t = "OK" if ok_t else "실패"
        mark_r = "OK" if ok_r else "실패"

        sq = dt = None
        if ok_t:
            from check_board import square_px, contrast_c
            sq = square_px(c_t, pat)
            dt = contrast_c(cel_t, c_t, pat)

        keep = ok_t and ok_r
        print(f"  {short(n):<24}{mark_t:>8}{mark_r:>8}"
              f"{(f'{sq:.1f}' if sq else '-'):>9}"
              f"{(f'{dt:.2f}℃' if dt else '-'):>9}")

        if keep:
            O.append(obj); PT.append(c_t.astype(np.float32))
            PR.append(c_r.astype(np.float32)); used.append(n)
            if save_dir:
                for g, c, tag in ((g_t, c_t, "th"), (g_r, c_r, "rgb")):
                    v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                    s = 5 if g.shape[1] < 400 else 1
                    if s > 1:
                        v = cv2.resize(v, None, fx=s, fy=s,
                                       interpolation=cv2.INTER_NEAREST)
                    cv2.drawChessboardCorners(v, pat, c*s, True)
                    cv2.imwrite(os.path.join(save_dir, f"{n}_{tag}.png"), v)

    return O, PT, PR, used, sh_th, sh_rgb


# ── 캘리브레이션 ──────────────────────────────────────────────────
def calibrate(obj, pts, size, name, fix_tangential=True, want_pose=False):
    flags = cv2.CALIB_ZERO_TANGENT_DIST if fix_tangential else 0
    flags |= cv2.CALIB_FIX_K3
    rms, K, D, rv, tv = cv2.calibrateCamera(obj, pts, size, None, None, flags=flags)
    per = []
    for i in range(len(obj)):
        proj, _ = cv2.projectPoints(obj[i], rv[i], tv[i], K, D)
        per.append(float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - pts[i].reshape(-1, 2))**2, axis=1)))))
    print(f"\n  [{name}] RMS 재투영 오차 {rms:.3f} px   "
          f"(자세별 {min(per):.2f} ~ {max(per):.2f})")
    print(f"    f = ({K[0,0]:.1f}, {K[1,1]:.1f})   중심 = ({K[0,2]:.1f}, {K[1,2]:.1f})")
    print(f"    왜곡 k1={D[0,0]:+.4f}  k2={D[0,1]:+.4f}")
    return (rms, K, D, per, rv, tv) if want_pose else (rms, K, D, per)


def main():
    ap = argparse.ArgumentParser(description="열화상↔RGB 캘리브레이션")
    ap.add_argument("root", help="calib/ 폴더 (안에 th/ 와 rgb/)")
    ap.add_argument("--pattern", default="7x4", help="내부 코너 수 (기본 7x4)")
    ap.add_argument("--cell", type=float, default=30.0, help="칸 크기 mm")
    ap.add_argument("--upscale", type=int, default=1, help="열화상 검출 확대")
    ap.add_argument("--baseline", type=float, default=None,
                    help="두 카메라 렌즈 간 거리 mm (시차 오차 계산용)")
    ap.add_argument("--depth", type=float, default=457.0,
                    help="정합 기준 깊이 mm. ★ RGB 렌즈 ~ 잎 표면 거리를 넣으십시오 "
                         "(평면 법선을 RGB 좌표계에서 정의하므로). 열화상 렌즈 "
                         "기준값을 넣으면 베이스라인의 Z 성분만큼 어긋납니다")
    ap.add_argument("--frames", type=int, default=12,
                    help="열화상 평균 프레임 수. 손으로 들고 찍었으면 3 (기본 12)")
    ap.add_argument("--out", default="calib_result.json")
    ap.add_argument("--overlay", default="overlay")
    args = ap.parse_args()

    check_board.NFRAMES = args.frames

    w, h = (int(v) for v in args.pattern.lower().split("x"))
    pat = (w, h)
    os.makedirs(args.overlay, exist_ok=True)

    print("=" * 80)
    print(f"대응점 수집 · 패턴 {w}x{h} · 칸 {args.cell:g} mm · OpenCV {cv2.__version__}")
    print("=" * 80)
    obj, pt, pr, used, sh_th, sh_rgb = collect(
        args.root, pat, args.cell, args.upscale, args.overlay)

    n = len(used)
    print(f"\n  양쪽 모두 검출된 자세 {n}개")
    if n < 8:
        print("  ✗ 자세가 부족합니다. 최소 8개, 권장 15개 이상.")
        return 1
    if n < 15:
        print("  ⚠ 15개 미만입니다. 왜곡 계수 추정이 불안정할 수 있습니다.")

    fixed, bad = resolve_ordering(obj, pt, pr, sh_th, sh_rgb, used)
    if bad:
        good = [i for i in range(n) if fixed[i] is not None]
        obj = [obj[i] for i in good]
        pr = [pr[i] for i in good]
        used = [used[i] for i in good]
        pt = [fixed[i] for i in good]
        n = len(used)
        print(f"\n  제외 후 자세 {n}개")
        if n < 8:
            print("  ✗ 남은 자세가 너무 적습니다.")
            return 1
    else:
        pt = fixed

    print("\n" + "=" * 80)
    print("내부 파라미터")
    print("=" * 80)
    rms_t, K_t, D_t, per_t = calibrate(obj, pt, sh_th, f"열화상 {sh_th[0]}x{sh_th[1]}")
    rms_r, K_r, D_r, per_r, rv_r, tv_r = calibrate(
        obj, pr, sh_rgb, f"RGB {sh_rgb[0]}x{sh_rgb[1]}", want_pose=True)

    # ── 스테레오 외부 파라미터 (RGB → 열화상) ──
    print("\n" + "=" * 80)
    print("스테레오 외부 파라미터")
    print("=" * 80)
    srms, *_, R, T, E, F = cv2.stereoCalibrate(
        obj, pr, pt, K_r, D_r, K_t, D_t, sh_th,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7))
    base = float(np.linalg.norm(T))
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    print(f"  스테레오 RMS {srms:.3f} px")
    print(f"  베이스라인(추정) {base:.1f} mm   T = "
          f"({T[0,0]:+.1f}, {T[1,0]:+.1f}, {T[2,0]:+.1f}) mm")
    print(f"  두 카메라 상대 회전 {ang:.2f}°")
    if args.baseline:
        print(f"  실측 베이스라인 {args.baseline:g} mm 와 차이 "
              f"{abs(base-args.baseline):.1f} mm"
              + ("  ✓" if abs(base-args.baseline) < 0.15*args.baseline else
                 "  ⚠ 10% 이상 어긋남 — 실측치나 촬영을 재확인하십시오"))

    # ── 평면 호모그래피 ──
    #   X_th = R·X_rgb + T,  평면은 RGB 좌표계에서 nᵀ·X_rgb = d
    #   ⇒ X_th = (R + T·nᵀ/d)·X_rgb  →  H = K_t (R + T·nᵀ/d) K_r⁻¹
    #   (부호 주의: 교과서의 R − t·nᵀ/d 는 평면을 nᵀX + d = 0 으로 두는 규약)
    Kr_inv = np.linalg.inv(K_r)
    nvec = np.array([[0.0], [0.0], [1.0]])          # RGB 광축에 수직인 평면

    def H_at(d):
        return K_t @ (R + (T @ nvec.T)/d) @ Kr_inv

    ut = [cv2.undistortPoints(p, K_t, D_t, P=K_t).reshape(-1, 2) for p in pt]
    ur = [cv2.undistortPoints(p, K_r, D_r, P=K_r).reshape(-1, 2) for p in pr]
    depth = []
    for i in range(n):
        Rb = cv2.Rodrigues(rv_r[i])[0]
        depth.append(float((Rb @ obj[i].T + tv_r[i])[2].mean()))

    def err_of(H, i):
        q = cv2.perspectiveTransform(ur[i].reshape(-1, 1, 2), H).reshape(-1, 2)
        return np.linalg.norm(q - ut[i], axis=1)

    H0 = H_at(args.depth)
    print("\n" + "=" * 80)
    print(f"평면 호모그래피 — 기준 깊이 {args.depth:g} mm")
    print("=" * 80)
    print(f"  {'자세':<24}{'깊이(mm)':>10}{'H(기준)오차':>13}{'H(그 깊이)오차':>15}")
    e_ref, e_own = [], []
    for i in range(n):
        a = err_of(H0, i).mean()
        b = err_of(H_at(depth[i]), i).mean()
        e_ref.append(a); e_own.append(b)
        near = "  ← 기준면 부근" if abs(depth[i]-args.depth) <= 20 else ""
        print(f"  {short(used[i]):<24}{depth[i]:>10.0f}{a:>12.2f}px{b:>14.2f}px{near}")
    e_ref, e_own = np.array(e_ref), np.array(e_own)
    near = np.abs(np.array(depth)-args.depth) <= 30
    print(f"\n  전체 평균      H(기준) {e_ref.mean():.2f} px · "
          f"H(각 깊이) {e_own.mean():.2f} px")
    if near.any():
        print(f"  기준면 ±30 mm  H(기준) {e_ref[near].mean():.2f} px  "
              f"({int(near.sum())}개 자세)  ← 실제 운용 정확도")
    else:
        lo, hi = min(depth), max(depth)
        gap = (args.depth-hi) if args.depth > hi else (lo-args.depth)
        print(f"  ⚠ 기준 깊이 ±30 mm 안에 촬영된 자세가 없습니다 "
              f"(자세 {lo:.0f}~{hi:.0f} mm, 외삽 {gap:.0f} mm).")
        print(f"     식물을 건드릴 수 없어 보드를 잎 표면까지 못 가져간 경우입니다.")
        print(f"     H(d) 는 스테레오 R·T 에서 수식으로 유도되므로 계산 자체는 됩니다.")
        print(f"     다만 그 깊이에서의 실측 검증이 없으니, 위 'H(그 깊이) 오차' 가")
        print(f"     여러 깊이에서 작게 나오는지로 깊이 모델을 간접 확인하십시오.")
    home_err = float(e_ref[near].mean()) if near.any() else float(e_ref.mean())

    # ── 판정 ──
    print("\n" + "=" * 80)
    print("합격 판정")
    print("=" * 80)
    checks = [
        ("자세 수 ≥ 15", n >= 15, f"{n}개"),
        ("열화상 RMS < 0.5 px", rms_t < 0.5, f"{rms_t:.3f} px (목표 0.3)"),
        ("RGB RMS < 1.0 px", rms_r < 1.0, f"{rms_r:.3f} px (목표 0.5)"),
        ("스테레오 RMS < 1.0 px", srms < 1.0, f"{srms:.3f} px"),
        ("기준면 호모그래피 < 1.5 px", home_err < 1.5, f"{home_err:.2f} px"),
        ("기준 깊이가 자세 범위 안 또는 ±50 mm",
         min(depth)-50 <= args.depth <= max(depth)+50,
         f"자세 {min(depth):.0f}~{max(depth):.0f} mm · 기준 {args.depth:.0f} mm"),
    ]
    for label, ok, val in checks:
        print(f"  {'✓' if ok else '✗'}  {label:<28}{val}")
    allok = all(c[1] for c in checks)

    # ── 시차 감도 (실측 R,T 기반) ──
    print("\n" + "=" * 80)
    print("깊이에 따른 정합 오차 — 실측 외부 파라미터로 계산")
    print("=" * 80)
    ctr = np.array([[sh_rgb[0]/2], [sh_rgb[1]/2], [1.0]])
    for d in (350, 400, 457, 500, 550, 600):
        q0 = H0 @ ctr; q1 = H_at(d) @ ctr
        e = float(np.linalg.norm(q0[:2]/q0[2] - q1[:2]/q1[2]))
        print(f"    깊이 {d:>3} mm : {e:5.2f} px"
              + ("   ← 기준면" if abs(d-args.depth) < 1 else ""))
    f_px = float(K_t[0, 0])
    tol = f_px*base/args.depth**2
    print(f"\n  1 px 을 지키려면 깊이가 ±{1/tol:.0f} mm 안에 있어야 합니다.")
    print(f"  (오차 ≈ f_px·b·|1/d0 − 1/d|,  f_px {f_px:.0f} · b {base:.0f} mm)")

    out = {
        "pattern": [w, h], "cell_mm": args.cell, "poses": used,
        "thermal": {"size": list(sh_th), "K": K_t.tolist(),
                    "dist": D_t.ravel().tolist(), "rms_px": rms_t},
        "rgb": {"size": list(sh_rgb), "K": K_r.tolist(),
                "dist": D_r.ravel().tolist(), "rms_px": rms_r},
        "stereo": {"R": R.tolist(), "T": T.ravel().tolist(),
                   "baseline_mm": base, "rms_px": srms},
        "H_rgb_to_thermal_at_reference_depth": H0.tolist(),
        "homography_err_px": {"reference_plane_mean": home_err,
                              "all_poses_mean": float(e_ref.mean())},
        "pose_depth_mm": depth,
        "reference_depth_mm": args.depth,
        "baseline_mm": args.baseline,
        "pass": bool(allok),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {args.out} · 코너 확인 이미지: {args.overlay}/")
    print("\n  " + ("전 항목 합격 — 이 값으로 고정하십시오."
                    if allok else "미달 항목이 있습니다. 위 판정표를 보고 보완 촬영하십시오."))

    print("""
  사용법 (RGB 마스크를 열화상 좌표로 옮길 때)

    import cv2, numpy as np, json
    c = json.load(open('calib_result.json'))
    K_r = np.array(c['rgb']['K']);     D_r = np.array(c['rgb']['dist'])
    K_t = np.array(c['thermal']['K']); D_t = np.array(c['thermal']['dist'])
    H   = np.array(c['H_rgb_to_thermal_at_reference_depth'])

    rgb_und  = cv2.undistort(rgb_img, K_r, D_r)          # ① RGB 왜곡 보정
    mask_und = cv2.undistort(mask,    K_r, D_r)          #    마스크도 동일하게
    mask_th  = cv2.warpPerspective(mask_und, H, (160, 120),
                                   flags=cv2.INTER_NEAREST)   # ② 열화상 좌표로
    # ③ 열화상 원본은 왜곡이 남아 있으므로, 비교할 열화상도 undistort 해서 쓰십시오
    th_und = cv2.undistort(th_img, K_t, D_t)
""")
    return 0 if allok else 2


if __name__ == "__main__":
    sys.exit(main())
