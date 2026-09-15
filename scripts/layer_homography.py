#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""캐노피 층마다 RGB → 열화상 호모그래피를 낸다.

    python3 scripts/layer_homography.py output/rgb_verify/stereo.npz \
        --depths 420 700 1000 --out output/layer_H

`H(d) = K_t (R + T nᵀ/d) K_r⁻¹` 는 깊이 d 에 대한 해석식이므로, 스테레오를
한 번 풀어 두면 층마다 d 만 바꿔 넣으면 됩니다.

같이 내는 것
    · 층별 GSD 와 시야 크기
    · 한 층의 H 를 다른 층에 잘못 쓰면 몇 화소 어긋나는지
    · RGB 화면에서 열화상이 보는 영역 (어노테이션 범위)

**RGB 좌표는 먼저 왜곡을 풀어야 합니다** — `cv2.undistortPoints(..., P=K)`.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

F_TH = 147.4
TH_W, TH_H = 160, 120


def main():
    ap = argparse.ArgumentParser(description="층별 호모그래피")
    ap.add_argument("stereo", help="verify_rgb.py 가 낸 stereo.npz")
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[420, 700, 1000])
    ap.add_argument("--out", default="output/layer_H")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.stereo)
    R, T, K, D, Kt = z["R"], z["T"], z["K"], z["D"], z["Kt"]
    b = float(np.linalg.norm(T))
    n = np.array([[0.], [0.], [1.]])           # 층을 정면 평면으로 본다

    print("=" * 84)
    print(f"층별 호모그래피 · 베이스라인 {b:.1f} mm · f_th {Kt[0,0]:.1f} px")
    print("=" * 84)
    out = {}
    print(f"  {'깊이':>7}{'GSD':>11}{'가로시야':>11}{'세로시야':>11}"
          f"{'크라운150mm':>12}")
    for d in args.depths:
        H = Kt @ (R + (T @ n.T)/d) @ np.linalg.inv(K)
        gsd = d/F_TH
        out[str(int(d))] = dict(H=H.tolist(), gsd_mm_per_px=round(gsd, 3),
                                fov_w_mm=round(TH_W*gsd, 1),
                                fov_h_mm=round(TH_H*gsd, 1),
                                crown150_px=round(150/gsd, 1))
        print(f"  {d:6.0f}mm{gsd:9.2f}mm/px{TH_W*gsd:9.0f}mm"
              f"{TH_H*gsd:9.0f}mm{150/gsd:10.0f}px")

    print(f"\n  층을 잘못 짚으면 몇 화소 어긋나는가  (f·b·|1/d0 − 1/d|)")
    hdr = "".join(f"{d:>10.0f}" for d in args.depths)
    print(f"  {'기준\\실제':<12}{hdr}")
    for d0 in args.depths:
        row = "".join(f"{F_TH*b*abs(1/d0-1/d):>10.1f}" for d in args.depths)
        print(f"  {d0:>8.0f} mm  {row}")

    # RGB 화면에서 열화상이 보는 영역
    print(f"\n  RGB 화면에서 열화상이 덮는 넓이")
    W, H_ = 1920, 1080
    if "size" in z:
        W, H_ = (int(v) for v in z["size"])
    edge = ([[a, 0] for a in range(0, TH_W, 4)]
            + [[TH_W-1, b_] for b_ in range(0, TH_H, 4)]
            + [[a, TH_H-1] for a in range(TH_W-1, -1, -4)]
            + [[0, b_] for b_ in range(TH_H-1, -1, -4)])
    for d in args.depths:
        Hm = np.array(out[str(int(d))]["H"])
        poly = cv2.perspectiveTransform(
            np.array(edge, np.float32).reshape(-1, 1, 2),
            np.linalg.inv(Hm)).reshape(-1, 2)
        m = np.zeros((H_, W), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        frac = float(m.mean()/255*100)
        out[str(int(d))]["rgb_coverage_pct"] = round(frac, 1)
        out[str(int(d))]["rgb_polygon"] = [[round(float(a), 1),
                                            round(float(b_), 1)]
                                           for a, b_ in poly]
        print(f"  {d:6.0f}mm   RGB 화면의 {frac:5.1f} %")

    json.dump(dict(baseline_mm=round(b, 2), f_thermal_px=F_TH,
                   R=R.tolist(), T=T.ravel().tolist(),
                   K_rgb=K.tolist(), dist_rgb=D.ravel().tolist(),
                   K_thermal=Kt.tolist(), layers=out,
                   note="RGB 좌표는 undistortPoints(..., P=K) 로 왜곡을 푼 뒤 "
                        "H 를 적용할 것. 열화상은 RGB 대비 180도 회전 장착."),
              open(os.path.join(args.out, "layer_homography.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  저장 {args.out}/layer_homography.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
