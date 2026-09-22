#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB 개체 어노테이션 패키지 — 확정 파라미터 · 층별 투영.

    python3 scripts/make_annotate.py \
        calib/raw/rgb/26-09-18/08-28-31.mp4 \
        calib/raw/th/raw_output/192_168_0_151_20260918_083000.y16raw \
        --out output/annotate_v2

열화상이 **동시에 찍힌 구간**에서만 프레임을 뽑습니다. 그래야 칠한 것을
열화상 좌표로 투영해 **정답 마스크**로 쓸 수 있습니다.

기존 `make_rgb_annotate.py` 는 폐기된 값으로 만들어져 있었습니다
(깊이 450 mm 단일 층 · roll 6.6° · f_rgb 1299.5). 이 스크립트는
[`params/`](../params/README.md) 의 확정값만 씁니다.

내는 것
    frames/       여기에 칠합니다 (1920×1080 PNG 원본)
    reference/    열화상 화각 경계 + 좌표 격자. 눈으로만 확인
    thermal/      **같은 순간의 열화상** — 무엇이 보이는지 대조용
    index.json    프레임별 시각·짝지은 열화상 프레임·화각 다각형
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

LAYERS = (420.0, 700.0, 1000.0)
COLS = {420.0: (60, 220, 255), 700.0: (80, 220, 80), 1000.0: (255, 150, 60)}


def fov_polygon(K, D, Kt, R, T, d, step=4):
    """깊이 d 에서 열화상 테두리가 RGB 화면의 어디에 오는가."""
    nv = np.array([[0.], [0.], [1.]])
    Hm = Kt @ (R + (T @ nv.T)/d) @ np.linalg.inv(K)
    e = ([[a, 0] for a in range(0, 160, step)]
         + [[159, b] for b in range(0, 120, step)]
         + [[a, 119] for a in range(159, -1, -step)]
         + [[0, b] for b in range(119, -1, -step)])
    pts = np.array(e, np.float32).reshape(-1, 1, 2)
    # 열화상 → (왜곡 푼) RGB
    und = cv2.perspectiveTransform(pts, np.linalg.inv(Hm)).reshape(-1, 1, 2)
    # 왜곡 푼 좌표를 실제 RGB 화소로 되돌린다
    K1 = np.linalg.inv(K)
    n = cv2.convertPointsToHomogeneous(und.reshape(-1, 2)).reshape(-1, 3) @ K1.T
    n = n[:, :2]/n[:, 2:3]
    dist = cv2.projectPoints(np.hstack([n, np.zeros((len(n), 1))]).astype(np.float64),
                             np.zeros(3), np.zeros(3), K, D)[0]
    return dist.reshape(-1, 2)


def thermal_frames(path):
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    ts = json.load(open(os.path.splitext(path)[0]+".y16meta",
                        encoding="utf-8")).get("timestamps") or []
    secs = [int(s[11:13])*3600 + int(s[14:16])*60 + float(s[17:]) for s in ts]
    return arr, conv, np.array(secs)


def to8(x):
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x-lo)/(hi-lo+1e-9)*255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="층별 어노테이션 패키지")
    ap.add_argument("video")
    ap.add_argument("thermal", help="같은 시각의 .y16raw")
    ap.add_argument("--params", default="params/thermal_rgb_stereo.npz")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out", default="output/annotate_v2")
    args = ap.parse_args()
    for d in ("frames", "reference", "thermal"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    z = np.load(os.path.join(HERE, args.params))
    K, D, Kt, R, T = z["K"], z["D"], z["Kt"], z["R"], z["T"]
    b = float(np.linalg.norm(T))

    arr, conv, tsec = thermal_frames(args.thermal)
    base = os.path.basename(args.video)
    h, m, s = (int(v) for v in base[:8].split("-"))
    s0 = h*3600 + m*60 + s
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lo = max(s0, tsec[0])
    hi = min(s0 + n/fps, tsec[-1])
    print("=" * 84)
    print(f"어노테이션 패키지 · 베이스라인 {b:.2f} mm · {W}x{Hh}")
    print("=" * 84)
    print(f"  RGB      {base}  {s0//3600}:{s0%3600//60:02d}:{s0%60:02d} "
          f"~ +{n/fps:.0f}초")
    print(f"  열화상   {os.path.basename(args.thermal)}  {arr.shape[0]}프레임")
    print(f"  겹치는 구간  {hi-lo:.1f}초")
    if hi - lo < 1:
        raise SystemExit("겹치는 구간이 없습니다 — 다른 짝을 주십시오.")

    polys = {d: fov_polygon(K, D, Kt, R, T, d) for d in LAYERS}
    ref = polys[LAYERS[0]]
    print(f"\n  층별 화각 경계가 서로 얼마나 다른가 (RGB 화소)")
    for d in LAYERS:
        dd = float(np.abs(polys[d]-ref).max())
        m0 = np.zeros((Hh, W), np.uint8)
        cv2.fillPoly(m0, [polys[d].astype(np.int32)], 255)
        print(f"    {d:6.0f}mm   최상단 대비 최대 {dd:5.1f} px   "
              f"RGB 화면의 {m0.mean()/255*100:4.1f} %")

    inside = np.zeros((Hh, W), np.uint8)
    cv2.fillPoly(inside, [polys[LAYERS[0]].astype(np.int32)], 255)

    sel = np.linspace(lo+0.5, hi-0.5, args.count)
    items = []
    print(f"\n  {args.count}장 추출 (열화상이 동시에 있는 시각에서만)")
    for k, t in enumerate(sel, 1):
        fi = int(round((t-s0)*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        j = int(np.argmin(np.abs(tsec-t)))
        dt = float(abs(tsec[j]-t))
        nm = f"{os.path.splitext(base)[0]}_f{fi:05d}"
        cv2.imwrite(os.path.join(args.out, "frames", nm+".png"), fr)

        cel = conv(np.asarray(arr[j])).astype(np.float64)[::-1, ::-1]
        th = cv2.cvtColor(to8(cel), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(args.out, "thermal", nm+"_th.png"),
                    cv2.resize(th, (640, 480), interpolation=cv2.INTER_NEAREST))

        ref_img = fr.copy()
        ref_img[inside == 0] = (ref_img[inside == 0]*0.35).astype(np.uint8)
        for d in LAYERS:
            p = polys[d].astype(np.int32)
            cv2.polylines(ref_img, [p], True, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.polylines(ref_img, [p], True, COLS[d], 2, cv2.LINE_AA)
        for x in range(0, W, 200):
            cv2.line(ref_img, (x, 0), (x, Hh), (150, 150, 150), 1)
            cv2.putText(ref_img, str(x), (x+4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (240, 240, 240), 1, cv2.LINE_AA)
        for y in range(0, Hh, 200):
            cv2.line(ref_img, (0, y), (W, y), (150, 150, 150), 1)
            cv2.putText(ref_img, str(y), (4, y+18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(ref_img, "annotate INSIDE the border only", (30, Hh-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(ref_img, "annotate INSIDE the border only", (30, Hh-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, "reference", nm+".png"), ref_img)
        items.append(dict(name=nm, rgb_frame=fi, t_sec=round(float(t), 3),
                          thermal_frame=int(j), dt_ms=round(dt*1000, 1)))
        print(f"    {k:>3}  {nm}  열화상 f{j:04d} (Δt {dt*1000:.0f} ms)")

    cap.release()
    cv2.imwrite(os.path.join(args.out, "thermal_fov_mask.png"), inside)
    json.dump(dict(rgb=base, thermal=os.path.basename(args.thermal),
                   width=W, height=Hh, fps=fps,
                   baseline_mm=round(b, 2),
                   layers_mm=list(LAYERS),
                   fov_polygon={str(int(d)): [[round(float(a), 1),
                                               round(float(c), 1)]
                                              for a, c in polys[d]]
                                for d in LAYERS},
                   note="fov_polygon 안쪽만 어노테이션. 개체마다 층(420/700/"
                        "1000)을 함께 적을 것. 투영은 그 층의 H(d) 로 한다.",
                   frames=items),
              open(os.path.join(args.out, "index.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  저장 {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
