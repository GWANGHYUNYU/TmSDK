#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB 개체 어노테이션 패키지 생성 — 열화상 화각 경계를 함께 표시.

    python3 scripts/make_rgb_annotate.py output/17-46-52_after.mp4 \
        output/pair_tight/stereo_final.npz --depth 450 \
        --out output/annotate_rgb

기존 `output/annotate/`(열화상 GSD용)와 같은 규약을 따릅니다.
    frames/     여기에 칠합니다 (원본 해상도)
    reference/  열화상 화각 경계 + 좌표 격자. 눈으로만 확인
    index.json  프레임별 메타

열화상 화각 밖은 열화상 데이터가 없으므로 칠해도 쓸 수 없습니다.
reference 의 노란 테두리 **안쪽만** 작업하십시오.
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


def main():
    ap = argparse.ArgumentParser(description="RGB 어노테이션 패키지")
    ap.add_argument("video")
    ap.add_argument("stereo", help="stereo_final.npz (열화상 화각 계산용)")
    ap.add_argument("--depth", type=float, default=450.0,
                    help="정합 기준 깊이 mm — 열화상 화각을 그릴 깊이")
    ap.add_argument("--count", type=int, default=10, help="뽑을 프레임 수")
    ap.add_argument("--out", default="output/annotate_rgb")
    args = ap.parse_args()

    for d in ("frames", "reference"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    z = np.load(args.stereo, allow_pickle=True)
    R, T, Kt, Kr, Dr = z["R"], z["T"], z["Kt"], z["Kr"], z["Dr"]
    nv = np.array([[0.], [0.], [1.]])
    Hd = Kt @ (R + (T @ nv.T)/args.depth) @ np.linalg.inv(Kr)
    Hi = np.linalg.inv(Hd)
    # 열화상 테두리를 촘촘히 잡아 RGB 로 보낸다 (왜곡 때문에 곡선이 된다)
    e = []
    for a in range(0, 160, 4):
        e.append([a, 0])
    for b in range(0, 120, 4):
        e.append([159, b])
    for a in range(159, -1, -4):
        e.append([a, 119])
    for b in range(119, -1, -4):
        e.append([0, b])
    poly = cv2.perspectiveTransform(
        np.array(e, np.float32).reshape(-1, 1, 2), Hi).reshape(-1, 2)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.splitext(os.path.basename(args.video))[0]
    print("=" * 82)
    print(f"RGB 어노테이션 패키지 · {W}x{H} · {n/fps:.0f}초 · {args.count}장")
    print("=" * 82)

    sel = np.linspace(n*0.05, n*0.95, args.count).astype(int)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inside, [poly.astype(np.int32)], 255)
    frac = inside.mean()/255*100
    items = []
    for k, fi in enumerate(sel, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        nm = f"{base}_f{int(fi):05d}"
        cv2.imwrite(os.path.join(args.out, "frames", nm+".png"), fr)

        ref = fr.copy()
        # 화각 밖을 어둡게 — 칠하지 말아야 할 곳이 한눈에
        ref[inside == 0] = (ref[inside == 0]*0.35).astype(np.uint8)
        cv2.polylines(ref, [poly.astype(np.int32)], True, (0, 0, 0), 8,
                      cv2.LINE_AA)
        cv2.polylines(ref, [poly.astype(np.int32)], True, (60, 220, 255), 3,
                      cv2.LINE_AA)
        for x in range(0, W, 200):
            cv2.line(ref, (x, 0), (x, H), (150, 150, 150), 1)
            cv2.putText(ref, str(x), (x+4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (240, 240, 240), 1, cv2.LINE_AA)
        for y in range(0, H, 200):
            cv2.line(ref, (0, y), (W, y), (150, 150, 150), 1)
            cv2.putText(ref, str(y), (4, y+18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(ref, f"THERMAL FOV @ {args.depth:.0f}mm  -  "
                    f"annotate INSIDE only", (30, H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(ref, f"THERMAL FOV @ {args.depth:.0f}mm  -  "
                    f"annotate INSIDE only", (30, H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, "reference", nm+".png"), ref)
        items.append(dict(name=nm, frame=int(fi), t_sec=round(fi/fps, 2)))
        print(f"  {k:>3}  {nm}  ({fi/fps:.0f}초)")

    cv2.imwrite(os.path.join(args.out, "thermal_fov_mask.png"), inside)
    json.dump(dict(video=os.path.basename(args.video), width=W, height=H,
                   fps=fps, depth_mm=args.depth,
                   fov_fraction_pct=round(frac, 1),
                   fov_polygon=[[round(float(a), 1), round(float(b), 1)]
                                for a, b in poly],
                   note="fov_polygon 안쪽만 어노테이션. 밖은 열화상 데이터 없음",
                   frames=items),
              open(os.path.join(args.out, "index.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  열화상 화각이 RGB 화면의 {frac:.0f} % (깊이 {args.depth:g} mm)")
    print(f"  마스크: {args.out}/thermal_fov_mask.png")
    print(f"  저장: {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
