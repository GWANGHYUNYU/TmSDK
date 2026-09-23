#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB 어노테이션을 열화상 좌표로 투영하고 눈으로 확인한다.

    python3 scripts/project_annot.py annotations.json output/annotate_v2 \
        --out output/projected

하는 일
    ① RGB 왜곡을 푼다            undistortPoints(..., P=K_rgb)
    ② 그 개체의 층으로 H(d) 적용  perspectiveTransform
    ③ 결과는 **180° 돌린 열화상** 좌표계
    ④ 같은 순간 열화상 위에 얹어 그림으로 낸다
    ⑤ 개체별 평균 엽온을 뽑는다

이것이 정합이 맞는지 보는 마지막 검증입니다. 잎을 칠한 자리에 마스크가
얹히면 끝, 한쪽으로 밀리면 «층을 정면 평면으로 본다»는 가정을 의심합니다.
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

CLS_COL = {"잎": (90, 220, 90), "꽃": (250, 250, 250),
           "딸기": (70, 70, 240), None: (200, 160, 60)}
TH_W, TH_H = 160, 120


def load_params(p):
    z = np.load(p)
    return z["K"], z["D"], z["Kt"], z["R"], z["T"]


def H_of(K, Kt, R, T, d):
    n = np.array([[0.], [0.], [1.]])
    return Kt @ (R + (T @ n.T)/d) @ np.linalg.inv(K)


def project(pts, K, D, Kt, R, T, d):
    """원본 RGB 파일 좌표 → 180° 돌린 열화상 좌표."""
    a = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(a, K, D, P=K)
    return cv2.perspectiveTransform(und, H_of(K, Kt, R, T, d)).reshape(-1, 2)


def thermal_at(raw, frame):
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(raw)
    arr = load_raw(raw, meta)
    conv = temperature_converter(meta)
    i = min(max(int(frame), 0), arr.shape[0]-1)
    return conv(np.asarray(arr[i])).astype(np.float64)[::-1, ::-1]


def to8(x):
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x-lo)/(hi-lo+1e-9)*255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="어노테이션 투영·검증")
    ap.add_argument("annot")
    ap.add_argument("package", help="make_annotate.py 가 낸 폴더 (index.json 필요)")
    ap.add_argument("--params", default="params/thermal_rgb_stereo.npz")
    ap.add_argument("--out", default="output/projected")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    K, D, Kt, R, T = load_params(os.path.join(HERE, args.params))
    idx = json.load(open(os.path.join(args.package, "index.json"),
                         encoding="utf-8"))
    byname = {f["name"]+".png": f for f in idx["frames"]}
    raw = None
    for c in (idx["thermal"],
              os.path.join(HERE, "calib/raw/th/raw_output", idx["thermal"])):
        if os.path.exists(c):
            raw = c
            break
    if raw is None:
        raise SystemExit(f"열화상 파일을 못 찾았습니다: {idx['thermal']}")

    A = json.load(open(args.annot, encoding="utf-8"))
    print("=" * 84)
    print(f"어노테이션 투영 · 베이스라인 {np.linalg.norm(T):.2f} mm")
    print("=" * 84)

    rows = []
    for im in A["images"]:
        nm = im["image"]
        if nm not in byname:
            print(f"  {nm}  index.json 에 없습니다 — 건너뜁니다")
            continue
        meta = byname[nm]
        cels = thermal_at(raw, meta["thermal_frame"])
        base = cv2.cvtColor(to8(cels), cv2.COLOR_GRAY2BGR)
        big = cv2.resize(base, (TH_W*6, TH_H*6),
                         interpolation=cv2.INTER_NEAREST)
        rgbp = os.path.join(args.package, "frames", nm)
        rgb = cv2.imread(rgbp)

        print(f"\n── {nm}  (열화상 f{meta['thermal_frame']}, "
              f"Δt {meta['dt_ms']:.0f} ms) " + "─"*24)
        print(f"  {'개체':<8}{'분류':<6}{'층':>6}{'열화상 안':>9}"
              f"{'면적(px)':>10}{'평균온도':>10}")
        nin = 0
        for o in im["objects"]:
            d = float(o["layer_mm"])
            p = project(o["points"], K, D, Kt, R, T, d)
            cls = o.get("class") or o.get("label")
            col = CLS_COL.get(cls, CLS_COL[None])
            inside = ((p[:, 0] >= 0) & (p[:, 0] < TH_W) &
                      (p[:, 1] >= 0) & (p[:, 1] < TH_H)).mean()*100
            m = np.zeros((TH_H, TH_W), np.uint8)
            if o["type"] in ("poly", "rect") and len(p) >= 3:
                cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
            area = int(m.sum())
            tmean = float(cels[m > 0].mean()) if area else float("nan")
            if inside > 50:
                nin += 1
            rows.append(dict(image=nm, id=o["id"], cls=cls, layer=d,
                             inside_pct=round(inside, 1), area_px=area,
                             mean_C=(round(tmean, 2) if area else None)))
            print(f"  {o['id']:<8}{(cls or '-'):<6}{d:>6.0f}{inside:>8.0f}%"
                  f"{area:>10}" +
                  (f"{tmean:>9.2f}℃" if area else f"{'-':>10}"))
            q = (p*6).astype(np.int32)
            if o["type"] in ("poly", "rect"):
                cv2.polylines(big, [q], True, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.polylines(big, [q], True, col, 2, cv2.LINE_AA)
            elif o["type"] == "line":
                cv2.polylines(big, [q], False, col, 2, cv2.LINE_AA)
            else:
                for z in q:
                    cv2.circle(big, tuple(z), 5, col, -1)
            if len(q):
                cv2.putText(big, o["id"].replace("obj", ""),
                            tuple(q.mean(0).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(big, o["id"].replace("obj", ""),
                            tuple(q.mean(0).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        print(f"  → 열화상 안에 절반 넘게 들어온 개체 {nin}/{len(im['objects'])}")

        cv2.imwrite(os.path.join(args.out, nm.replace(".png", "_th.png")), big)
        if rgb is not None:
            v = cv2.rotate(rgb, cv2.ROTATE_180)
            for o in im["objects"]:
                cls = o.get("class") or o.get("label")
                col = CLS_COL.get(cls, CLS_COL[None])
                q = np.array([[1919-x, 1079-y] for x, y in o["points"]],
                             np.int32)
                cv2.polylines(v, [q], o["type"] != "line", col, 3, cv2.LINE_AA)
            v = cv2.resize(v, (TH_W*6, int(TH_W*6*1080/1920)))
            pad = np.zeros((big.shape[0], v.shape[1], 3), np.uint8)
            pad[:v.shape[0]] = v
            cv2.imwrite(os.path.join(args.out, nm.replace(".png", "_pair.png")),
                        np.hstack([pad, big]))

    import csv
    with open(os.path.join(args.out, "objects.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  저장 {args.out}/  (그림 · objects.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
