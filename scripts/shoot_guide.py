#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""촬영 참고 자료 — 잘 나온 판과 안 나온 판을 나란히 보여 준다.

    python3 scripts/shoot_guide.py --out output/shoot_guide

실제로 찍힌 녹화에서 **보드 체커 대비**(검정 칸과 금속 칸의 겉보기 온도차)
순서로 골라 붙입니다. 현장에서 「이 정도면 되고 이 정도면 안 된다」를
눈으로 맞춰 보는 용도입니다.

같이 표시하는 것
    · 초록 십자   검출된 코너
    · 빨강        보드보다 4 ℃ 이상 뜨거운 화소 (LED 바와 그 반사)
    · 제목        대비 ℃ · 금속 칸끼리의 산포 ℃ (반사 얼룩의 크기)

**대비가 반사 얼룩의 3배 이상이면 검출이 안정적입니다.**
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import detect, to8                       # noqa: E402
from read_y16 import (load_meta, load_raw,                # noqa: E402
                      temperature_converter)

HOT_OVER_BOARD = 4.0
TILE = (320, 240)


def poly(c, pat):
    q = c.reshape(pat[1], pat[0], 2)
    return np.array([q[0, 0], q[0, -1], q[-1, -1], q[-1, 0]], np.float32)


def squares(cels, c, pat):
    """칸 중심들의 평균 온도 → (검정 칸들, 금속 칸들)."""
    q = c.reshape(pat[1], pat[0], 2)
    v = {0: [], 1: []}
    for j in range(pat[1]-1):
        for i in range(pat[0]-1):
            cx, cy = q[j:j+2, i:i+2].reshape(4, 2).mean(0)
            x, y = int(cx), int(cy)
            if 1 <= x < cels.shape[1]-1 and 1 <= y < cels.shape[0]-1:
                v[(i+j) % 2].append(float(cels[y-1:y+2, x-1:x+2].mean()))
    a, b = np.array(v[0]), np.array(v[1])
    return (a, b) if a.std() < b.std() else (b, a)     # 산포 작은 쪽이 검정


def first_hit(path, pat):
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    for i in range(0, min(arr.shape[0], 200), 2):
        cels = conv(np.asarray(arr[i])).astype(np.float64)
        ok, c, pt, _ = detect(to8(cels), [pat])
        if ok:
            return cels, c, pt
    return None


def tile(cels, c, pat, ct, sc, name, verdict, colour):
    bd = float(np.median(cels[cv2.fillPoly(
        np.zeros(cels.shape, np.uint8), [poly(c, pat).astype(np.int32)], 1) > 0]))
    g = cv2.cvtColor(to8(cels), cv2.COLOR_GRAY2BGR)
    g[cels > bd + HOT_OVER_BOARD] = (0, 0, 255)
    for q in c.reshape(-1, 2):
        cv2.drawMarker(g, tuple(np.int32(q)), (0, 255, 0),
                       cv2.MARKER_CROSS, 3, 1)
    g = cv2.resize(g, TILE, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(g, (0, 0), (TILE[0]-1, TILE[1]-1), colour, 4)
    cv2.rectangle(g, (0, 0), (TILE[0], 40), (0, 0, 0), -1)
    cv2.putText(g, f"{verdict}  contrast {ct:.2f}C", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cv2.putText(g, f"{name}  reflection {sc:.2f}C  ratio {ct/max(sc,.01):.1f}x",
                (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
                cv2.LINE_AA)
    return g


def main():
    ap = argparse.ArgumentParser(description="촬영 참고 자료")
    ap.add_argument("--out", default="output/shoot_guide")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = list(csv.DictReader(
        open(os.path.join(HERE, "calib/INDEX.csv"), encoding="utf-8-sig")))
    items = []
    for r in rows:
        p = os.path.join(HERE, "calib", r["날짜"], "thermal", r["파일"]+".y16raw")
        pat = tuple(int(v) for v in r["격자"].split("x"))
        got = first_hit(p, pat)
        if got is None:
            continue
        cels, c, pt = got
        blk, met = squares(cels, c, pt)
        if len(blk) < 4 or len(met) < 4:
            continue
        ct = abs(float(blk.mean()-met.mean()))
        sc = float(met.std())
        items.append(dict(cels=cels, c=c, pat=pt, ct=ct, sc=sc,
                          name=r["파일"][:5]+" "+r["날짜"][5:],
                          full=(r["비고"] == "전체격자")))
    items.sort(key=lambda x: -x["ct"])
    print(f"조사 {len(items)}건 · 대비 {items[-1]['ct']:.2f} ~ {items[0]['ct']:.2f} ℃")

    bands = [("GOOD  >= 3C", lambda x: x["ct"] >= 3.0, (80, 220, 80)),
             ("MARGINAL 1.5-3C", lambda x: 1.5 <= x["ct"] < 3.0, (0, 200, 255)),
             ("BAD  < 1.5C", lambda x: x["ct"] < 1.5, (60, 60, 255))]
    panels = []
    for label, cond, col in bands:
        sel = [x for x in items if cond(x)][:4]
        print(f"  {label:<18}{sum(1 for x in items if cond(x)):>3}건")
        if not sel:
            continue
        ts = [tile(x["cels"], x["c"], x["pat"], x["ct"], x["sc"], x["name"],
                   label.split()[0], col) for x in sel]
        while len(ts) < 4:
            ts.append(np.zeros((TILE[1], TILE[0], 3), np.uint8))
        panels.append(np.hstack(ts))
    cv2.imwrite(os.path.join(args.out, "contrast_bands.png"), np.vstack(panels))
    print(f"  저장 {args.out}/contrast_bands.png")

    # 대비 대 반사 산란 — 검출 성공 여부와의 관계
    ct = np.array([x["ct"] for x in items])
    sc = np.array([x["sc"] for x in items])
    fu = np.array([x["full"] for x in items])
    print(f"\n  전체격자(7x4) 성공  대비 중앙 {np.median(ct[fu]):.2f}℃ · "
          f"대비/반사 {np.median(ct[fu]/np.maximum(sc[fu],.01)):.1f}배")
    print(f"  부분격자로 떨어짐   대비 중앙 {np.median(ct[~fu]):.2f}℃ · "
          f"대비/반사 {np.median(ct[~fu]/np.maximum(sc[~fu],.01)):.1f}배")
    for th in (2, 3, 5, 8):
        m = (ct/np.maximum(sc, .01)) >= th
        if m.sum():
            print(f"  대비/반사 {th}배 이상인 {int(m.sum()):>2}건 중 "
                  f"7x4 성공 {int(fu[m].sum())}건 ({fu[m].mean()*100:.0f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
