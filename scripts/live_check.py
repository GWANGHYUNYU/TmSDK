#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""촬영 중에 돌려 놓는 즉시 판정기 — 한 건 찍을 때마다 합격/불합격을 말해 준다.

    python3 scripts/live_check.py <녹화폴더>            # 새 파일을 기다린다
    python3 scripts/live_check.py <녹화폴더> --once    # 이미 있는 것만 훑는다

3차 촬영에서 123건을 찍어 68건만 건졌습니다. 현장에서 알았으면 그 자리에서
다시 찍었을 것들입니다. 이 도구는 녹화가 끝나고 **몇 초 안에**

    · 잡혔는가
    · 렌즈~보드 거리 · 칸 크기 · 대비 · 기울임 · 면내회전 · 화면 점유
    · **아직 모자란 구간이 무엇인지**

를 알려 줍니다. 노트북을 옆에 두고 띄워 놓고 쓰십시오.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import (detect, to8, inplane_deg,        # noqa: E402
                         square_px, contrast_c)

F_PX = 147.4
NEED_PER_BIN = 4
NEED_TILT = 6            # 20도 이상 기울인 자세

MIN_SQ = 6.0             # findChessboardCornersSB 가 코너를 잡는 하한
MIN_CONTRAST = 2.0       # 3차는 중앙 1.24 ℃ 였습니다. 2 ℃ 를 바닥으로 둡니다
MAX_INPLANE = 20.0       # 눕혀 들 것 — 돌리면 보드가 작아집니다
MIN_OCC = 20.0

# 칸 크기별로 쓸 수 있는 거리대. 먼 쪽은 칸이 커야 합니다.
#   쓸 수 있는 최대 거리 = F_PX × 칸 / MIN_SQ
#     칸 30 mm → 737 mm      칸 50 mm → 1228 mm
BOARDS = {30.0: [(330, 430), (430, 530), (530, 650)],
          50.0: [(600, 750), (750, 900), (900, 1100)]}


def measure(path, cell, PAT):
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    best = None
    for i in range(0, min(arr.shape[0], 120), 2):
        cels = conv(np.asarray(arr[i])).astype(np.float64)
        ok, c, pt, _ = detect(to8(cels), [PAT])
        if not ok:
            continue
        ct = contrast_c(cels, c, pt) or 0.0
        if best is None or ct > best[0]:
            best = (ct, cels, c, pt)
    if best is None:
        return None
    ct, cels, c, pt = best
    sq = square_px(c, pt)
    q = c.reshape(pt[1], pt[0], 2)
    v = np.array([q[0, 0], q[0, -1], q[-1, -1], q[-1, 0]])
    a = 0.5*abs(np.dot(v[:, 0], np.roll(v[:, 1], -1))
                - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
    occ = a*(pt[0]+1)*(pt[1]+1)/((pt[0]-1)*(pt[1]-1))/(160*120)*100
    # 판 기울기 — 마주보는 두 변의 길이 비로 본다
    top = np.linalg.norm(q[0, -1]-q[0, 0])
    bot = np.linalg.norm(q[-1, -1]-q[-1, 0])
    lft = np.linalg.norm(q[-1, 0]-q[0, 0])
    rgt = np.linalg.norm(q[-1, -1]-q[0, -1])
    fore = min(top, bot)/max(top, bot) * min(lft, rgt)/max(lft, rgt)
    tilt = float(np.degrees(np.arccos(np.clip(fore, 0, 1))))
    return dict(dist=F_PX*cell/sq, sq=sq, ct=ct, occ=occ,
                rot=inplane_deg(c, pt), tilt=tilt)


def verdict(m):
    """→ (쓸 수 있는가, 표시, 할 말).

    대비는 **낙제가 아니라 경고**입니다. 실제로 0.4 ℃ 에서도 코너는 잡혔습니다.
    다만 여유가 없어 조금만 흐트러져도 통째로 실패하므로, 2 ℃ 밑이면
    재가열하라고 알려 줍니다 (2026-08-28 은 중앙 4.1 ℃ 였습니다).
    """
    if m is None:
        return False, "× 다시", "코너 못 잡음 — 판을 다시 데우고 다시 드십시오"
    bad, warn = [], []
    if m["sq"] < MIN_SQ:
        bad.append(f"칸 {m['sq']:.1f}px — 칸이 더 큰 보드를 쓰거나 다가가십시오")
    if m["rot"] > MAX_INPLANE:
        bad.append(f"면내회전 {m['rot']:.0f}도 · 눕혀 드십시오")
    if m["ct"] < MIN_CONTRAST:
        warn.append(f"대비 {m['ct']:.1f}℃ · 다음 자세 전에 재가열")
    if m["occ"] < MIN_OCC:
        warn.append(f"점유 {m['occ']:.0f}%")
    if bad:
        return False, "× 다시", " · ".join(bad + warn)
    if warn:
        return True, "△ 주의", " · ".join(warn)
    return True, "○ 합격", ""


def progress(hits, BINS):
    out = []
    for lo, hi in BINS:
        n = sum(1 for m in hits if lo <= m["dist"] < hi)
        mark = "#"*min(n, NEED_PER_BIN) + "."*max(0, NEED_PER_BIN-n)
        out.append(f"{lo}[{mark}]")
    nt = sum(1 for m in hits if m["tilt"] >= 20)
    return (f"  진행  {' '.join(out)}   기울임20도+ {nt}/{NEED_TILT}   "
            f"합계 {len(hits)}")


def todo(hits, BINS):
    need = []
    for lo, hi in BINS:
        n = sum(1 for m in hits if lo <= m["dist"] < hi)
        if n < NEED_PER_BIN:
            need.append(f"{lo}~{hi}mm {NEED_PER_BIN-n}개")
    nt = sum(1 for m in hits if m["tilt"] >= 20)
    if nt < NEED_TILT:
        need.append(f"20도 이상 기울인 자세 {NEED_TILT-nt}개")
    return "  남은 것  " + (", ".join(need) if need else "★ 다 찼습니다")


def main():
    ap = argparse.ArgumentParser(description="촬영 중 즉시 판정")
    ap.add_argument("folder")
    ap.add_argument("--cell", type=float, default=30.0,
                    help="체커 한 칸 mm. 30 이면 최대 737 mm, 50 이면 1228 mm")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--once", action="store_true", help="기다리지 않고 한 번만")
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()
    PAT = tuple(int(v) for v in args.pattern.lower().split("x"))
    BINS = BOARDS.get(args.cell)
    if BINS is None:                     # 등록되지 않은 칸 크기 → 3등분
        e = np.linspace(330, F_PX*args.cell/MIN_SQ, 4).round().astype(int)
        BINS = list(zip(e[:-1].tolist(), e[1:].tolist()))

    print("=" * 88)
    print(f"즉시 판정 · {args.folder}")
    print(f"보드  칸 {args.cell:g} mm · {PAT[0]}x{PAT[1]} · "
          f"이 보드로 잡을 수 있는 최대 거리 {F_PX*args.cell/MIN_SQ:.0f} mm")
    print(f"기준  × 낙제: 코너 못 잡음 · 칸 < {MIN_SQ}px · 면내회전 > {MAX_INPLANE:g}도")
    print(f"      △ 경고: 대비 < {MIN_CONTRAST:g}℃ · 점유 < {MIN_OCC:g}%")
    print("=" * 88)

    seen, hits = set(), []
    while True:
        try:
            fs = sorted(f for f in os.listdir(args.folder)
                        if f.endswith(".y16raw"))
        except OSError:
            fs = []
        for f in fs:
            if f in seen:
                continue
            p = os.path.join(args.folder, f)
            if not os.path.exists(os.path.splitext(p)[0]+".y16meta"):
                continue                      # 아직 녹화 중
            s = os.path.getsize(p)
            time.sleep(0.3)
            if os.path.getsize(p) != s:
                continue                      # 아직 쓰는 중
            seen.add(f)
            try:
                m = measure(p, args.cell, PAT)
            except Exception as e:
                print(f"  {f[-10:]:<12} 읽기 실패 {e}")
                continue
            ok, tag, why = verdict(m)
            stem = os.path.splitext(f)[0][-6:]
            if m is None:
                print(f"\n[{len(seen):>3}] {stem}  {tag}   {why}")
            else:
                print(f"\n[{len(seen):>3}] {stem}  {tag}  "
                      f"{m['dist']:4.0f}mm  칸{m['sq']:4.1f}px  "
                      f"대비{m['ct']:4.1f}℃  기울임{m['tilt']:3.0f}도  "
                      f"면내{m['rot']:3.0f}도  점유{m['occ']:3.0f}%")
                if why:
                    print(f"      → {why}")
                if ok:
                    hits.append(m)
            print(progress(hits, BINS))
            print(todo(hits, BINS))
        if args.once:
            break
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
