#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""녹화 전수 조사 — 어느 녹화를 남기고 어느 것을 버릴지 판정한다.

    python3 scripts/inventory.py calib/th_260914 --out output/inventory

각 `.y16raw` 를 프레임 단위로 훑어 자세를 뽑고, 검출이 아예 안 되는 녹화는
왜 안 되는지(판이 안 데워짐 / 칸이 너무 작음 / 판이 화면 밖)까지 가른다.
`inventory.json` 에 녹화별 결과를 남겨 정리 스크립트가 그대로 읽는다.

판정 기준 — 하나라도 어기면 버린다
    · 검출 0 장                            쓸 수 없음
    · 칸 < 6.0 px                         SB 검출기 한계. 코너가 안 잡힌다
    · 대비 < 0.4 ℃                        코너 위치가 온도잡음에 묻힌다
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import (detect, to8, short, inplane_deg, square_px,   # noqa: E402
                         scene_stats, contrast_c)

CELL_MM = 30.0
F_PX = 147.4
PAT = (7, 4)
MIN_SQ = 6.0            # 이보다 작으면 검출기가 코너를 못 잡는다
MIN_CONTRAST = 0.4      # ℃


def rec_time(name):
    m = re.search(r"(\d{8})_(\d{6})", name)
    return (dt.datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")
            if m else None)


def occupancy(corners, pat, size):
    """보드가 화면에서 차지하는 면적 비율 (신발끈 공식, 바깥 칸 보정)."""
    q = corners.reshape(pat[1], pat[0], 2)
    v = np.array([q[0, 0], q[0, -1], q[-1, -1], q[-1, 0]])
    a = 0.5*abs(np.dot(v[:, 0], np.roll(v[:, 1], -1))
                - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
    full = a*(pat[0]+1)*(pat[1]+1)/((pat[0]-1)*(pat[1]-1))
    return float(full/(size[0]*size[1]))


def scan(path, stride, min_move):
    """녹화 하나에서 서로 다른 자세를 뽑는다 → (자세목록, 진단)."""
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    n = arr.shape[0]
    secs = None
    mp = os.path.splitext(path)[0] + ".y16meta"
    try:
        ts = json.load(open(mp, encoding="utf-8")).get("timestamps") or []
        secs = [dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f") for s in ts]
    except Exception:
        secs = None

    poses, rng_all, loc_all, ndet = [], [], [], 0
    for i in range(0, n, stride):
        cels = conv(np.asarray(arr[i])).astype(np.float64)
        g = to8(cels)
        r, lo = scene_stats(cels)
        rng_all.append(r)
        loc_all.append(lo)
        ok, c, pat, how = detect(g, [PAT])
        if not ok:
            continue
        ndet += 1
        rot = inplane_deg(c, pat)
        if poses and abs(rot - poses[-1]["rot"]) < min_move:
            continue
        sq = square_px(c, pat)
        t = (secs[i] if secs and i < len(secs) else None)
        poses.append(dict(
            frame=i, rot=round(rot, 1), sq=round(sq, 2),
            dist=round(F_PX*CELL_MM/sq, 1) if sq > 0 else None,
            occ=round(occupancy(c, pat, (160, 120))*100, 1),
            contrast=round(contrast_c(cels, c, pat) or 0, 2),
            t=(t.strftime("%H:%M:%S.%f")[:-3] if t else None),
            how=how))
    return poses, dict(frames=n, scanned=len(range(0, n, stride)), ndet=ndet,
                       rng=round(float(np.nanmax(rng_all)), 2) if rng_all else 0,
                       loc=round(float(np.nanmax(loc_all)), 2) if loc_all else 0,
                       t0=(secs[0].strftime("%H:%M:%S") if secs else None),
                       t1=(secs[-1].strftime("%H:%M:%S") if secs else None))


def verdict(poses, diag):
    """남길지 버릴지와 그 이유."""
    if not poses:
        if diag["loc"] < 0.5:
            return False, f"판이 안 데워짐 (국소대비 {diag['loc']:.2f}℃)"
        return False, f"검출 0 (대비는 {diag['loc']:.2f}℃ 있음 — 자세/거리 문제)"
    sq = max(p["sq"] for p in poses)
    ct = max(p["contrast"] for p in poses)
    if sq < MIN_SQ:
        return False, f"칸 {sq:.1f} px < {MIN_SQ} — 너무 멀다"
    if ct < MIN_CONTRAST:
        return False, f"대비 {ct:.2f}℃ < {MIN_CONTRAST}"
    return True, (f"검출 {diag['ndet']}/{diag['scanned']} · 자세 {len(poses)}개 "
                  f"· 칸 {sq:.1f} px · 대비 {ct:.2f}℃")


def main():
    ap = argparse.ArgumentParser(description="녹화 전수 조사")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-move", type=float, default=8.0)
    ap.add_argument("--out", default="output/inventory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raws = []
    for root in args.roots:
        for dp, _, fns in os.walk(root):
            for f in sorted(fns):
                if f.endswith(".y16raw"):
                    raws.append(os.path.join(dp, f))
    raws.sort()
    print("=" * 96)
    print(f"녹화 전수 조사 · {len(raws)}건")
    print("=" * 96)

    recs, cur = [], None
    for p in raws:
        grp = os.path.basename(os.path.dirname(p))
        if grp != cur:
            cur = grp
            print(f"\n── {grp} " + "─"*(90-len(grp)))
        nm = os.path.splitext(os.path.basename(p))[0]
        try:
            poses, diag = scan(p, args.stride, args.min_move)
        except Exception as e:
            print(f"  {short(nm, 32):<32}읽기 실패 {e}")
            continue
        keep, why = verdict(poses, diag)
        sz = sum(os.path.getsize(os.path.splitext(p)[0]+e)
                 for e in (".y16raw", ".y16meta", ".avi", ".csv")
                 if os.path.exists(os.path.splitext(p)[0]+e))
        recs.append(dict(path=os.path.relpath(p, HERE).replace("\\", "/"),
                         group=grp, name=nm, keep=keep, why=why,
                         bytes=sz, poses=poses, **diag))
        mark = "○" if keep else "×"
        ds = (f"{min(p['dist'] for p in poses):.0f}~"
              f"{max(p['dist'] for p in poses):.0f}mm" if poses else "-")
        print(f"  {mark} {nm[-6:]}  {diag['frames']:>4}f  {ds:<14}{why}")

    ok = [r for r in recs if r["keep"]]
    print("\n" + "=" * 96)
    print(f"남길 녹화 {len(ok)}/{len(recs)}건 · "
          f"{sum(r['bytes'] for r in ok)/1e6:.0f} MB "
          f"(버릴 것 {sum(r['bytes'] for r in recs if not r['keep'])/1e6:.0f} MB)")
    tot = sum(len(r["poses"]) for r in ok)
    if ok:
        d = [p["dist"] for r in ok for p in r["poses"]]
        print(f"자세 {tot}개 · 거리 {min(d):.0f} ~ {max(d):.0f} mm")
    json.dump(recs, open(os.path.join(args.out, "inventory.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장: {args.out}/inventory.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
