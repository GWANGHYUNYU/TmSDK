#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""한 촬영 세션을 `thermal/` + `rgb/` 구조로 정리한다.

    python3 scripts/organise_session.py calib/2026-09-18 \
        output/inv_260918/inventory.json              # 계획만
    python3 scripts/organise_session.py calib/2026-09-18 \
        output/inv_260918/inventory.json --apply

`consolidate_calib.py` 는 2026-08-28·09-14 두 세션에 맞춰 경로가 박혀 있어서
새 세션마다 고쳐야 했습니다. 이쪽은 **세션 폴더 하나만** 받습니다.

    calib/2026-09-18/th_260918*/  →  calib/2026-09-18/thermal/
    calib/2026-09-18/rgb/         →  그대로

파일명은 기존 규칙 그대로 `d{거리}_{YYYYMMDD}_{HHMMSS}` 입니다. 뒤의 시각은
RGB 짝짓기에 쓰이므로 반드시 남깁니다.

`--apply` 없이 돌리면 무엇을 옮기고 무엇을 버릴지만 출력합니다.
버릴 것의 진단 수치는 `DROPPED.csv` 에 남깁니다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = (".y16raw", ".y16meta", ".avi", ".csv")


def stamp(name):
    m = re.search(r"(\d{8})_(\d{6})", name)
    return m.groups() if m else (None, None)


def main():
    ap = argparse.ArgumentParser(description="세션 폴더 정리")
    ap.add_argument("session", help="예: calib/2026-09-18")
    ap.add_argument("inventory", help="inventory.py 가 낸 inventory.json")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    recs = json.load(open(args.inventory, encoding="utf-8"))
    dst = os.path.join(args.session, "thermal")
    moves, drops, seen = [], [], {}

    for r in sorted(recs, key=lambda r: r["poses"][0]["dist"]
                    if r["poses"] and r["poses"][0]["dist"] else 9999):
        base = os.path.splitext(os.path.join(HERE, r["path"]))[0]
        files = [(base+e, os.path.getsize(base+e))
                 for e in EXTS if os.path.exists(base+e)]
        if not files:
            continue
        ymd, hms = stamp(r["name"])
        d = [q["dist"] for q in r["poses"] if q.get("dist")]
        info = dict(대비_C=(max(q["contrast"] for q in r["poses"])
                          if r["poses"] else None),
                    거리_mm=(min(d) if d else None),
                    칸_px=(max(q["sq"] for q in r["poses"])
                           if r["poses"] else None),
                    검출프레임=f"{r.get('ndet')}/{r.get('scanned')}",
                    국소대비_C=r.get("loc"), 프레임수=r.get("frames"))
        if not r["keep"] or ymd is None:
            drops += [(p, s, r["why"], info) for p, s in files]
            continue
        nm = f"d{int(round(min(d))):04d}_{ymd}_{hms}"
        if nm in seen:          # 같은 시각이 두 폴더에 복사돼 있던 경우
            drops += [(p, s, f"{seen[nm]} 와 같은 녹화의 사본", info)
                      for p, s in files]
            continue
        seen[nm] = r["name"]
        for p, s in files:
            moves.append((p, os.path.join(dst, nm+os.path.splitext(p)[1]), s))
        r["newname"] = nm

    kept = [r for r in recs if r.get("newname")]
    mv = sum(s for _, _, s in moves)
    dl = sum(s for _, s, _, _ in drops)
    print("=" * 88)
    print(f"{args.session} · {'실행' if args.apply else '계획 (바꾸지 않음)'}")
    print("=" * 88)
    print(f"  남길 녹화  {len(kept)} / {len(recs)}건")
    if kept:
        ds = sorted(min(q["dist"] for q in r["poses"] if q.get("dist"))
                    for r in kept)
        cs = sorted(max(q["contrast"] for q in r["poses"]) for r in kept)
        print(f"  거리       {ds[0]:.0f} ~ {ds[-1]:.0f} mm")
        print(f"  보드 대비  {cs[0]:.2f} ~ {cs[-1]:.2f} ℃ "
              f"(중앙 {cs[len(cs)//2]:.2f})")
    print(f"  옮길 파일  {len(moves)}개  {mv/1e6:.1f} MB")
    print(f"  버릴 파일  {len(drops)}개  {dl/1e6:.1f} MB")
    agg = {}
    for _, s, why, _i in drops:
        k = re.sub(r"[\d.]+", "N", why)[:50]
        n, b = agg.get(k, (0, 0))
        agg[k] = (n+1, b+s)
    for k, (n, b) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"    {k:<52}{n:>4}개  {b/1e6:6.1f} MB")

    if not args.apply:
        print("\n  --apply 를 붙이면 실행합니다.")
        return 0

    os.makedirs(dst, exist_ok=True)
    cols = ["대비_C", "거리_mm", "칸_px", "검출프레임", "국소대비_C", "프레임수"]
    with open(os.path.join(args.session, "DROPPED.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["버린 파일", "바이트", "이유"] + cols)
        for p, s, why, i in drops:
            w.writerow([os.path.relpath(p, HERE).replace("\\", "/"), s, why]
                       + [i.get(c, "") for c in cols])
    for src, d, _ in moves:
        shutil.move(src, d)
    for p, _, _, _ in drops:
        os.remove(p)
    for sub in sorted(os.listdir(args.session)):
        p = os.path.join(args.session, sub)
        if os.path.isdir(p) and sub not in ("thermal", "rgb") and not os.listdir(p):
            os.rmdir(p)

    idx = os.path.join(args.session, "INDEX.csv")
    with open(idx, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["파일", "거리_mm", "격자", "대비_C", "자세수", "프레임",
                    "원래폴더", "검출프레임"])
        for r in sorted(kept, key=lambda r: r["newname"]):
            d = [q["dist"] for q in r["poses"] if q.get("dist")]
            w.writerow([r["newname"], int(round(min(d))), "7x4",
                        round(max(q["contrast"] for q in r["poses"]), 2),
                        len(r["poses"]), r["frames"], r["group"],
                        f"{r.get('ndet')}/{r.get('scanned')}"])
    print(f"\n  완료 — {dst}/ · {idx} · DROPPED.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
