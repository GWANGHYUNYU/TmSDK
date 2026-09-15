#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calib/ 정리 — 쓸 수 있는 녹화만 남기고 날짜별로 묶는다.

    python3 scripts/consolidate_calib.py                 # 계획만 출력
    python3 scripts/consolidate_calib.py --apply         # 실제로 옮기고 지운다

`inventory.py` 와 부분격자 판정 결과를 읽어 다음 구조로 만든다.

    calib/
      2026-08-28/thermal/  d0387_20260828_162609.y16raw ...
      2026-08-28/rgb/
      2026-09-14/thermal/
      2026-09-14/rgb/
      INDEX.csv     녹화별 거리·대비·자세·옛 이름
      DELETED.csv   지운 파일 목록 (되돌릴 수는 없지만 무엇이 있었는지는 남는다)

파일명 앞에 **실측 거리**를 붙여 `ls` 만으로 거리순 정렬이 되게 한다.
뒤의 `YYYYMMDD_HHMMSS` 는 RGB 짝짓기에 쓰이므로 그대로 둔다.
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
CALIB = os.path.join(HERE, "calib")
EXTS = (".y16raw", ".y16meta", ".avi", ".csv")

# 날짜 → 폴더
DATE_OF = {"20260828": "2026-08-28", "20260914": "2026-09-14"}

# 옛 _all 사본의 이름 (문서가 이 이름으로 참조한다)
OLD_ALIAS = {
    "20260828_162609": "d350_01", "20260828_162622": "d350_02",
    "20260828_162644": "d350_03", "20260828_162707": "d400_01",
    "20260828_162714": "d400_02", "20260828_162721": "d400_03",
    "20260828_162752": "d450_01", "20260828_162800": "d450_02",
    "20260828_162811": "d450_03", "20260828_161340": "dUNK_01",
    "20260828_161425": "dUNK_02", "20260828_161548": "dUNK_03",
}

RGB_DEST = {"calib/rgb": "2026-08-28", "calib/th_260914/rgb/cam3": "2026-09-14"}
# 원본으로 대체됐거나 열 수 없는 영상
RGB_DROP = {"KakaoTalk_20260829_183807940.mp4": "원본 4개로 대체됨",
            "17-52-43.mp4": "손상 — 열리지 않음",
            "17-55-48.mp4": "손상 — 열리지 않음"}


def stamp(name):
    m = re.search(r"(\d{8})_(\d{6})", name)
    return (m.group(1), m.group(2)) if m else (None, None)


def load():
    """녹화별 판정 — (경로 → dict)."""
    keep = {}
    for f in ("output/inv_old/inventory.json", "output/inv_260914/inventory.json"):
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            raise SystemExit(f"{f} 가 없습니다. 먼저 inventory.py 를 돌리십시오.")
        for r in json.load(open(p, encoding="utf-8")):
            d = [q["dist"] for q in r["poses"] if q["dist"]]
            keep[r["path"]] = dict(
                r, kind="전체격자" if r["keep"] else None,
                dist=(min(d) if d else None), pat="7x4",
                contrast=(max(q["contrast"] for q in r["poses"])
                          if r["poses"] else 0))
    sv = os.path.join(HERE, "output/salvage.json")
    if os.path.exists(sv):
        for path, v in json.load(open(sv, encoding="utf-8")).items():
            if path in keep and keep[path]["kind"] is None:
                keep[path].update(kind="부분격자", dist=v["dist"],
                                  pat=v["pat"], contrast=v["ct"],
                                  why=f"{v['pat']} 부분격자 {v['n']}프레임 "
                                      f"· 칸 {v['sq']} px · 대비 {v['ct']}℃")
    return keep


def main():
    ap = argparse.ArgumentParser(description="calib 정리")
    ap.add_argument("--apply", action="store_true", help="실제로 옮기고 지운다")
    args = ap.parse_args()
    recs = load()

    moves, dels = [], []
    for path, r in sorted(recs.items(), key=lambda kv: kv[1].get("dist") or 9999):
        base = os.path.splitext(os.path.join(HERE, path))[0]
        ymd, hms = stamp(r["name"])
        group = DATE_OF.get(ymd)
        files = [(base+e, os.path.getsize(base+e))
                 for e in EXTS if os.path.exists(base+e)]
        if r["kind"] is None or group is None:
            dels += [(p, s, r["why"]) for p, s in files]
            continue
        r["dist"] = int(round(r["dist"]))
        nm = f"d{r['dist']:04d}_{ymd}_{hms}"
        dst = os.path.join(CALIB, group, "thermal")
        for p, s in files:
            moves.append((p, os.path.join(dst, nm+os.path.splitext(p)[1]), s))
        r["newname"] = nm

    # RGB
    for src, group in RGB_DEST.items():
        d = os.path.join(HERE, src)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".mp4", ".avi")):
                continue
            p = os.path.join(d, f)
            s = os.path.getsize(p)
            if f in RGB_DROP:
                dels.append((p, s, RGB_DROP[f]))
            else:
                moves.append((p, os.path.join(CALIB, group, "rgb", f), s))

    # _all — 원본과 바이트까지 같은 사본
    ad = os.path.join(CALIB, "_all")
    if os.path.isdir(ad):
        for f in sorted(os.listdir(ad)):
            p = os.path.join(ad, f)
            dels.append((p, os.path.getsize(p), "th/th350/th400/th450 와 동일한 사본"))

    # 같은 시각의 녹화가 두 폴더에 복사돼 있으면 목적지가 겹친다 (th_260914/
    # 와 th_260914_before/ 에 180655·180708 이 양쪽에 있다). 하나만 옮기고
    # 나머지는 사본으로 지운다.
    seen, uniq = {}, []
    for src, dst, s in moves:
        if dst in seen:
            dels.append((src, s, f"{os.path.relpath(seen[dst], HERE)} 와 같은 녹화의 사본"))
            continue
        seen[dst] = src
        uniq.append((src, dst, s))
    moves = uniq

    mv = sum(s for _, _, s in moves)
    dl = sum(s for _, s, _ in dels)
    kept = [r for r in recs.values() if r["kind"]]
    print("=" * 92)
    print(f"{'실행' if args.apply else '계획 (실제로는 아무것도 바꾸지 않음)'}")
    print("=" * 92)
    print(f"  남길 녹화  {len(kept):>3} / {len(recs)}건   "
          f"(전체격자 {sum(1 for r in kept if r['kind']=='전체격자')} · "
          f"부분격자 {sum(1 for r in kept if r['kind']=='부분격자')})")
    ds = sorted(r["dist"] for r in kept)
    print(f"  거리 범위  {ds[0]} ~ {ds[-1]} mm")
    print(f"  옮길 파일  {len(moves):>3}개  {mv/1e6:7.1f} MB")
    print(f"  지울 파일  {len(dels):>3}개  {dl/1e6:7.1f} MB")

    agg = {}
    for p, s, why in dels:
        k = re.sub(r"[\d.]+", "N", why)[:46]
        n, b = agg.get(k, (0, 0))
        agg[k] = (n+1, b+s)
    print("\n  지울 이유별")
    for k, (n, b) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"    {k:<48}{n:>4}개  {b/1e6:7.1f} MB")

    if not args.apply:
        print("\n  --apply 를 붙이면 실행합니다.")
        return 0

    for g in DATE_OF.values():
        for sub in ("thermal", "rgb"):
            os.makedirs(os.path.join(CALIB, g, sub), exist_ok=True)
    with open(os.path.join(CALIB, "DELETED.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["지운 파일", "바이트", "이유"])
        for p, s, why in dels:
            w.writerow([os.path.relpath(p, HERE).replace("\\", "/"), s, why])
    for src, dst, _ in moves:
        shutil.move(src, dst)
    for p, _, _ in dels:
        os.remove(p)
    # 빈 폴더 치우기
    for d in ("_all", "th", "th350", "th400", "th450", "th-test", "th-test-01",
              "rgb", "th_260914"):
        p = os.path.join(CALIB, d)
        for dp, dns, fns in os.walk(p, topdown=False):
            if not fns and not os.listdir(dp):
                os.rmdir(dp)

    with open(os.path.join(CALIB, "INDEX.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["날짜", "파일", "거리_mm", "격자", "대비_C", "자세수",
                    "프레임", "원래폴더", "옛이름", "비고"])
        for r in sorted(kept, key=lambda r: r["dist"]):
            ymd, hms = stamp(r["name"])
            # 사본이라 옮기지 않은 녹화는 목록에 넣지 않는다
            if not os.path.exists(os.path.join(CALIB, DATE_OF[ymd], "thermal",
                                               r["newname"]+".y16raw")):
                continue
            w.writerow([DATE_OF[ymd], r["newname"], r["dist"], r["pat"],
                        r["contrast"], len(r["poses"]), r["frames"],
                        r["group"], OLD_ALIAS.get(f"{ymd}_{hms}", ""),
                        r["kind"]])
    print(f"\n  완료 — calib/INDEX.csv · calib/DELETED.csv 기록")
    return 0


if __name__ == "__main__":
    sys.exit(main())
