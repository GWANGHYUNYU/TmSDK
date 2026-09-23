#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""뽑을 수 있는 엽온을 전부 CSV 로 — 마스크가 유효한 날만.

    python3 scripts/leaf_temp_export.py \
        output/annot_in/annotations_2026-09-23_labeled.json \
        --out output/leaftemp

어노테이션은 **.151 카메라의 2026-09-18 08:00 프레임 한 장**에 칠했습니다.
그 마스크를 다른 날에 그대로 쓰려면 «그 사이에 장면이 안 움직였다» 가
성립해야 합니다. 잎은 바람·생장으로 움직이므로 날마다 확인해야 합니다.

★ 이 스크립트의 핵심은 온도를 재는 것이 아니라, **잴 자격이 있는 날을
  가르는 것**입니다. 자격 심사 없이 숫자만 뽑으면, 마스크가 엉뚱한 자리를
  덮은 채로 소수점 셋째 자리까지 그럴듯한 값이 나옵니다.

판정 방법 — 기준 프레임의 «구조 영상»(기울기 크기)과 각 슬롯을 상호상관해
  ① 최대상관이 얼마나 높은가
  ② 최대가 되는 이동량이 (0,0) 인가
  ③ 그 이동량이 슬롯마다 «같은 값» 인가        ← 이것이 제일 중요합니다
를 봅니다. ③ 이 무너지면 상관 봉우리가 잡음이라는 뜻이고, 슬롯마다 최적
이동이 제각각이라 «하나의 마스크로 하루를 설명한다» 가 성립하지 않습니다.

다른 카메라(.152)에는 쓸 수 없습니다 — 화각이 다릅니다. 중간보고 §8 참조.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

_spec = importlib.util.spec_from_file_location(
    "lts", os.path.join(HERE, "scripts", "leaf_temp_series.py"))
LTS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LTS)

# 낮 시간대(점등 09~17시)에만 장면 정합을 잽니다. 소등 중에는 잎 경계가
# 열화상에 없어서 «움직였는지» 자체를 물을 수 없습니다.
DAY_H = (9, 17)
MIN_CORR = 0.20      # 최대상관이 이보다 낮으면 정합 근거 없음
MIN_AGREE = 0.60     # 최적 이동이 같은 값으로 모이는 슬롯 비율


def struct(x):
    g = cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.0)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0),
                    cv2.Sobel(g, cv2.CV_32F, 0, 1))


def best_shift(a, b, rad=10):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a*a).sum()*(b*b).sum()) + 1e-12
    return max(((float((a*np.roll(np.roll(b, dy, 0), dx, 1)).sum()/den), dx, dy)
                for dy in range(-rad, rad+1) for dx in range(-rad, rad+1)))


def read_slots(raw, cam, day):
    """(시각표기, 시(소수), 평균온도맵, 프레임수) 목록. 빈 파일은 건너뜁니다."""
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, raw,
                                           f"{cam}_{day}_*.y16raw"))):
        if os.path.getsize(p) < 1e6:
            continue
        t = re.search(r"_(\d{6})\.y16raw", p).group(1)
        try:
            cels, ok, tot = LTS.slot_mean(p, full=True)
        except Exception:
            continue
        out.append((f"{t[:2]}:{t[2:4]}", int(t[:2])+int(t[2:4])/60,
                    cels, ok, tot))
    return out


def judge(slots, ref):
    """마스크를 이 날에 써도 되는가 — (판정, 상관, dx, dy, 일관성)."""
    r = [best_shift(ref, struct(s[2])) for s in slots
         if DAY_H[0] <= int(s[0][:2]) <= DAY_H[1]]
    if not r:
        return "판정불가", float("nan"), 0, 0, float("nan")
    a = np.array(r)
    dx, dy = int(np.median(a[:, 1])), int(np.median(a[:, 2]))
    agree = float(((a[:, 1] == dx) & (a[:, 2] == dy)).mean())
    corr = float(np.median(a[:, 0]))
    if corr < MIN_CORR or agree < MIN_AGREE:
        v = "불가"
    elif (dx, dy) != (0, 0):
        v = "불가"          # 밀린 장면. 통째로 밀어도 잎은 제각각 움직였다.
    elif corr < 0.40:
        v = "주의"
    else:
        v = "확인"
    return v, corr, dx, dy, agree


def main():
    ap = argparse.ArgumentParser(description="엽온 전량 CSV 추출")
    ap.add_argument("annot")
    ap.add_argument("--package", default="output/annotate_v2")
    ap.add_argument("--params", default="params/thermal_rgb_stereo.npz")
    ap.add_argument("--cam", default="192_168_0_151")
    ap.add_argument("--ref-day", default="20260918")
    ap.add_argument("--ref-slot", default="080000",
                    help="어노테이션을 칠한 슬롯 (정합 기준)")
    ap.add_argument("--raw", default="calib/raw/th/raw_output")
    ap.add_argument("--out", default="output/leaftemp")
    ap.add_argument("--include-invalid", action="store_true",
                    help="마스크 판정 «불가» 인 날도 CSV 에 넣는다 (기본 제외)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    objs = LTS.masks_from(args.annot, args.package, args.params)
    polys = [o["poly"] for o in objs]
    print("=" * 88)
    print(f"엽온 전량 추출 · 카메라 {args.cam[-3:]} · 기준 "
          f"{args.ref_day} {args.ref_slot[:2]}:{args.ref_slot[2:4]}")
    print("=" * 88)
    print(f"  개체 {len(objs)}개 "
          f"(잎 {sum(1 for o in objs if o['cls']=='잎')} · "
          f"꽃 {sum(1 for o in objs if o['cls']=='꽃')} · "
          f"딸기 {sum(1 for o in objs if o['cls']=='딸기')})")

    refp = os.path.join(HERE, args.raw,
                        f"{args.cam}_{args.ref_day}_{args.ref_slot}.y16raw")
    ref = struct(LTS.slot_mean(refp)[0])

    days = sorted({re.search(r"_(\d{8})_", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(
                       HERE, args.raw, f"{args.cam}_*.y16raw"))})
    print(f"  이 카메라의 날짜 {len(days)}개: {' '.join(days)}\n")

    print(f"  {'날짜':>10}{'슬롯':>6}{'주간':>6}{'상관':>8}{'이동':>9}"
          f"{'일관성':>8}  판정")
    rows, summary = [], []
    for day in days:
        slots = read_slots(args.raw, args.cam, day)
        if not slots:
            print(f"  {day:>10}{0:>6}     — 쓸 수 있는 슬롯 없음")
            continue
        nday = sum(1 for s in slots if DAY_H[0] <= int(s[0][:2]) <= DAY_H[1])
        v, corr, dx, dy, agree = judge(slots, ref)
        print(f"  {day:>10}{len(slots):>6}{nday:>6}{corr:>8.3f}"
              f"{f'({dx:+d},{dy:+d})':>9}{agree*100:>7.0f}%  {v}"
              + ("" if v != "불가" else "  ← 마스크 못 씀"))
        summary.append(dict(날짜=day, 슬롯수=len(slots), 주간슬롯=nday,
                            상관=round(corr, 3), dx=dx, dy=dy,
                            일관성=round(agree, 3), 마스크판정=v))
        # ★ «판정불가» 도 빼야 한다. 주간 슬롯이 없어 심사를 못 한 날은
        #   «통과» 가 아니라 «모름» 이다. 09-14 가 그런 날인데, 하필
        #   체커보드를 들고 찍은 캘리브레이션 촬영일이라 장면이 아예 다르다.
        if v in ("불가", "판정불가") and not args.include_invalid:
            continue
        for lab, hh, cels, ok, tot in slots:
            fit, gmag = LTS.edge_fit(cels, polys)
            for o in objs:
                rows.append({
                    "날짜": day, "시각": lab, "시": round(hh, 3),
                    "카메라": args.cam[-3:], "개체": o["key"],
                    "출처프레임": o["src"], "분류": o["cls"], "층_mm": o["layer"],
                    "면적_px": o["area"],
                    "엽온_C": round(float(cels[o["mask"]].mean()), 3),
                    "화면평균_C": round(float(cels.mean()), 3),
                    "유효프레임": ok, "전체프레임": tot,
                    "슬롯온전": int(ok >= 0.95*tot), "마스크판정": v,
                    "마스크적합": round(fit, 3),
                    "열구조세기": round(gmag, 3),
                    "점등": int(8 <= hh < 18),
                })

    if not rows:
        raise SystemExit("\n  마스크가 유효한 날이 없습니다.")

    # ── 교차검증: 개체별 «편차» 가 기준일과 같은가 ────────────────────
    # ★ 화면 상호상관보다 이쪽이 낫습니다. 상호상관은 그 날의 열 구조가
    #   얼마나 뚜렷한지에 좌우되지만, 개체별 편차는 «마스크가 같은 잎을
    #   덮고 있는가» 만 묻습니다. 실제로 09-17 은 상관 0.300(주의)인데
    #   편차 상관은 +0.954 로, 같은 날 앞뒤 절반끼리(0.878)보다 높습니다.
    leaf_keys = [o["key"] for o in objs if o["cls"] == "잎"]
    bykey = {}
    for r in rows:
        if r["점등"] and r["분류"] == "잎":
            bykey.setdefault((r["날짜"], r["시각"]), {})[r["개체"]] = r["엽온_C"]

    def devvec(day):
        ss = [k for k in bykey if k[0] == day]
        if not ss:
            return None
        M = np.array([[bykey[s][k] for k in leaf_keys] for s in ss])
        return (M - M.mean(1, keepdims=True)).mean(0)

    base = devvec(args.ref_day)
    print(f"\n  교차검증 — 개체별 편차가 기준일({args.ref_day})과 같은가")
    for s in summary:
        v = devvec(s["날짜"])
        s["개체편차상관"] = ("" if v is None or base is None
                       else round(float(np.corrcoef(v, base)[0, 1]), 3))
        if s["개체편차상관"] == "":
            continue
        c = s["개체편차상관"]
        mark = ("동일 개체를 덮고 있음" if c >= 0.85 else
                "확인 필요" if c >= 0.60 else "★ 다른 자리를 덮고 있음")
        print(f"    {s['날짜']}  r = {c:+.3f}   {mark}")

    p1 = os.path.join(args.out, "leaf_temp_all.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 개체별 요약 — 긴 표(3577행)를 그대로 보기는 어렵습니다.
    p3 = os.path.join(args.out, "leaf_temp_objects.csv")
    agg = {}
    for r in rows:
        k = (r["날짜"], r["개체"])
        a = agg.setdefault(k, dict(날짜=r["날짜"], 개체=r["개체"],
                                   분류=r["분류"], 층_mm=r["층_mm"],
                                   면적_px=r["면적_px"], _on=[], _off=[]))
        (a["_on"] if r["점등"] else a["_off"]).append(r["엽온_C"])
    orows = []
    for a in agg.values():
        on, off = np.array(a.pop("_on")), np.array(a.pop("_off"))
        a["점등슬롯"] = len(on)
        a["소등슬롯"] = len(off)
        a["점등_평균_C"] = round(float(on.mean()), 3) if on.size else ""
        a["소등_평균_C"] = round(float(off.mean()), 3) if off.size else ""
        a["주야차_C"] = (round(float(on.mean()-off.mean()), 3)
                      if on.size and off.size else "")
        orows.append(a)
    # 같은 슬롯의 잎 평균 대비 편차 — 개체 «고유» 차이는 이 열로 봅니다.
    for day in sorted({r["날짜"] for r in rows}):
        for on in (1, 0):
            ss = sorted({r["시각"] for r in rows
                         if r["날짜"] == day and r["점등"] == on})
            if not ss:
                continue
            base = {s: np.mean([r["엽온_C"] for r in rows
                                if r["날짜"] == day and r["시각"] == s
                                and r["분류"] == "잎"]) for s in ss}
            per = {}
            for r in rows:
                if r["날짜"] == day and r["점등"] == on:
                    per.setdefault(r["개체"], []).append(
                        r["엽온_C"] - base[r["시각"]])
            col = "점등_편차_C" if on else "소등_편차_C"
            for a in orows:
                if a["날짜"] == day and a["개체"] in per:
                    a[col] = round(float(np.mean(per[a["개체"]])), 3)
    cols = ["날짜", "개체", "분류", "층_mm", "면적_px", "점등슬롯", "소등슬롯",
            "점등_평균_C", "소등_평균_C", "주야차_C",
            "점등_편차_C", "소등_편차_C"]
    with open(p3, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(orows, key=lambda a: (a["날짜"], a["개체"])))

    p2 = os.path.join(args.out, "leaf_temp_days.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    days_in = sorted({r["날짜"] for r in rows})
    print(f"\n  저장  {p1}  ({len(rows)}행 · 날짜 {len(days_in)}개"
          f" · 슬롯 {len(rows)//max(len(objs),1)}개)")
    print(f"        {p2}  (날짜별 판정 {len(summary)}행)")
    print(f"        {p3}  (개체별 요약 {len(orows)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
