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
import time

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
MAX_FIX_PX = 2       # 이 이하로 «일관되게» 밀린 것은 되밀어 쓴다


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


def th_files(raw, pat):
    """raw 아래를 **재귀로** 훑습니다.

    ★ 한 폴더만 보면 안 됩니다. 현장에서 날짜·카메라별로 하위 폴더를
      만들어 넣는 일이 있는데(calib/raw/th/151/), 그러면 조용히 그만큼을
      빼고 계산합니다 — 09-19·09-20 의 91슬롯이 그렇게 빠질 뻔했습니다.
    """
    return sorted(glob.glob(os.path.join(HERE, raw, "**", pat),
                            recursive=True))


def shift_obj(o, dx, dy):
    """개체 마스크·폴리곤을 (dx, dy) 만큼 옮긴 사본.

    ★ 방향 주의. best_shift(ref, b) 는 «b 를 (dx,dy) 만큼 굴리면 ref 와
      가장 닮는다» 를 줍니다. 마스크는 ref 좌표계에 있으므로, b(그 날)
      좌표계로 보내려면 **반대로** 굴려야 합니다. 부호를 뒤집으면 오차가
      두 배가 되므로, 적용 뒤 적합도로 확인합니다.
    """
    m = np.roll(np.roll(o["mask"], -dy, axis=0), -dx, axis=1)
    p = o["poly"].copy()
    p[:, 0] -= dx
    p[:, 1] -= dy
    return {**o, "mask": m, "poly": p}


def write_csv(path, rows, fields=None):
    """CSV 를 씁니다. 파일이 잠겨 있으면 옆에 이름을 바꿔 씁니다.

    ★ 엑셀로 열어 둔 CSV 는 Windows 에서 «쓰기 금지» 입니다. 그냥 두면
      수십 분 돌린 계산이 마지막 줄에서 PermissionError 로 통째로 날아갑니다.
      결과를 잃는 것보다 파일 이름이 하나 늘어나는 편이 낫습니다.
    """
    f = fields or list(rows[0].keys())
    for p in (path, None):
        target = p or (os.path.splitext(path)[0] +
                       f"_{int(time.time())}.csv")
        try:
            with open(target, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=f, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            if p is None:
                print(f"  ⚠ {os.path.basename(path)} 가 잠겨 있어 "
                      f"{os.path.basename(target)} 로 저장했습니다 "
                      "(엑셀에서 닫고 다시 돌리면 제자리에 씁니다)")
            return target
        except PermissionError:
            continue
    raise IOError(path)


def read_slots(raw, cam, day):
    """(시각표기, 시(소수), 평균온도맵, 프레임수) 목록. 빈 파일은 건너뜁니다."""
    out = []
    for p in th_files(raw, f"{cam}_{day}_*.y16raw"):
        if os.path.getsize(p) < 1e6:
            continue
        t = re.search(r"_(\d{6})\.y16raw", p).group(1)
        try:
            cels, ok, tot = LTS.slot_mean(p, full=True)
        except Exception:
            continue
        out.append((f"{t[:2]}:{t[2:4]}", int(t[:2])+int(t[2:4])/60,
                    cels, ok, tot))
    # ★ 경로 순서 ≠ 시각 순서. 녹화가 여러 하위 폴더에 나뉘면 폴더가 먼저
    #   갈라져 «18:01 다음 00:00» 같은 순서가 됩니다.
    out.sort(key=lambda s: s[1])
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
    elif max(abs(dx), abs(dy)) > MAX_FIX_PX:
        # 많이 밀린 장면. 통째로 밀어도 잎은 제각각 움직였을 것이다.
        v = "불가"
    elif (dx, dy) != (0, 0):
        # ★ «일관되게» 1~2 화소 밀린 것은 버릴 이유가 없다. 슬롯마다
        #   최적 이동이 제각각이면(일관성 낮음) 위에서 이미 걸러졌고,
        #   여기 남은 것은 카메라가 살짝 건드려진 경우다. 그 값만큼
        #   마스크를 되밀면 된다. 09-19 는 (+1,0) 하나로 48슬롯이
        #   통째로 버려질 뻔했다. 보정 여부는 CSV 에 남긴다.
        v = "보정"
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
    ap.add_argument("--cams", default="",
                    help="비우면 raw 폴더의 카메라를 전부 훑는다")
    ap.add_argument("--cam", default="192_168_0_151",
                    help="기준 프레임이 속한 카메라")
    ap.add_argument("--cam-ref",
                    default="151=20260918_080000,152=20260921_140000",
                    help="카메라별 기준 슬롯 «번호=날짜_시각»")
    ap.add_argument("--raw", default="calib/raw/th",
                    help="아래 하위 폴더까지 재귀로 훑습니다")
    ap.add_argument("--out", default="output/leaftemp")
    ap.add_argument("--include-invalid", action="store_true",
                    help="마스크 판정 «불가» 인 날도 CSV 에 넣는다 (기본 제외)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 88)
    print("엽온 전량 추출")
    print("=" * 88)

    # ★ 카메라마다 «자기» 마스크와 «자기» 기준 슬롯을 씁니다. 예전에는
    #   .151 하나뿐이라 전역으로 두었는데, .152 어노테이션이 생긴 뒤로는
    #   그대로 두면 .151 마스크를 .152 에 들이대게 됩니다.
    refs = {}
    for kv in args.cam_ref.split(","):
        c, v = kv.split("=")
        refs[c.strip()] = v.strip()
    print("  기준 슬롯: " + "  ".join(f".{c} {v}" for c, v in refs.items()))

    allf = th_files(args.raw, "*.y16raw")
    cams = ([c.strip() for c in args.cams.split(",") if c.strip()] or
            sorted({re.match(r"(.+)_\d{8}_\d{6}\.y16raw",
                             os.path.basename(p)).group(1) for p in allf}))
    print(f"  카메라 {len(cams)}대: {' '.join(c[-3:] for c in cams)}")

    rows, summary = [], []
    for cam in cams:
        short = cam[-3:]
        objs = LTS.masks_from(args.annot, args.package, args.params,
                              camera=short)
        days = sorted({re.search(r"_(\d{8})_", os.path.basename(p)).group(1)
                       for p in th_files(args.raw, f"{cam}_*.y16raw")})
        print(f"\n── 카메라 {short} · 날짜 {len(days)}개 "
              f"({days[0]} ~ {days[-1]}) " + "─"*30)
        if not objs:
            print("  이 카메라의 어노테이션이 없습니다 — 건너뜁니다")
            for day in days:
                summary.append(dict(
                    카메라=short, 날짜=day,
                    슬롯수=len(th_files(args.raw, f"{cam}_{day}_*.y16raw")),
                    주간슬롯="", 상관="", dx="", dy="", 일관성="",
                    마스크판정="어노테이션 없음"))
            continue
        polys = [o["poly"] for o in objs]
        oth = sum(1 for o in objs if o["cls"] not in ("잎", "꽃", "딸기"))
        print(f"  개체 {len(objs)}개 "
              f"(잎 {sum(1 for o in objs if o['cls']=='잎')} · "
              f"꽃 {sum(1 for o in objs if o['cls']=='꽃')} · "
              f"딸기 {sum(1 for o in objs if o['cls']=='딸기')} · "
              f"기타 {oth})")
        if short not in refs:
            print("  기준 슬롯이 지정되지 않았습니다 — 건너뜁니다")
            continue
        cand = th_files(args.raw, f"{cam}_{refs[short]}.y16raw")
        if not cand:
            print(f"  기준 슬롯 {refs[short]} 를 못 찾았습니다 — 건너뜁니다")
            continue
        rc = LTS.slot_mean(cand[0])[0]
        ref = struct(rc)
        print(f"  기준 {refs[short]}  ·  마스크 적합도 "
              f"{LTS.edge_fit(rc, polys)[0]:.3f}")

        print(f"  {'날짜':>10}{'슬롯':>6}{'주간':>6}{'상관':>8}{'이동':>9}"
              f"{'일관성':>8}  판정")
        for day in days:
            slots = read_slots(args.raw, cam, day)
            if not slots:
                print(f"  {day:>10}{0:>6}     — 쓸 수 있는 슬롯 없음")
                continue
            nday = sum(1 for s in slots
                       if DAY_H[0] <= int(s[0][:2]) <= DAY_H[1])
            v, corr, dx, dy, agree = judge(slots, ref)
            print(f"  {day:>10}{len(slots):>6}{nday:>6}{corr:>8.3f}"
                  f"{f'({dx:+d},{dy:+d})':>9}{agree*100:>7.0f}%  {v}"
                  + ("" if v != "불가" else "  ← 마스크 못 씀"))
            srow = dict(카메라=cam[-3:], 날짜=day, 슬롯수=len(slots),
                        주간슬롯=nday, 상관=round(corr, 3),
                        dx=dx, dy=dy, 일관성=round(agree, 3),
                        마스크판정=v)
            summary.append(srow)
        # ★ «판정불가» 도 빼야 한다. 주간 슬롯이 없어 심사를 못 한 날은
        #   «통과» 가 아니라 «모름» 이다. 09-14 가 그런 날인데, 하필
        #   체커보드를 들고 찍은 캘리브레이션 촬영일이라 장면이 아예 다르다.
            if v in ("불가", "판정불가") and not args.include_invalid:
                continue
            use, upoly = objs, polys
            if (dx, dy) != (0, 0):
                # ★ 되밀어 보고 «나아졌을 때만» 씁니다. 상호상관이 1화소를
                #   가리켜도 그것이 실제 이동이 아니라 잡음일 수 있습니다.
                #   적합도가 떨어지면 그게 증거이므로 보정을 버립니다 —
                #   재 보고 나빠졌는데도 적용하면, 측정을 해 놓고 무시하는
                #   꼴이 됩니다.
                cand = [shift_obj(o, dx, dy) for o in objs]
                cpoly = [o["poly"] for o in cand]
                c = slots[len(slots)//2][2]
                f0 = LTS.edge_fit(c, polys)[0]
                f1 = LTS.edge_fit(c, cpoly)[0]
                if f1 >= f0:
                    use, upoly = cand, cpoly
                    print(f"      보정 ({dx:+d},{dy:+d}) 적용 — 적합도 "
                          f"{f0:.3f} → {f1:.3f}")
                    srow["적합도_보정전"] = round(f0, 3)
                    srow["적합도_보정후"] = round(f1, 3)
                else:
                    print(f"      보정 ({dx:+d},{dy:+d}) 버림 — 적합도가 "
                          f"{f0:.3f} → {f1:.3f} 로 나빠집니다")
                    dx = dy = 0
                    v = "확인" if corr >= 0.40 else "주의"
                    srow.update(dx=0, dy=0, 마스크판정=v)
            for lab, hh, cels, ok, tot in slots:
                fit, gmag = LTS.edge_fit(cels, upoly)
                for o in use:
                    rows.append({
                        "날짜": day, "시각": lab, "시": round(hh, 3),
                        "카메라": cam[-3:], "개체": o["key"],
                        "출처프레임": o["src"], "분류": o["cls"],
                        "층_mm": o["layer"], "면적_px": o["area"],
                        "엽온_C": round(float(cels[o["mask"]].mean()), 3),
                        "화면평균_C": round(float(cels.mean()), 3),
                        "유효프레임": ok, "전체프레임": tot,
                        "슬롯온전": int(ok >= 0.95*tot), "마스크판정": v,
                        "보정dx": dx, "보정dy": dy,
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
    leaf_keys = sorted({r["개체"] for r in rows if r["분류"] == "잎"})
    bykey = {}
    for r in rows:
        if r["점등"] and r["분류"] == "잎":
            bykey.setdefault((r["카메라"], r["날짜"], r["시각"]),
                             {})[r["개체"]] = r["엽온_C"]

    def devvec(cam, day):
        """카메라까지 봐야 한다 — 날짜만 보면 .152 행이 .151 값을 끌어온다."""
        ss = [k for k in bykey if k[0] == cam and k[1] == day]
        if not ss:
            return None
        ks = [k for k in leaf_keys if k in bykey[ss[0]]]
        M = np.array([[bykey[s][k] for k in ks] for s in ss])
        return (M - M.mean(1, keepdims=True)).mean(0)

    base = {c: devvec(c, refs[c].split("_")[0]) for c in refs}
    print("\n  교차검증 — 개체별 편차가 그 카메라의 기준일과 같은가")
    for s in summary:
        v = devvec(s["카메라"], s["날짜"])
        b0 = base.get(s["카메라"])
        s["개체편차상관"] = ("" if v is None or b0 is None or len(v) != len(b0)
                       else round(float(np.corrcoef(v, b0)[0, 1]), 3))
        if s["개체편차상관"] == "":
            continue
        c = s["개체편차상관"]
        mark = ("동일 개체를 덮고 있음" if c >= 0.85 else
                "확인 필요" if c >= 0.60 else "★ 다른 자리를 덮고 있음")
        print(f"    {s['카메라']} {s['날짜']}  r = {c:+.3f}   {mark}")

    p1 = os.path.join(args.out, "leaf_temp_all.csv")
    write_csv(p1, rows)
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
    write_csv(p3, sorted(orows, key=lambda a: (a["날짜"], a["개체"])), cols)

    p2 = os.path.join(args.out, "leaf_temp_days.csv")
    write_csv(p2, summary)

    days_in = sorted({r["날짜"] for r in rows})
    print(f"\n  저장  {p1}  ({len(rows)}행 · 날짜 {len(days_in)}개"
          f" · 슬롯 {len(rows)//max(len(objs),1)}개)")
    print(f"        {p2}  (날짜별 판정 {len(summary)}행)")
    print(f"        {p3}  (개체별 요약 {len(orows)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
