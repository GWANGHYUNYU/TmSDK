#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""열화상 엽온의 **절대 오프셋**을 접촉식 센서로 물어봅니다.

    python3 scripts/absolute_offset.py --leaf output/leaftemp/leaf_temp_all.csv

이 과제의 오차 예산은 0.56 ℃ 인데 카메라 사양은 ±2 ℃ 입니다. 그래서
«개체 간 비교» 는 해 왔지만 **절대값은 주장할 수 없었습니다.** 접촉식
엽온센서가 들어오면서 처음으로 물을 수 있게 됐습니다.

★ 한 가지를 혼동하지 마십시오. 접촉식 센서는 **잎 하나**를, 카메라는
  **잎 수십 개의 평균**을 잽니다. 둘의 차이에는 두 가지가 섞여 있습니다.

      (카메라 − 센서) = 카메라의 절대 편향  +  (평균 잎 − 그 잎)

  그래서 «차이의 평균» 을 그대로 카메라 오차라고 부르면 안 됩니다.
  대신 세 가지를 따로 봅니다.

      ① 차이가 **시간에 따라 일정한가** — 일정하면 더하기 상수(편향)
      ② 차이가 **온도에 따라 기우는가** — 기울면 곱하기 오차(이득)
      ③ **소등 중** 차이 — 증산이 멈춰 잎 사이 편차가 줄어드는 때라
        그때의 차이가 카메라 편향에 가장 가깝습니다

★ 센서가 어느 잎에 붙어 있는지는 모릅니다. 그래서 개체마다 센서와의
  상관을 재서, 가장 잘 따라가는 개체도 같이 보고합니다. 그 개체가
  센서가 붙은 잎일 가능성이 높습니다.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT_MIN = 3          # 녹화 슬롯 길이(분)
LIGHTS = (8, 18)      # 점등 08:01 / 소등 18:01


def read_contact(path):
    """접촉식 엽온 → {카메라: {분 단위 시각: 값}}"""
    out = {"151": {}, "152": {}}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            t = datetime.strptime(r["timestamp"].split(" Asia")[0],
                                  "%Y-%m-%d %H:%M:%S.%f").replace(
                                      second=0, microsecond=0)
            for cam in ("151", "152"):
                v = r.get(f"thermal_{cam}", "")
                if v not in ("", None):
                    out[cam][t] = float(v)
    return out


def read_env(path):
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            t = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
            try:
                out[t] = (float(r["온도"]), float(r["습도"]))
            except (ValueError, KeyError):
                pass
    return out


def slot_mean(series, t0, minutes=SLOT_MIN, shift=0):
    """슬롯 구간의 센서 평균. `shift` 분만큼 뒤쪽을 본다 (시계 오차 보정)."""
    v = [series[t0 + timedelta(minutes=shift+i)]
         for i in range(minutes) if t0 + timedelta(minutes=shift+i) in series]
    return float(np.mean(v)) if v else None


def find_lag(series, rows, cam, lo=-60, hi=360, step=15):
    """센서 시계가 얼마나 어긋나 있나 — 카메라와 가장 닮는 당김을 찾는다.

    ★ 이 검사를 건너뛰면 안 됩니다. 접촉식 로거의 시각이 3시간 앞서
      있었는데, 그대로 비교하면 상관 0.56 · 차이 표준편차 3.8 ℃ 가 나오고
      회귀 기울기가 −0.5 로 «이득 오차» 처럼 보입니다. 3시간을 당기면
      상관 0.984 · 표준편차 0.85 ℃ 로 떨어집니다. 잎의 열 시정수는 수십
      초라, 2~3시간 지연은 물리가 아니라 시계입니다.
    """
    base = collections.defaultdict(list)
    for r in rows:
        if r["분류"] == "잎" and r["카메라"] == cam:
            base[(r["날짜"], r["시각"])].append(float(r["엽온_C"]))
    pts = [(datetime.strptime(d+t, "%Y%m%d%H:%M"), float(np.mean(v)))
           for (d, t), v in base.items()]
    best = None
    for sh in range(lo, hi+1, step):
        xs, ys = [], []
        for t0, cv in pts:
            sv = slot_mean(series, t0, shift=sh)
            if sv is not None:
                xs.append(cv)
                ys.append(sv)
        if len(xs) < 20:
            continue
        r = float(np.corrcoef(xs, ys)[0, 1])
        if best is None or r > best[0]:
            best = (r, sh, float(np.std(np.array(xs)-np.array(ys))))
    return best


def es(T):
    """포화수증기압 [kPa] — Tetens."""
    return 0.61078*np.exp(17.27*T/(T + 237.3))


def main():
    ap = argparse.ArgumentParser(description="절대 오프셋 — 접촉식 센서 대조")
    ap.add_argument("--leaf", default="output/leaftemp/leaf_temp_all.csv")
    ap.add_argument("--contact",
                    default="calib/raw/환경센서/전남농업기술원 접촉식엽온센서_2026-09-01_2026-10-01.csv")
    ap.add_argument("--env",
                    default="calib/raw/환경센서/전라남도농업기술원_전체데이터.csv")
    ap.add_argument("--out", default="output/offset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    contact = read_contact(os.path.join(HERE, args.contact))
    env = read_env(os.path.join(HERE, args.env))
    rows = list(csv.DictReader(open(os.path.join(HERE, args.leaf),
                                    encoding="utf-8-sig")))

    print("=" * 86)
    print("열화상 엽온의 절대 오프셋 — 접촉식 센서 대조")
    print("=" * 86)
    print(f"  접촉식  .151 {len(contact['151'])}분 · .152 {len(contact['152'])}분")
    print(f"  환경센서 {len(env)}분")
    print(f"  엽온 CSV {len(rows)}행")

    # ── 슬롯마다: 카메라 잎 평균 · 접촉식 · 기온·습도 ────────────
    bucket = collections.defaultdict(list)
    for r in rows:
        if r["분류"] != "잎":
            continue
        bucket[(r["카메라"], r["날짜"], r["시각"])].append(float(r["엽온_C"]))

    # ── 센서 시계가 맞는지 먼저 확인 ────────────────────────
    print("\n" + "─"*86)
    print("⓪ 센서 시계 점검 — 잎의 열 시정수는 수십 초입니다")
    lag = {}
    for cam in ("151", "152"):
        b = find_lag(contact[cam], rows, cam)
        if b is None:
            continue
        r0, sh, sd = b
        lag[cam] = sh
        r_raw = find_lag(contact[cam], rows, cam, 0, 0, 1)
        print(f"  .{cam}  최적 당김 {sh:>4}분  상관 {r0:+.3f} · 차이 SD {sd:.2f} ℃"
              + (f"   (당기지 않으면 상관 {r_raw[0]:+.3f} · SD {r_raw[2]:.2f} ℃)"
                 if r_raw else ""))
        if abs(sh) >= 30:
            print(f"      ★ {sh}분은 물리로 설명되지 않습니다 — 로거 시계를 "
                  "확인하십시오. 아래 분석은 이만큼 당겨서 합니다.")

    recs = []
    for (cam, day, hhmm), vs in bucket.items():
        t0 = datetime.strptime(day + hhmm, "%Y%m%d%H:%M")
        c = slot_mean(contact[cam], t0, shift=lag.get(cam, 0))
        if c is None:
            continue
        e = [env[t0 + timedelta(minutes=i)]
             for i in range(SLOT_MIN) if t0 + timedelta(minutes=i) in env]
        Ta, RH = (float(np.mean([x[0] for x in e])),
                  float(np.mean([x[1] for x in e]))) if e else (None, None)
        recs.append(dict(카메라=cam, 날짜=day, 시각=hhmm,
                         시=t0.hour + t0.minute/60,
                         점등=int(LIGHTS[0] <= t0.hour < LIGHTS[1]),
                         카메라_잎평균_C=round(float(np.mean(vs)), 3),
                         개체수=len(vs),
                         접촉식_C=round(c, 3), 센서당김_분=lag.get(cam, 0),
                         차이_C=round(float(np.mean(vs)) - c, 3),
                         기온_C=None if Ta is None else round(Ta, 2),
                         습도_pct=None if RH is None else round(RH, 1)))
    if not recs:
        raise SystemExit("겹치는 슬롯이 없습니다 — 기간을 확인하십시오.")
    recs.sort(key=lambda r: (r["카메라"], r["날짜"], r["시각"]))
    print(f"  → 접촉식과 겹치는 슬롯 {len(recs)}개 "
          f"({min(r['날짜'] for r in recs)} ~ {max(r['날짜'] for r in recs)})")

    # ── ① 차이가 일정한가 ────────────────────────────────────
    print("\n" + "─"*86)
    print("① 카메라 − 접촉식  (단위 ℃)")
    print(f"  {'카메라':>6}{'구분':>6}{'슬롯':>6}{'평균':>9}{'중앙':>9}"
          f"{'표준편차':>10}{'10%':>8}{'90%':>8}")
    summ = {}
    for cam in ("151", "152"):
        for on, lab in ((1, "점등"), (0, "소등"), (None, "전체")):
            g = [r for r in recs if r["카메라"] == cam
                 and (on is None or r["점등"] == on)]
            if not g:
                continue
            d = np.array([r["차이_C"] for r in g])
            summ[(cam, lab)] = d
            print(f"  {cam:>6}{lab:>6}{len(d):>6}{d.mean():>+8.3f}"
                  f"{np.median(d):>+9.3f}{d.std():>10.3f}"
                  f"{np.percentile(d,10):>+8.2f}{np.percentile(d,90):>+8.2f}")

    # ── ② 온도에 따라 기우는가 ───────────────────────────────
    print("\n" + "─"*86)
    print("② 이득 오차 — 접촉식 온도에 대한 회귀  (차이 = a·접촉식 + b)")
    print(f"  {'카메라':>6}{'기울기 a':>11}{'절편 b':>10}{'상관 r':>9}"
          f"{'관측범위 영향':>16}{'판정':>22}")
    for cam in ("151", "152"):
        g = [r for r in recs if r["카메라"] == cam]
        if len(g) < 10:
            continue
        x = np.array([r["접촉식_C"] for r in g])
        y = np.array([r["차이_C"] for r in g])
        a, b = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        # ★ 기울기만 보고 판정하면 안 됩니다. 같은 기울기라도 관측 온도
        #   범위가 좁으면 실제 영향은 작습니다. «이 데이터에서 실제로
        #   몇 ℃ 벌어지는가» 로 바꿔 씁니다.
        span = float(x.max() - x.min())
        eff = abs(a)*span
        v = ("상수 보정으로 충분" if eff < 0.56 else
             f"상수 보정만으로는 부족")
        print(f"  {cam:>6}{a:>+11.4f}{b:>+10.3f}{r:>+9.3f}"
              f"{f'{eff:.2f} ℃ / {span:.1f} ℃':>16}{v:>22}")

    # ── ③ 소등 중 차이 = 카메라 편향에 가장 가까운 값 ─────────
    print("\n" + "─"*86)
    print("③ 소등 중 차이 — 증산이 멈춰 잎 사이 편차가 줄어드는 때")
    for cam in ("151", "152"):
        d = summ.get((cam, "소등"))
        if d is None:
            continue
        se = d.std()/np.sqrt(len(d))
        print(f"  .{cam}  보정 후보 {d.mean():+.3f} ℃  "
              f"(95 % 구간 {d.mean()-1.96*se:+.3f} ~ {d.mean()+1.96*se:+.3f}, "
              f"n={len(d)})")
    print("  → 이 값을 빼면 카메라 엽온이 접촉식과 평균적으로 일치합니다.")
    print("    다만 «평균 잎 − 센서 잎» 이 섞여 있으므로 상한으로 보십시오.")

    # ── ④ 센서가 붙은 잎 찾기 ────────────────────────────────
    print("\n" + "─"*86)
    print("④ 접촉식을 가장 잘 따라가는 개체 — 센서가 붙은 잎일 가능성")
    per = collections.defaultdict(dict)
    for r in rows:
        if r["분류"] != "잎":
            continue
        per[(r["카메라"], r["개체"])][(r["날짜"], r["시각"])] = float(r["엽온_C"])
    ref = {(r["카메라"], r["날짜"], r["시각"]): r["접촉식_C"] for r in recs}
    for cam in ("151", "152"):
        best = []
        for (c, key), series in per.items():
            if c != cam:
                continue
            xy = [(v, ref[(c, d, t)]) for (d, t), v in series.items()
                  if (c, d, t) in ref]
            if len(xy) < 20:
                continue
            a = np.array(xy)
            rr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
            best.append((rr, float(np.mean(a[:, 0]-a[:, 1])), key, len(xy)))
        if not best:
            continue
        # ★ 시계를 맞추고 나면 **모든** 개체가 r≈0.98 로 똑같이 잘 따라가
        #   상관으로는 가릴 수 없습니다. 센서가 붙은 잎이라면 차이가 가장
        #   작아야 하므로, 차이의 절대값으로 줄 세웁니다.
        best.sort(key=lambda z: abs(z[1]))
        rs = [z[0] for z in best]
        print(f"  .{cam}  개체 {len(best)}개 · 상관 범위 "
              f"{min(rs):+.3f}~{max(rs):+.3f} — 상관으로는 못 가립니다")
        print(f"     차이가 작은 순:")
        for rr, bias, key, n in best[:3]:
            print(f"       {bias:+.3f} ℃  (r={rr:+.3f})  {key}")
        rr, bias, key, n = best[-1]
        print(f"     (가장 큰) {bias:+.3f} ℃  {key}")
        sp = abs(best[-1][1] - best[0][1])
        print(f"     개체 간 폭 {sp:.2f} ℃ — 편향 {abs(best[0][1]):.1f} ℃ 보다 "
              f"{'작습니다 → 편향은 개체차로 설명 안 됩니다' if sp < abs(best[0][1]) else '큽니다'}")

    # ── ⑤ leaf-to-air VPD ───────────────────────────────────
    print("\n" + "─"*86)
    print("⑤ leaf-to-air VPD — 이 과제의 목표량  (단위 kPa)")
    print("     VPD_leaf = es(T_leaf) − RH/100 × es(T_air)")
    print(f"  {'카메라':>6}{'구분':>6}{'공기 VPD':>10}{'잎 VPD(보정전)':>15}"
          f"{'잎 VPD(보정후)':>15}{'목표 이탈':>10}")
    for cam in ("151", "152"):
        off = summ.get((cam, "소등"))
        off = float(off.mean()) if off is not None else 0.0
        for on, lab in ((1, "점등"), (0, "소등")):
            g = [r for r in recs if r["카메라"] == cam and r["점등"] == on
                 and r["기온_C"] is not None]
            if not g:
                continue
            Tl = np.array([r["카메라_잎평균_C"] for r in g])
            Ta = np.array([r["기온_C"] for r in g])
            RH = np.array([r["습도_pct"] for r in g])/100
            air = es(Ta)*(1-RH)
            raw = es(Tl) - RH*es(Ta)
            cor = es(Tl-off) - RH*es(Ta)
            out = 100*float(((cor < 0.6) | (cor > 1.2)).mean())
            print(f"  {cam:>6}{lab:>6}{air.mean():>10.3f}{raw.mean():>15.3f}"
                  f"{cor.mean():>15.3f}{out:>9.0f}%")

    p = os.path.join(args.out, "offset_slots.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\n  저장  {p}  ({len(recs)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
