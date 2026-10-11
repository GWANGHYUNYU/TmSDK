#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""세 가지 엽온 측정을 한자리에 놓고 대조합니다.

    python3 scripts/sensor_crosscheck.py --leaf output/leaftemp_1011b/leaf_temp_all.csv

    ① 열화상 카메라     개체 수십 개의 마스크 평균
    ② 접촉식 (열전대)   구역당 1채널, 1분 간격        ← 유일한 «비(非) 적외선» 기준
    ③ 비접촉식 SIL-411  구역당 2채널, 10분 간격        ← 적외선, 카메라와 같은 계열

★ ②와 ③은 원리가 다릅니다. 접촉식은 방사율과 무관하고, SIL-411 은 카메라와
  똑같이 방사율 가정을 씁니다. 그래서

      카메라 vs SIL-411  차이  →  주로 «시야·거리·보정» 문제
      카메라 vs 접촉식   차이  →  위에 더해 «방사율 가정» 까지 포함

  둘을 같이 봐야 +3 ℃ 가 어디서 왔는지 좁혀집니다.

★ 측정 위치 (260915 PPT «엽온 센서 측정 위치»)

      .151 (일사센서 없는 구역)  맨 윗줄 6-4~6-2 · 3번째 열 3-3~3-1
      .152 (일사센서 있는 구역)  맨 윗줄 6-4.5~6-3 · 셋째줄 1x2~1x23

  구역마다 **서로 다른 두 줄** 에 하나씩 붙어 있습니다. 그래서 카메라
  전체 잎 평균과 비교하면 안 되고, **층별로** 따로 봐야 합니다.
  어느 채널이 어느 층인지는 이름이 아니라 **데이터로** 정합니다.

★ 모든 채널에 시계 검사를 겁니다. 접촉식 로거가 3시간 앞서 있었는데,
  그대로 쓰면 상관 0.56 · 차이 SD 3.8 ℃ 가 나와 «이득 오차» 라는 가짜
  결론이 됩니다. 잎의 열 시정수는 수십 초입니다.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOT_MIN = 3
LAYERS = ("420", "700", "1000")


# ── 읽기 ─────────────────────────────────────────────────────────
def read_contact(path):
    out = collections.defaultdict(dict)
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            t = datetime.strptime(r["timestamp"].split(" Asia")[0],
                                  "%Y-%m-%d %H:%M:%S.%f").replace(
                                      second=0, microsecond=0)
            for cam in ("151", "152"):
                v = r.get(f"thermal_{cam}", "")
                if v not in ("", None):
                    out[(cam, "접촉식")][t] = float(v)
    return out


def read_sil(pattern):
    """SIL-411 비접촉식 — 구역마다 Target 2채널."""
    import openpyxl
    out = collections.defaultdict(dict)
    for p in sorted(glob.glob(pattern)):
        cam = "151" if "151" in os.path.basename(p) else "152"
        wb = openpyxl.load_workbook(p, read_only=True)
        ws = wb["Config 1"]
        for r in ws.iter_rows(min_row=4, values_only=True):
            t = r[0]
            if not isinstance(t, datetime):
                continue
            t = t.replace(second=0, microsecond=0)
            for ci, nm in ((1, "비접촉A"), (3, "비접촉B")):
                if isinstance(r[ci], (int, float)):
                    out[(cam, nm)][t] = float(r[ci])
        wb.close()
    return out


def read_env(path):
    out = {}
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                out[datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")] = (
                    float(r["온도"]), float(r["습도"]), float(r["일사"]))
            except (ValueError, KeyError, TypeError):
                pass
    return out


# ── 도구 ─────────────────────────────────────────────────────────
def window_mean(series, t0, minutes=SLOT_MIN, shift=0, tol=10):
    """슬롯 구간 평균. 10분 간격 센서를 위해 ±tol 분까지 당겨 씁니다."""
    v = [series[t0 + timedelta(minutes=shift+i)]
         for i in range(minutes) if t0 + timedelta(minutes=shift+i) in series]
    if v:
        return float(np.mean(v))
    near = [(abs(i), series[t0 + timedelta(minutes=shift+i)])
            for i in range(-tol, tol+1)
            if t0 + timedelta(minutes=shift+i) in series]
    return float(min(near)[1]) if near else None


def lag_scan(series, cam_pts, lo=-60, hi=360, step=15):
    best = None
    for sh in range(lo, hi+1, step):
        xs, ys = [], []
        for t0, cv in cam_pts:
            sv = window_mean(series, t0, shift=sh)
            if sv is not None:
                xs.append(cv)
                ys.append(sv)
        if len(xs) < 20:
            continue
        r = float(np.corrcoef(xs, ys)[0, 1])
        d = np.array(xs) - np.array(ys)
        if best is None or r > best[0]:
            best = (r, sh, float(d.std()), float(d.mean()), len(xs))
    return best


def main():
    ap = argparse.ArgumentParser(description="엽온 세 측정의 대조")
    ap.add_argument("--leaf", default="output/leaftemp/leaf_temp_all.csv")
    ap.add_argument("--contact",
                    default="calib/raw/환경센서/전남농업기술원 접촉식엽온센서_2026-09-01_2026-10-01.csv")
    ap.add_argument("--sil", default="calib/raw/환경센서/*.xlsx")
    ap.add_argument("--env",
                    default="calib/raw/환경센서/전라남도농업기술원_전체데이터.csv")
    ap.add_argument("--out", default="output/offset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    S = read_contact(os.path.join(HERE, args.contact))
    S.update(read_sil(os.path.join(HERE, args.sil)))
    env = read_env(os.path.join(HERE, args.env))
    rows = list(csv.DictReader(open(os.path.join(HERE, args.leaf),
                                    encoding="utf-8-sig")))

    print("=" * 92)
    print("엽온 세 측정의 대조 — 카메라 · 접촉식 · 비접촉식 SIL-411")
    print("=" * 92)
    for k in sorted(S):
        ts = sorted(S[k])
        print(f"  {k[0]} {k[1]:<7} {len(ts):>6}점  {ts[0]:%m-%d %H:%M} ~ "
              f"{ts[-1]:%m-%d %H:%M}")

    # 카메라: 전체 잎 평균과 층별 평균
    allpt = collections.defaultdict(list)
    laypt = collections.defaultdict(list)
    for r in rows:
        if r["분류"] != "잎":
            continue
        t0 = datetime.strptime(r["날짜"] + r["시각"], "%Y%m%d%H:%M")
        allpt[(r["카메라"], t0)].append(float(r["엽온_C"]))
        laypt[(r["카메라"], r["층_mm"], t0)].append(float(r["엽온_C"]))
    cam_pts = collections.defaultdict(list)
    for (cam, t0), v in allpt.items():
        cam_pts[cam].append((t0, float(np.mean(v))))
    for cam in cam_pts:
        cam_pts[cam].sort()
    lay_pts = collections.defaultdict(list)
    for (cam, L, t0), v in laypt.items():
        lay_pts[(cam, L)].append((t0, float(np.mean(v))))

    # ── ① 시계 ──────────────────────────────────────────────
    print("\n" + "─"*92)
    print("① 시계 점검 — 잎의 열 시정수는 수십 초. 30분 넘는 지연은 물리가 아닙니다")
    print(f"  {'구역':>5}{'센서':>9}{'최적 당김':>10}{'상관':>9}{'차이 SD':>9}"
          f"{'평균차':>9}{'n':>6}   당기지 않으면")
    lag = {}
    for k in sorted(S):
        cam = k[0]
        if cam not in cam_pts:
            continue
        b = lag_scan(S[k], cam_pts[cam])
        z = lag_scan(S[k], cam_pts[cam], 0, 0, 1)
        if b is None:
            continue
        lag[k] = b[1]
        print(f"  {cam:>5}{k[1]:>9}{b[1]:>9}분{b[0]:>+9.3f}{b[2]:>9.2f}"
              f"{b[3]:>+9.2f}{b[4]:>6}   "
              + (f"r={z[0]:+.3f} SD={z[2]:.2f}" if z else ""))
    bad = {k: v for k, v in lag.items() if abs(v) >= 30}
    if bad:
        print("  ★ 아래 채널은 로거 시계를 확인해야 합니다 — "
              + ", ".join(f"{k[0]}/{k[1]} {v}분" for k, v in bad.items()))

    # ── ② 층별 대응 ─────────────────────────────────────────
    print("\n" + "─"*92)
    print("② 어느 센서가 어느 층인가 — 이름이 아니라 «차이가 가장 작은 층» 으로")
    print(f"  {'구역':>5}{'센서':>9}" + "".join(f"{L+' mm':>12}" for L in LAYERS)
          + "   가장 가까운 층")
    pair = {}
    for k in sorted(S):
        cam = k[0]
        row, vals = [], {}
        for L in LAYERS:
            pts = lay_pts.get((cam, L))
            if not pts:
                row.append("      —")
                continue
            d = []
            for t0, cv in pts:
                sv = window_mean(S[k], t0, shift=lag.get(k, 0))
                if sv is not None:
                    d.append(cv - sv)
            if len(d) < 20:
                row.append("      —")
                continue
            vals[L] = float(np.mean(d))
            row.append(f"{vals[L]:>+11.2f}")
        if not vals:
            continue
        bestL = min(vals, key=lambda L: abs(vals[L]))
        pair[k] = bestL
        # ★ «가장 가까운 층» 을 그냥 적으면 안 됩니다. 층끼리의 차이가
        #   편향보다 훨씬 작으면 그 순위에는 의미가 없습니다.
        spread = max(vals.values()) - min(vals.values())
        bias = abs(np.mean(list(vals.values())))
        ok = spread > 0.5*bias
        print(f"  {cam:>5}{k[1]:>9}" + "".join(row) +
              (f"   {bestL} mm" if ok else
               f"   가릴 수 없음 (층간 폭 {spread:.2f} ≪ 편향 {bias:.2f})"))
    print("  → 층간 폭이 편향보다 훨씬 작으면 이 방법으로는 층을 못 가립니다.")
    print("    PPT 의 «맨 윗줄 / 3번째 열» 대응은 설치 기록으로 확정해야 합니다.")

    # ── ③ 세 측정의 삼각 대조 ───────────────────────────────
    print("\n" + "─"*92)
    print("③ 삼각 대조 — 접촉식(열전대) vs 비접촉식(적외선)")
    print("     접촉식은 방사율과 무관하고, SIL-411 은 카메라와 같은 가정을 씁니다.")
    for cam in sorted(cam_pts):
        c = S.get((cam, "접촉식"))
        if c is None:
            continue
        line = []
        for nm in ("비접촉A", "비접촉B"):
            s = S.get((cam, nm))
            if s is None:
                continue
            d = []
            for t in sorted(set(c) & set(s)):
                d.append(s[t] - c[t])
            if d:
                line.append(f"{nm} − 접촉식 = {np.mean(d):+.2f} ℃ (n={len(d)})")
        print(f"  .{cam}  " + "   ".join(line) if line else f"  .{cam}  —")
    print("  → 비접촉식도 접촉식보다 높으면, 카메라만의 문제가 아니라")
    print("    «적외선 측정 전반» 또는 «접촉식이 낮게 읽음» 쪽입니다.")

    # ── 저장 ────────────────────────────────────────────────
    out = []
    for cam in sorted(cam_pts):
        for t0, cv in cam_pts[cam]:
            rec = dict(카메라=cam, 날짜=t0.strftime("%Y%m%d"),
                       시각=t0.strftime("%H:%M"), 시=t0.hour + t0.minute/60,
                       점등=int(8 <= t0.hour < 18), 카메라_잎평균_C=round(cv, 3))
            for L in LAYERS:
                pts = dict(lay_pts.get((cam, L), []))
                if t0 in pts:
                    rec[f"카메라_{L}mm_C"] = round(pts[t0], 3)
            for k in sorted(S):
                if k[0] != cam:
                    continue
                v = window_mean(S[k], t0, shift=lag.get(k, 0))
                rec[k[1]+"_C"] = None if v is None else round(v, 3)
                rec[k[1]+"_당김분"] = lag.get(k, 0)
            e = env.get(t0)
            if e:
                rec["기온_C"], rec["습도_pct"], rec["일사"] = (
                    round(e[0], 2), round(e[1], 1), round(e[2], 1))
            out.append(rec)
    keys = sorted({k for r in out for k in r}, key=lambda s: (
        0 if s in ("카메라", "날짜", "시각", "시", "점등") else 1, s))
    p = os.path.join(args.out, "sensor_crosscheck.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(f"\n  저장  {p}  ({len(out)}행 × {len(keys)}열)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
