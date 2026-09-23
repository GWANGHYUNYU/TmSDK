#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개체별 엽온 시계열 — 한 번 칠한 마스크를 하루치 열화상에 적용한다.

    python3 scripts/leaf_temp_series.py \
        output/annot_in/annotations_2026-09-23_labeled.json \
        --day 20260918 --out output/leaftemp

어노테이션은 한 시각(08:00)에 칠했지만 **카메라가 고정**이므로, 투영한
마스크를 같은 날 다른 슬롯에 그대로 쓸 수 있습니다.

    RGB 어노테이션 → 층별 H(d) → 열화상 마스크(고정) → 슬롯마다 평균 엽온

★ 마스크가 언제까지 유효한가도 함께 잽니다. 잎은 바람·생장으로 움직이므로,
  슬롯마다 «마스크 테두리가 열화상 경계와 얼마나 겹치는지»를 내고 08:00
  대비 얼마나 떨어졌는지 표시합니다. 크게 떨어지면 그 시각은 믿지 마십시오.

FFC(셔터) 톱니는 3분 슬롯 전체를 평균하면 상쇄됩니다 (주기 182.6초,
슬롯이 그 0.986배). 그래서 슬롯 안 모든 프레임을 씁니다.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))
sys.path.insert(0, os.path.join(HERE, "scripts"))

TH_W, TH_H = 160, 120
CLS_COL = {"잎": "#2e8b57", "꽃": "#c8a200", "딸기": "#c0392b"}
LAY_COL = {420: "#1f77b4", 700: "#2ca02c", 1000: "#d62728"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    for p in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib


def masks_from(annot, package, params):
    """어노테이션 → 열화상 좌표 마스크 (개체별)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pa", os.path.join(HERE, "scripts", "project_annot.py"))
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    K, D, Kt, R, T = pa.load_params(os.path.join(HERE, params))
    A = json.load(open(annot, encoding="utf-8"))
    out = []
    for im in A["images"]:
        for o in im["objects"]:
            if o["type"] not in ("poly", "rect"):
                continue
            p = pa.project(o["points"], K, D, Kt, R, T, float(o["layer_mm"]))
            m = np.zeros((TH_H, TH_W), np.uint8)
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
            if m.sum() < 8:
                continue
            out.append(dict(src=im["image"], id=o["id"],
                            key=im["image"].replace(".png", "")+"/"+o["id"],
                            cls=o.get("class") or "-",
                            layer=int(o["layer_mm"]), mask=m.astype(bool),
                            poly=p, area=int(m.sum())))
    return out


def slot_mean(path):
    """슬롯 전체 프레임의 평균 온도맵 (FFC 톱니 상쇄) → (평균맵, 프레임수)."""
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    n = arr.shape[0]
    acc = np.zeros((TH_H, TH_W), np.float64)
    for i in range(n):
        acc += conv(np.asarray(arr[i])).astype(np.float64)[::-1, ::-1]
    return acc/n, n


def edge_fit(cels, polys):
    """마스크 테두리가 «화면 평균보다» 얼마나 강한 경계 위에 있나.

    ★ 절대 기울기로 재면 안 된다. 야간에는 잎과 공기가 평형이라 화면 전체
      기울기가 작아지는데, 그것을 «마스크가 틀어졌다»고 오해하게 된다.
      화면 전체 평균으로 나누면 대비와 무관한 값이 된다. 1.0 이면 테두리가
      아무 데나 있는 것이고, 클수록 실제 경계에 놓인 것이다.

    ★ 그런데 이 값이 1 근처라고 «마스크가 틀렸다»고 읽으면 또 틀린다. 밤에는
      잎이 공기와 평형이라 열화상에 잎 경계가 «아예 없다». 맞출 경계가 없으니
      1 근처가 나오는 것이 정상이다. 카메라가 움직였는지는 이 값이 아니라
      08:00 대비 상호상관으로 따로 확인했고 — 주간 전 슬롯이 (0,0)화소 —
      장면은 고정이다. 그래서 두 번째 반환값(화면 기울기 세기)으로 «잴 수
      있는 때인가»를 먼저 가르고, 그때만 적합도를 판정한다.
    """
    g = cv2.GaussianBlur(cels.astype(np.float32), (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = np.zeros(cels.shape, np.uint8)
    for p in polys:
        cv2.polylines(m, [np.round(p).astype(np.int32)], True, 1, 1)
    if not m.sum() or mag.mean() < 1e-9:
        return 0.0, 0.0
    return float(mag[m > 0].mean()/mag.mean()), float(mag.mean())


def main():
    ap = argparse.ArgumentParser(description="개체별 엽온 시계열")
    ap.add_argument("annot")
    ap.add_argument("--package", default="output/annotate_v2")
    ap.add_argument("--params", default="params/thermal_rgb_stereo.npz")
    ap.add_argument("--cam", default="192_168_0_151")
    ap.add_argument("--day", default="20260918")
    ap.add_argument("--raw", default="calib/raw/th/raw_output")
    ap.add_argument("--out", default="output/leaftemp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mpl = setup_mpl()
    import matplotlib.pyplot as plt

    objs = masks_from(args.annot, args.package, args.params)
    print("=" * 84)
    print(f"개체별 엽온 시계열 · {args.cam[-3:]} · {args.day}")
    print("=" * 84)
    print(f"  개체 {len(objs)}개 "
          f"(잎 {sum(1 for o in objs if o['cls']=='잎')} · "
          f"꽃 {sum(1 for o in objs if o['cls']=='꽃')} · "
          f"딸기 {sum(1 for o in objs if o['cls']=='딸기')})")

    slots = []
    for p in sorted(glob.glob(os.path.join(
            HERE, args.raw, f"{args.cam}_{args.day}_*.y16raw"))):
        if os.path.getsize(p) < 1e6:
            continue
        t = re.search(r"_(\d{6})\.y16raw", p).group(1)
        try:
            cels, n = slot_mean(p)
        except Exception:
            continue
        slots.append((f"{t[:2]}:{t[2:4]}", int(t[:2])+int(t[2:4])/60, cels, n))
    if not slots:
        raise SystemExit("쓸 수 있는 슬롯이 없습니다.")
    print(f"  슬롯 {len(slots)}개  {slots[0][0]} ~ {slots[-1][0]}")

    polys = [o["poly"] for o in objs]

    rows, fits = [], []
    print(f"\n  {'시각':>6}{'프레임':>7}{'화면평균':>9}{'잎 평균':>9}"
          f"{'잎 표준편차':>11}{'마스크 적합':>12}")
    for lab, hh, cels, n in slots:
        f, gmag = edge_fit(cels, polys)
        fits.append(f)
        lv = []
        for o in objs:
            v = float(cels[o["mask"]].mean())
            rows.append({"시각": lab, "시": round(hh, 3), "개체": o["key"],
                         "출처": o["src"], "분류": o["cls"], "층": o["layer"],
                         "면적_px": o["area"], "엽온_C": round(v, 3)})
            if o["cls"] == "잎":
                lv.append(v)
        lv = np.array(lv)
        print(f"  {lab:>6}{n:>7}{cels.mean():>8.2f}℃{lv.mean():>8.2f}℃"
              f"{lv.std():>10.3f}℃{f:>11.2f}"
              + ("  잎 경계 없음(소등) — 판정 보류" if gmag < 0.60
                 else ("  ⚠ 마스크 확인" if f < 1.03 else "  마스크 일치")))

    with open(os.path.join(args.out, f"leaf_temp_{args.day}.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── 그림 ─────────────────────────────────────────────────────
    hs = [s[1] for s in slots]
    labs = [s[0] for s in slots]
    byid = {}
    for r in rows:
        byid.setdefault(r["개체"], []).append(r["엽온_C"])
    meta = {o["key"]: o for o in objs}

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"개체별 엽온 · {args.day[:4]}-{args.day[4:6]}-{args.day[6:]}"
                 f" · {args.cam[-3:]} · 개체 {len(objs)}개",
                 fontsize=13, fontweight="bold")

    a = ax[0, 0]
    for oid, v in byid.items():
        a.plot(hs, v, lw=0.8, alpha=.5,
               color=CLS_COL.get(meta[oid]["cls"], "#888"))
    for c, col in CLS_COL.items():
        v = np.array([[byid[o["key"]][i] for o in objs if o["cls"] == c]
                      for i in range(len(hs))])
        if v.size:
            a.plot(hs, v.mean(1), lw=2.6, color=col, label=f"{c} 평균")
    a.axvspan(8.017, 18.017, color="#ffd54f", alpha=.13, zorder=0)
    a.text(13, a.get_ylim()[0], " 점등 08:01~18:01", fontsize=9, color="#8a6d00")
    a.set_xlabel("시각 (시)"); a.set_ylabel("온도 (℃)")
    a.set_title("분류별 — 가는 선이 개체 하나")
    a.legend(fontsize=9); a.grid(alpha=.25)

    a = ax[0, 1]
    for L, col in LAY_COL.items():
        ids = [o["key"] for o in objs if o["layer"] == L and o["cls"] == "잎"]
        if not ids:
            continue
        v = np.array([[byid[i][k] for i in ids] for k in range(len(hs))])
        a.plot(hs, v.mean(1), lw=2.4, color=col, label=f"{L} mm (n={len(ids)})")
        a.fill_between(hs, v.mean(1)-v.std(1), v.mean(1)+v.std(1),
                       color=col, alpha=.15)
    a.axvspan(8.017, 18.017, color="#ffd54f", alpha=.13, zorder=0)
    a.set_xlabel("시각 (시)"); a.set_ylabel("온도 (℃)")
    a.set_title("캐노피 층별 (잎만) — 띠는 ±표준편차")
    a.legend(fontsize=9); a.grid(alpha=.25)

    a = ax[1, 0]
    v = np.array([[byid[o["key"]][k] for o in objs if o["cls"] == "잎"]
                  for k in range(len(hs))])
    a.plot(hs, v.std(1), lw=2.4, color="#6a3d9a", marker="o", ms=4)
    a.set_xlabel("시각 (시)"); a.set_ylabel("개체 간 표준편차 (℃)")
    a.set_title("★ 개체 간 엽온 편차 — 이 과제가 겨냥한 값")
    a.axvspan(8.017, 18.017, color="#ffd54f", alpha=.13, zorder=0)
    a.grid(alpha=.25)
    a2 = a.twinx()
    a2.plot(hs, fits, lw=1.2, ls="--", color="#999")
    a2.set_ylabel("마스크 적합도 (화면평균 대비 배)", color="#777", fontsize=9)
    a2.axhline(1.0, color="#c00", lw=.8, ls=":")

    a = ax[1, 1]
    k = len(hs)-1
    cels = slots[k][2]
    a.imshow(cels, cmap="inferno")
    for o in objs:
        q = o["poly"]
        a.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]],
               lw=.9, color=CLS_COL.get(o["cls"], "#fff"))
    a.set_title(f"{labs[k]} 열화상 + 마스크"); a.axis("off")

    fig.tight_layout()
    p1 = os.path.join(args.out, f"leaf_temp_{args.day}.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    print(f"\n  저장  {p1}")
    print(f"        {args.out}/leaf_temp_{args.day}.csv  ({len(rows)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
