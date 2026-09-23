#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""열화상 어노테이션 검수 그림 — 개체마다 «확인 / 보류 / 미확인».

    python3 scripts/check_th_annot.py output/annot_in/annotations_final.json \
        --out output/th152_check

통합 파일(version 2)의 `fit_grade` 를 그대로 씁니다. 등급의 뜻은 하나입니다 —
**«그 자리에 실제 열 경계가 있는가»**. 틀렸다/맞았다가 아닙니다.

  확인   경계 위에 있다
  보류   그 구역 대비가 약해 확인도 반증도 안 된다
  미확인 경계라 부를 것이 없는 자리다

★ 1.0 같은 딱 떨어지는 값에서 자르면 안 됩니다. 0.97 과 1.00 을 «실패/성공»
  으로 가르는 것은 의미가 없습니다. 그래서 띠로 나누고, 색도 3단계로 씁니다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, "scripts", fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


SG = _load("sg", "make_seg_guide.py")

CLS_COL = {"잎": (90, 220, 90), "꽃": (255, 255, 255),
           "딸기": (60, 60, 240), "엽온센서": (0, 200, 255)}
GRADE_COL = {"확인": (90, 230, 90), "보류": (60, 200, 255),
             "미확인": (60, 60, 255)}
LAY_DASH = {420: 0, 700: 1, 1000: 2}


def main():
    ap = argparse.ArgumentParser(description="열화상 어노테이션 검수 그림")
    ap.add_argument("annot", default="output/annot_in/annotations_final.json",
                    nargs="?")
    ap.add_argument("--by", choices=["grade", "class", "layer"],
                    default="grade", help="테두리 색 기준")
    ap.add_argument("--out", default="output/th152_check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A = json.load(open(args.annot, encoding="utf-8"))
    print("=" * 84)
    print(f"열화상 어노테이션 검수 · 색 기준 «{args.by}»")
    print("=" * 84)

    tot = {"확인": 0, "보류": 0, "미확인": 0}
    for im in A["images"]:
        if im.get("space") != "thermal":
            continue
        day, hhmm = os.path.splitext(im["image"])[0].split("_")
        p, cels = SG.find_slot(im["camera"].split(".")[-1], day,
                               [int(hhmm[:2])])
        if p is None:
            print(f"  {im['image']}  슬롯을 못 찾았습니다")
            continue
        base = SG.th_render(cels, "dev")
        up = im.get("upscale", 6)
        cnt = {"확인": 0, "보류": 0, "미확인": 0}
        for o in im["objects"]:
            q = np.array(o.get("points_thermal")
                         or np.array(o["points"], float)/up, float)
            g = o.get("fit_grade", "보류")
            cnt[g] = cnt.get(g, 0) + 1
            col = (GRADE_COL[g] if args.by == "grade" else
                   CLS_COL.get(o.get("class"), (200, 160, 60)))
            pts = (q*up).astype(np.int32)
            cv2.polylines(base, [pts], True, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(base, [pts], True, col, 2, cv2.LINE_AA)
            c = pts.mean(0).astype(int)
            txt = (f"{o['id'][3:]} {o.get('edge_fit', 0):.2f}"
                   if args.by == "grade"
                   else f"{o['id'][3:]} {o.get('class') or '-'}"
                   f" {o.get('layer_mm')}")
            for cc, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(base, txt, (c[0]-26, c[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, cc, th, cv2.LINE_AA)
        for k in tot:
            tot[k] += cnt[k]
        sub = ("green=verified  orange=inconclusive  red=no boundary there"
               if args.by == "grade"
               else "green=leaf  white=flower  red=fruit  cyan=leaf-temp sensor")
        SG.imwrite(
            os.path.join(args.out, f"{im['image'][:-4]}_{args.by}.png"),
            SG.label(base,
                     f"{im['camera']}  {day[4:6]}-{day[6:]} "
                     f"{hhmm[:2]}:{hhmm[2:]}  -  {len(im['objects'])} objects",
                     sub))
        print(f"  {im['image']:<20} 확인 {cnt['확인']:>2} · "
              f"보류 {cnt['보류']:>2} · 미확인 {cnt['미확인']:>2}")

    print(f"\n  합계  확인 {tot['확인']} · 보류 {tot['보류']} · "
          f"미확인 {tot['미확인']}")
    print(f"  저장  {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
