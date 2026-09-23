#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""어노테이션을 하나의 최종 파일로 합칩니다.

    python3 scripts/merge_annotations.py output/annot_in/*.json \
        --out output/annot_in/annotations_final.json

도구가 내보내는 파일에는 **좌표계가 다른 프레임이 섞여** 있습니다.

  · RGB (.151 짝)  1920×1080, 원본 파일 좌표 → H(층) 로 열화상에 투영
  · 열화상 (.152)  960×720 = 160×120 ×6, 이미 열화상 좌표 (투영 불필요)

그런데 도구는 최상위에 **RGB 규약 한 줄만** 적습니다. 그대로 두면 다음
사람이 `.152` 좌표에도 투영을 걸게 됩니다. 그래서 프레임마다 좌표계·카메라·
배율을 명시해 붙이고, 열화상 좌표를 따로 계산해 둡니다.

★ 품질도 같이 적습니다. 열화상 어노테이션은 «그 자리에 실제 경계가 있는가»
  를 잴 수 있습니다(적합도). 1 미만이면 경계 위에 없다는 뜻인데, **틀렸다는
  뜻은 아닙니다** — 그 구역의 열 구조가 약해서 «확인할 수 없다» 는 뜻입니다.
  지우지 않고 수치로 남겨, 쓰는 쪽이 판단하게 합니다.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from datetime import datetime

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


LTS = _load("lts", "leaf_temp_series.py")
SG = _load("sg", "make_seg_guide.py")

TH_W, TH_H = 160, 120
RGB_SIZE = [1920, 1080]
# 열화상 프레임 이름은 make_th_annotate.py 가 <날짜>_<시각> 으로 냅니다.
TH_NAME = re.compile(r"^(\d{8})_(\d{2})(\d{2})$")


def poly_area(p):
    return float(0.5*abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) -
                         np.dot(p[:, 1], np.roll(p[:, 0], 1))))


def main():
    ap = argparse.ArgumentParser(description="어노테이션 통합")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--th-cam", default="152",
                    help="열화상 프레임이 속한 카메라")
    ap.add_argument("--rgb-cam", default="151",
                    help="RGB 와 스테레오가 잡힌 카메라")
    ap.add_argument("--upscale", type=int, default=6)
    ap.add_argument("--out", default="output/annot_in/annotations_final.json")
    args = ap.parse_args()

    files = []
    for g in args.inputs:
        files.extend(sorted(glob.glob(g)))
    files = [f for f in files if os.path.basename(f) != os.path.basename(args.out)]
    if not files:
        raise SystemExit("입력 파일이 없습니다.")

    print("=" * 84)
    print("어노테이션 통합")
    print("=" * 84)

    # ★ 같은 프레임이 여러 파일에 있으면 «가장 최근 것» 을 씁니다. 도구는
    #   내보낼 때마다 전체를 다시 쓰므로, 옛 파일을 섞으면 되돌아갑니다.
    picked, origin = {}, {}
    for f in files:
        try:
            A = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  {os.path.basename(f)} 읽기 실패 — 건너뜁니다 ({e})")
            continue
        ts = A.get("saved_at") or datetime.fromtimestamp(
            os.path.getmtime(f)).isoformat()
        n = 0
        for im in A.get("images", []):
            k = im["image"]
            if k not in picked or ts >= origin[k][1]:
                picked[k] = im
                origin[k] = (os.path.basename(f), ts)
                n += 1
        print(f"  {os.path.basename(f):<44} 프레임 {len(A.get('images', []))}"
              f"  채택 {n}  ({ts[:19]})")

    out_images, stats = [], dict(rgb=0, th=0, obj=0, noclass=0, lowfit=0)
    print()
    for name in sorted(picked):
        im = picked[name]
        objs = [o for o in im["objects"] if o["type"] in ("poly", "rect")]
        m = TH_NAME.match(os.path.splitext(name)[0])
        if m:
            # ── 열화상 프레임 ────────────────────────────────────
            day, hh, mm = m.groups()
            up = args.upscale
            p, cels = SG.find_slot(args.th_cam, day, [int(hh)])
            rec = dict(image=name, space="thermal",
                       camera=f"192.168.0.{args.th_cam}",
                       size=[TH_W*up, TH_H*up], upscale=up,
                       thermal_size=[TH_W, TH_H],
                       coord=("PNG 좌표 ÷ upscale = 열화상 화소. "
                              "좌표계는 프로젝트 규약의 180° 회전 상태."),
                       projection="불필요 — 이미 열화상 좌표입니다",
                       slot=os.path.basename(p) if p else None,
                       source=origin[name][0])
            oo = []
            for o in objs:
                q = np.array(o["points"], float)/up
                mk = np.zeros((TH_H, TH_W), np.uint8)
                cv2.fillPoly(mk, [np.round(q).astype(np.int32)], 1)
                d = dict(id=o["id"], type=o["type"],
                         **{"class": o.get("class")},
                         layer_mm=o.get("layer_mm"),
                         points=o["points"],
                         points_thermal=[[round(x, 3), round(y, 3)]
                                         for x, y in q],
                         area_px=int(mk.sum()))
                if cels is not None and mk.sum() >= 3:
                    d["edge_fit"] = round(LTS.edge_fit(cels, [q])[0], 3)
                    d["mean_C"] = round(float(cels[mk > 0].mean()), 3)
                    if d["edge_fit"] < 1.0:
                        stats["lowfit"] += 1
                if not o.get("class"):
                    stats["noclass"] += 1
                oo.append(d)
            rec["objects"] = oo
            stats["th"] += 1
            fits = [o["edge_fit"] for o in oo if "edge_fit" in o]
            print(f"  {name:<22} 열화상 .{args.th_cam}  개체 {len(oo):>2}  "
                  f"적합도 중앙 {np.median(fits) if fits else float('nan'):.2f}  "
                  f"1.0 미만 {sum(1 for f in fits if f < 1.0)}개")
        else:
            # ── RGB 프레임 ──────────────────────────────────────
            rec = dict(image=name, space="rgb",
                       camera=f"192.168.0.{args.rgb_cam}",
                       size=RGB_SIZE, upscale=1,
                       coord=("원본 RGB 파일 좌표 (뒤집지 않은 상태). "
                              "화면에는 180° 뒤집어 보여 줍니다."),
                       projection=("undistortPoints(K_rgb, dist) → "
                                   "perspectiveTransform(H[layer]) → "
                                   "180° 회전한 열화상 좌표"),
                       source=origin[name][0])
            oo = []
            for o in objs:
                if not o.get("class"):
                    stats["noclass"] += 1
                oo.append(dict(id=o["id"], type=o["type"],
                               **{"class": o.get("class")},
                               layer_mm=o.get("layer_mm"),
                               points=o["points"],
                               area_px=int(round(
                                   poly_area(np.array(o["points"], float))))))
            rec["objects"] = oo
            stats["rgb"] += 1
            print(f"  {name:<22} RGB .{args.rgb_cam}       개체 {len(oo):>2}")
        stats["obj"] += len(rec["objects"])
        out_images.append(rec)

    classes = sorted({o.get("class") for im in out_images
                      for o in im["objects"] if o.get("class")})
    doc = dict(
        version=2,
        created=datetime.now().isoformat(timespec="seconds"),
        note=("좌표계가 프레임마다 다릅니다. 반드시 각 image 의 `space` 와 "
              "`coord` 를 보고 쓰십시오. 최상위에 규약을 하나만 적으면 "
              "열화상 좌표에도 투영을 걸게 됩니다."),
        classes=classes,
        layers_mm=[420, 700, 1000],
        quality_note=("열화상 개체의 `edge_fit` 은 «그 자리에 실제 열 경계가 "
                      "있는가» 입니다. 1 미만은 틀렸다는 뜻이 아니라 그 구역의 "
                      "열 구조가 약해 확인할 수 없다는 뜻입니다."),
        sources=[os.path.basename(f) for f in files],
        images=out_images,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(doc, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print()
    print(f"  프레임 {len(out_images)}개 (RGB {stats['rgb']} · 열화상 "
          f"{stats['th']})  ·  개체 {stats['obj']}개")
    print(f"  분류: {' '.join(classes)}")
    if stats["noclass"]:
        print(f"  ⚠ 분류가 비어 있는 개체 {stats['noclass']}개 — "
              "쓰기 전에 채우거나 빼십시오")
    if stats["lowfit"]:
        print(f"  ⚠ 적합도 1.0 미만인 열화상 개체 {stats['lowfit']}개 — "
              "지우지 않고 수치로 남겼습니다")
    print(f"\n  저장  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
