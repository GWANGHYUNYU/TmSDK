#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""층별·분류별 마스크 그림 — 두 카메라 모두 **실측 어노테이션**으로.

    python3 scripts/make_mask_figs.py output/annot_in/annotations_final.json \
        --out output/maskfigs

카메라마다 근거가 다릅니다. 그림 제목에 그대로 적습니다.

  `.151`  RGB 에 찍은 어노테이션을 층별 H(d) 로 **투영**한 마스크.
          RGB 낱장 · 열화상 낱장 · 둘을 나란히 붙인 대조 시트까지 냅니다.
  `.152`  RGB 스테레오가 없어 **열화상에 직접** 찍은 마스크. 투영이 없으니
          대조 시트도 없습니다. 층은 사람이 눈으로 넣은 값이라 검증되지
          않았습니다 — 그림에 그렇게 적습니다.

★ 예전 판은 `.152` 를 «주위보다 차가운 덩어리» 로 추정했습니다. 이제
  실측이 있으므로 추정은 걷어냈습니다. 둘을 섞어 두면 어느 쪽 근거로 나온
  그림인지 나중에 구분할 수 없습니다.

★ 그릴 날짜는 손으로 고르지 않고 추출의 판정 결과(`leaf_temp_days.csv`)를
  따릅니다. 쓸 수 없다고 판정한 날의 그림을 내놓으면, 그림만 보고 그 날
  데이터가 쓸 만하다고 오해하게 됩니다.
"""
from __future__ import annotations

import argparse
import csv as _csv
import glob
import importlib.util
import json
import os
import re
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


LTS = _load("lts", "leaf_temp_series.py")
SG = _load("sg", "make_seg_guide.py")

TH_W, TH_H = 160, 120
LAYERS = (420, 700, 1000)
LAY_COL = {420: (255, 120, 40), 700: (60, 200, 60), 1000: (60, 60, 240)}
CLS_COL = {"잎": (90, 220, 90), "꽃": (255, 255, 255),
           "딸기": (60, 60, 240), "엽온센서": (0, 200, 255)}
CLS_EN = {"잎": "LEAF", "꽃": "FLOWER", "딸기": "FRUIT",
          "엽온센서": "LEAF-TEMP SENSOR"}


def draw_masks(base, items, fill=0.30):
    out = base.copy()
    lay = np.zeros_like(out)
    hit = np.zeros(out.shape[:2], np.uint8)
    for m, col in items:
        big = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        lay[big > 0] = col
        hit[big > 0] = 1
    out = np.where(hit[..., None] > 0,
                   (out*(1-fill) + lay*fill).astype(np.uint8), out)
    for m, col in items:
        big = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.drawContours(out, cs, -1, col, 2, cv2.LINE_AA)
    return out


# ★ 대응 방향 — 이미 한 번 헷갈린 지점입니다.
#   H(d) 는 «RGB 원본 → 180° 돌린 열화상» 으로 맵니다. 사람이 보는 방향은
#   RGB 를 뒤집은 쪽이라, 한쪽만 돌려 붙이면 위아래가 반대로 보입니다.
#   나란히 그림에서는 **양쪽을 같이** 돌립니다. 데이터 규약은 그대로입니다.
def flip_th(x):
    return x[::-1, ::-1]


def side_by_side(left, right, gap=10):
    h = max(left.shape[0], right.shape[0])

    def pad(im):
        s = h/im.shape[0]
        return cv2.resize(im, (int(im.shape[1]*s), h))
    return np.hstack([pad(left), np.full((h, gap, 3), 40, np.uint8),
                      pad(right)])


def rgb_view(package, name):
    im = cv2.imread(os.path.join(HERE, package, "frames", name))
    return None if im is None else cv2.rotate(im, cv2.ROTATE_180)


def rgb_poly(o, W=1920, H=1080):
    return np.array([[W-1-x, H-1-y] for x, y in o["points"]], np.int32)


def fill_rgb(shape, q):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [q], 1)
    return m


def groups(objs, key, order):
    return [(k, [o for o in objs if o.get(key) == k]) for k in order]


def main():
    ap = argparse.ArgumentParser(description="층별·분류별 마스크 그림")
    ap.add_argument("annot", nargs="?",
                    default="output/annot_in/annotations_final.json")
    ap.add_argument("--package", default="output/annotate_v2")
    ap.add_argument("--params", default="params/thermal_rgb_stereo.npz")
    ap.add_argument("--days", default="",
                    help="«151=20260918,...» 비우면 판정 CSV 를 따릅니다")
    ap.add_argument("--days-csv", default="output/leaftemp/leaf_temp_days.csv")
    ap.add_argument("--out", default="output/maskfigs")
    args = ap.parse_args()

    A = json.load(open(args.annot, encoding="utf-8"))
    print("=" * 84)
    print("층별·분류별 마스크 그림 — 두 카메라 실측")
    print("=" * 84)

    use_days = {}
    if args.days:
        for kv in args.days.split(","):
            c, d = kv.split("=")
            use_days.setdefault(c, []).append(d)
    elif os.path.exists(args.days_csv):
        for r in _csv.DictReader(open(args.days_csv, encoding="utf-8-sig")):
            if r["마스크판정"] in ("확인", "주의", "보정"):
                use_days.setdefault(r["카메라"], []).append(r["날짜"])
    for c in sorted(use_days):
        print(f"  .{c} 그릴 날짜 {len(use_days[c])}개: "
              f"{' '.join(d[4:] for d in use_days[c])}")

    for cam in sorted({im["camera"][-3:] for im in A["images"]}):
        objs = LTS.masks_from(args.annot, args.package, args.params,
                              camera=cam)
        if not objs:
            continue
        space = objs[0]["space"]
        d = os.path.join(args.out,
                         f"{cam}_{'투영' if space == 'rgb' else '직접'}")
        os.makedirs(d, exist_ok=True)
        src = ("RGB annotation projected with H(d)" if space == "rgb"
               else "drawn directly on thermal (no RGB stereo)")
        lay_note = ("projected with that layer's H(d)" if space == "rgb"
                    else "layer assigned by eye - not verified for this camera")
        cls_note = ("classes come from RGB - thermal cannot separate them"
                    if space == "rgb"
                    else "drawn on thermal; only LEAF is separable here")
        print(f"\n── .{cam}  개체 {len(objs)}  근거: {src} " + "─"*18)

        for day in use_days.get(cam, []):
            p, cels = SG.find_slot(cam, day, range(10, 17))
            if p is None:
                print(f"  {day} 점등 슬롯 없음")
                continue
            g = re.search(r"_(\d{2})(\d{2})\d{2}\.y16raw$", p)
            tm = f"{g.group(1)}:{g.group(2)}"
            base = SG.th_render(cels, "dev")
            tag = f".{cam}  {day[4:6]}-{day[6:]} {tm}"
            SG.imwrite(os.path.join(d, f"th{cam}_{day}_00_원본.png"),
                       SG.label(base, f"{tag}  -  no mask",
                                f"deviation from frame mean "
                                f"{cels.mean():.2f}C   |   {src}"))
            SG.imwrite(
                os.path.join(d, f"th{cam}_{day}_01_전체_{len(objs)}개.png"),
                SG.label(draw_masks(base, [
                    (o["mask"], CLS_COL.get(o["cls"], (200, 160, 60)))
                    for o in objs]), f"{tag}  -  all {len(objs)}", src))
            nl = nc = 0
            for L, sel in groups(objs, "layer", LAYERS):
                if not sel:
                    continue
                nl += 1
                SG.imwrite(
                    os.path.join(d, f"th{cam}_{day}_02_층{L}_{len(sel)}개.png"),
                    SG.label(draw_masks(base, [(o["mask"], LAY_COL[L])
                                               for o in sel]),
                             f"{tag}  -  layer {L} mm  ({len(sel)})", lay_note))
            for c, sel in groups(objs, "cls", CLS_EN):
                if not sel:
                    continue
                nc += 1
                SG.imwrite(
                    os.path.join(d,
                                 f"th{cam}_{day}_03_분류_{c}_{len(sel)}개.png"),
                    SG.label(draw_masks(base, [(o["mask"], CLS_COL[c])
                                               for o in sel]),
                             f"{tag}  -  {CLS_EN[c]}  ({len(sel)})", cls_note))
            print(f"  {day} {tm}  원본+전체+층{nl}+분류{nc}")

        if space != "rgb":
            continue

        # ── RGB 낱장 + 나란히 대조 (투영 카메라만) ─────────────
        raw = {im["image"]: im for im in A["images"] if im["space"] == "rgb"}
        pr, pcels = SG.find_slot(cam, "20260918", [8])
        pbase = SG.th_render(flip_th(pcels), "dev") if pcels is not None else None
        for nm, im in raw.items():
            view = rgb_view(args.package, nm)
            if view is None:
                continue
            stem = nm[:-4]
            mine = im["objects"]
            SG.imwrite(os.path.join(d, f"rgb_{stem}_00_원본.png"), view)
            SG.imwrite(
                os.path.join(d, f"rgb_{stem}_01_전체_{len(mine)}개.png"),
                SG.label(draw_masks(view, [
                    (fill_rgb(view.shape, rgb_poly(o)),
                     CLS_COL.get(o.get("class"), (200, 160, 60)))
                    for o in mine]), f"RGB  {stem}  -  all {len(mine)}",
                    "green=leaf  white=flower  red=fruit"))
            for L, sel in groups(mine, "layer_mm", LAYERS):
                if not sel:
                    continue
                SG.imwrite(
                    os.path.join(d, f"rgb_{stem}_02_층{L}_{len(sel)}개.png"),
                    SG.label(draw_masks(view, [
                        (fill_rgb(view.shape, rgb_poly(o)), LAY_COL[L])
                        for o in sel]),
                        f"RGB  {stem}  -  layer {L} mm  ({len(sel)})",
                        "far layers look hazy - that is the depth cue"))
                if pbase is None:
                    continue
                ts = [o for o in objs if o["layer"] == L and o["src"] == nm]
                SG.imwrite(
                    os.path.join(d, f"pair_{stem}_층{L}_{len(sel)}개.png"),
                    SG.label(side_by_side(
                        draw_masks(view, [(fill_rgb(view.shape, rgb_poly(o)),
                                           LAY_COL[L]) for o in sel]),
                        draw_masks(pbase, [(flip_th(o["mask"]), LAY_COL[L])
                                           for o in ts])),
                        f"RGB (left) vs .{cam} thermal 08:00 (right)"
                        f"   -   layer {L} mm  ({len(sel)})",
                        "both sides turned to viewing orientation"))
            for c, sel in groups(mine, "class", CLS_EN):
                if not sel:
                    continue
                SG.imwrite(
                    os.path.join(d,
                                 f"rgb_{stem}_03_분류_{c}_{len(sel)}개.png"),
                    SG.label(draw_masks(view, [
                        (fill_rgb(view.shape, rgb_poly(o)), CLS_COL[c])
                        for o in sel]),
                        f"RGB  {stem}  -  {CLS_EN[c]}  ({len(sel)})", ""))
            print(f"  {stem}  RGB 낱장 + 나란히")

    n = len(glob.glob(os.path.join(args.out, "*", "*.png")))
    print(f"\n  저장  {args.out}/  ({n}장)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
