#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""층별·분류별 마스크 그림 — 육안 확인용.

    python3 scripts/make_mask_figs.py \
        output/annot_in/annotations_2026-09-23_labeled.json --out output/maskfigs

두 부분으로 나뉩니다.

  A. `.151` — **실측 마스크**
     RGB 에 직접 찍은 어노테이션과, 그것을 층별 H(d) 로 투영한 열화상 마스크.
     층(420/700/1000)별·분류(잎/꽃/딸기)별로 따로 냅니다.

  B. `.152` — **추정 마스크**
     매칭되는 RGB 가 없어 투영이 불가능합니다. 열화상만으로 «주위보다 차가운
     덩어리 = 잎» 을 잡습니다.

★ B 는 A 와 근거의 급이 다릅니다. A 는 사람이 RGB 에서 보고 찍은 것이고,
  B 는 온도만 보고 기계가 추측한 것입니다. 그림 제목에 그렇게 적습니다.
  분류(잎/꽃/딸기)는 B 에서 **아예 시도하지 않습니다** — 열화상으로는
  꽃 66 % · 딸기 44 % 로 갈리지 않습니다 (docs/segmentation_guide.md §0).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
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
UP = 6
LAYERS = (420, 700, 1000)
LAY_COL = {420: (255, 120, 40), 700: (60, 200, 60), 1000: (60, 60, 240)}
CLS_COL = {"잎": (90, 220, 90), "꽃": (255, 255, 255), "딸기": (60, 60, 240)}
CLS_EN = {"잎": "LEAF", "꽃": "FLOWER", "딸기": "FRUIT"}


def th_base(cels):
    """열화상 배경 — 화면평균 대비 편차. 잎이 보이는 유일한 표현입니다."""
    return SG.th_render(cels, "dev")


def draw_masks(base, items, up=UP, fill=0.30):
    """마스크를 테두리 + 옅은 채움으로 얹습니다."""
    out = base.copy()
    lay = np.zeros_like(out)
    any_px = np.zeros(out.shape[:2], np.uint8)
    for m, col in items:
        big = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        lay[big > 0] = col
        any_px[big > 0] = 1
    out = np.where(any_px[..., None] > 0,
                   (out*(1-fill) + lay*fill).astype(np.uint8), out)
    for m, col in items:
        big = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # ★ 색선을 1 px 로 그리면 검은 테두리에 묻혀 층·분류가 구분이 안 됩니다.
        cv2.drawContours(out, cs, -1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.drawContours(out, cs, -1, col, 2, cv2.LINE_AA)
    return out


# ★ 대응 방향 주의 — 이 프로젝트에서 이미 한 번 헷갈린 지점입니다.
#
#   H(d) 는 «원본 RGB 파일 좌표 → 180° 돌린 열화상 좌표» 로 맵니다.
#   즉 대응하는 짝은  (RGB 원본) ↔ (열화상 rotated) 입니다.
#   그런데 사람이 보는 방향은 RGB 를 뒤집은 쪽이라, 낱장 그림은
#   (RGB 반전) 과 (열화상 rotated) 를 쓰고 있어서 «위아래가 반대» 로 보입니다.
#
#   실제로 420 mm 층은 RGB 원본에서 아래(y중앙 957/1080), 열화상 rotated
#   에서도 아래(y중앙 90/120) 입니다 — 대응은 맞습니다.
#
#   그래서 나란히 그림에서는 **양쪽을 같이** 돌려 사람이 보는 방향으로
#   맞춥니다. 데이터 규약을 바꾸는 것이 아니라 이 그림에서만 하는 표시입니다.
#   방향이 맞는지는 추측하지 말고 마스크 적합도로 확인했습니다:
#     08:00 현재 1.218 / 상하 1.038 / 좌우 1.106 / 양쪽 1.069  → 현재가 최고
#     10:30·11:30·16:00 에서도 현재가 최고이고 상하는 1.0 미만(무작위 이하)
def flip_th(x):
    """rotated 열화상 좌표계를 사람이 보는 방향으로 되돌립니다."""
    return x[::-1, ::-1]


def side_by_side(left, right, gap=10):
    """RGB(왼쪽)와 열화상(오른쪽)을 같은 높이로 붙입니다.

    ★ 육안 대조에는 이 한 장이 낱장 두 장보다 낫습니다. 같은 잎을 두
      영상에서 번갈아 찾을 필요가 없어집니다.
    """
    h = max(left.shape[0], right.shape[0])
    def pad(im):
        s = h/im.shape[0]
        im = cv2.resize(im, (int(im.shape[1]*s), h))
        return im
    a, b = pad(left), pad(right)
    sep = np.full((h, gap, 3), 40, np.uint8)
    return np.hstack([a, sep, b])


def rgb_view(package, name):
    p = os.path.join(HERE, package, "frames", name)
    im = cv2.imread(p)
    if im is None:
        return None
    return cv2.rotate(im, cv2.ROTATE_180)      # RGB 가 뒤집힌 쪽입니다


def rgb_poly(o, W=1920, H=1080):
    """원본 파일 좌표 → 반전 보기 좌표."""
    return np.array([[W-1-x, H-1-y] for x, y in o["points"]], np.int32)


def fill_rgb(shape, polys):
    m = np.zeros(shape[:2], np.uint8)
    for q in polys:
        cv2.fillPoly(m, [q], 1)
    return m


def pick_slot(cam, day, hours):
    return SG.find_slot(cam, day, hours)


# ─────────────────────────────────────────────────────────────
# B. `.152` 잎 추정 — 주위보다 차가운 덩어리
# ─────────────────────────────────────────────────────────────
def estimate_leaves(cels, k=0.25, amin=12, amax=1200):
    """국소 배경보다 k ℃ 이상 차가운 덩어리를 잎 후보로 잡습니다.

    ★ 전역 평균으로 자르면 안 됩니다. 화면에 공조 얼룩이 있어 한쪽 절반이
      통째로 «차갑게» 나옵니다. 국소 배경(큰 블러) 대비로 잡아야 잎 덩어리만
      남습니다 — RGB 에서 꽃을 잡을 때와 같은 이유입니다.
    """
    x = cels.astype(np.float32)
    bg = cv2.blur(x, (41, 41))
    m = ((bg - x) > k).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    keep = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if amin <= a <= amax:
            keep.append((lab == i, a, cen[i]))
    return keep


def main():
    ap = argparse.ArgumentParser(description="층별·분류별 마스크 그림")
    ap.add_argument("annot")
    ap.add_argument("--package", default="output/annotate_v2")
    ap.add_argument("--params", default="params/thermal_rgb_stereo.npz")
    ap.add_argument("--days-151", default="20260917,20260918,20260919,20260920")
    ap.add_argument("--days-152", default="20260919,20260921,20260923")
    ap.add_argument("--out", default="output/maskfigs")
    args = ap.parse_args()

    d151 = os.path.join(args.out, "151_실측")
    d152 = os.path.join(args.out, "152_추정")
    for d in (d151, d152):
        os.makedirs(d, exist_ok=True)

    objs = LTS.masks_from(args.annot, args.package, args.params)
    import json
    A = json.load(open(args.annot, encoding="utf-8"))
    raw_obj = [(im["image"], o) for im in A["images"] for o in im["objects"]
               if o["type"] in ("poly", "rect")]

    print("=" * 84)
    print("층별·분류별 마스크 그림")
    print("=" * 84)
    print(f"  개체 {len(objs)}개  ·  실측 {d151}  ·  추정 {d152}")

    # ── A-1. RGB (어노테이션을 찍은 그 프레임) ──────────────────
    srcs = sorted({o["src"] for o in objs})
    print(f"\n── A. .151 실측 — RGB 프레임 {len(srcs)}장 " + "─"*34)
    for src in srcs:
        view = rgb_view(args.package, src)
        if view is None:
            print(f"  {src} 프레임 파일 없음 — 건너뜁니다")
            continue
        stem = src.replace(".png", "")
        SG.imwrite(os.path.join(d151, f"rgb_{stem}_00_원본.png"), view)

        mine = [(nm, o) for nm, o in raw_obj if nm == src]
        allm = [(fill_rgb(view.shape, [rgb_poly(o)]),
                 CLS_COL.get(o.get("class"), (200, 160, 60))) for _, o in mine]
        SG.imwrite(os.path.join(d151, f"rgb_{stem}_01_전체_{len(mine)}개.png"),
                   SG.label(draw_masks(view, allm, up=1),
                            f"RGB  {stem}  -  all {len(mine)} objects",
                            "green=leaf  white=flower  red=fruit"))

        for L in LAYERS:
            sel = [o for _, o in mine if int(o["layer_mm"]) == L]
            if not sel:
                continue
            items = [(fill_rgb(view.shape, [rgb_poly(o)]), LAY_COL[L])
                     for o in sel]
            SG.imwrite(os.path.join(d151, f"rgb_{stem}_02_층{L}_{len(sel)}개.png"),
                       SG.label(draw_masks(view, items, up=1),
                                f"RGB  {stem}  -  layer {L} mm  ({len(sel)})",
                                "far layers look hazy - that is the depth cue"))
        for cls in ("잎", "꽃", "딸기"):
            sel = [o for _, o in mine if o.get("class") == cls]
            if not sel:
                continue
            items = [(fill_rgb(view.shape, [rgb_poly(o)]), CLS_COL[cls])
                     for o in sel]
            SG.imwrite(
                os.path.join(d151, f"rgb_{stem}_03_분류_{cls}_{len(sel)}개.png"),
                SG.label(draw_masks(view, items, up=1),
                         f"RGB  {stem}  -  {CLS_EN[cls]}  ({len(sel)})", ""))
        print(f"  {stem}  전체 {len(mine)}  "
              + "  ".join(f"{L}mm {sum(1 for _,o in mine if int(o['layer_mm'])==L)}"
                          for L in LAYERS)
              + "  |  " + "  ".join(
                  f"{c} {sum(1 for _,o in mine if o.get('class')==c)}"
                  for c in ("잎", "꽃", "딸기")))

    # ── A-1b. RGB ↔ 열화상 나란히 (같은 순간) ───────────────────
    # ★ 기준 슬롯은 어노테이션을 찍은 08:00 입니다. 낮 슬롯과 붙이면 «시간이
    #   달라서 다른 건지, 마스크가 틀려서 다른 건지» 를 가릴 수 없습니다.
    pr, pcels = pick_slot("151", "20260918", [8])
    if pr is not None:
        print(f"\n── A. .151 실측 — RGB↔열화상 나란히 (08:00 기준) " + "─"*24)
        pbase = th_base(flip_th(pcels))     # 양쪽을 사람이 보는 방향으로
        for src in srcs:
            view = rgb_view(args.package, src)
            if view is None:
                continue
            stem = src.replace(".png", "")
            mine = [(nm, o) for nm, o in raw_obj if nm == src]
            for L in LAYERS:
                sel = [o for _, o in mine if int(o["layer_mm"]) == L]
                tsel = [o for o in objs if o["layer"] == L and o["src"] == src]
                if not sel:
                    continue
                l = draw_masks(view, [(fill_rgb(view.shape, [rgb_poly(o)]),
                                       LAY_COL[L]) for o in sel], up=1)
                r = draw_masks(pbase, [(flip_th(o["mask"]), LAY_COL[L])
                                       for o in tsel])
                SG.imwrite(
                    os.path.join(d151, f"pair_{stem}_층{L}_{len(sel)}개.png"),
                    SG.label(side_by_side(l, r),
                             f"RGB (left)  vs  .151 thermal 08:00 (right)"
                             f"   -   layer {L} mm  ({len(sel)})",
                             "both sides turned to viewing orientation - "
                             "masks must land on the same leaves"))
            for cls in ("잎", "꽃", "딸기"):
                sel = [o for _, o in mine if o.get("class") == cls]
                tsel = [o for o in objs if o["cls"] == cls and o["src"] == src]
                if not sel:
                    continue
                l = draw_masks(view, [(fill_rgb(view.shape, [rgb_poly(o)]),
                                       CLS_COL[cls]) for o in sel], up=1)
                r = draw_masks(pbase, [(flip_th(o["mask"]), CLS_COL[cls])
                                       for o in tsel])
                SG.imwrite(
                    os.path.join(d151,
                                 f"pair_{stem}_분류_{cls}_{len(sel)}개.png"),
                    SG.label(side_by_side(l, r),
                             f"RGB (left)  vs  .151 thermal 08:00 (right)"
                             f"   -   {CLS_EN[cls]}  ({len(sel)})",
                             "flower/fruit: obvious in RGB, invisible in "
                             "thermal - that is why classes come from RGB"))
            print(f"  {stem}  층 3 + 분류 "
                  f"{sum(1 for c in ('잎','꽃','딸기') if any(o.get('class')==c for _,o in mine))}")

    # ── A-2. 열화상 (투영된 마스크) ─────────────────────────────
    print(f"\n── A. .151 실측 — 열화상 " + "─"*48)
    for day in args.days_151.split(","):
        p, cels = pick_slot("151", day, range(10, 17))
        if p is None:
            print(f"  {day} 점등 슬롯 없음")
            continue
        g = re.search(r"_(\d{2})(\d{2})\d{2}\.y16raw$", p)
        tm = f"{g.group(1)}:{g.group(2)}"
        base = th_base(cels)
        SG.imwrite(os.path.join(d151, f"th151_{day}_00_원본.png"),
                   SG.label(base, f".151  {day} {tm}  -  no mask",
                            f"deviation from frame mean {cels.mean():.2f}C"))
        SG.imwrite(os.path.join(d151, f"th151_{day}_01_전체_{len(objs)}개.png"),
                   SG.label(draw_masks(base, [
                       (o["mask"], CLS_COL.get(o["cls"], (200, 160, 60)))
                       for o in objs]),
                       f".151  {day} {tm}  -  all {len(objs)} projected",
                       "green=leaf  white=flower  red=fruit"))
        for L in LAYERS:
            sel = [o for o in objs if o["layer"] == L]
            SG.imwrite(
                os.path.join(d151, f"th151_{day}_02_층{L}_{len(sel)}개.png"),
                SG.label(draw_masks(base, [(o["mask"], LAY_COL[L])
                                           for o in sel]),
                         f".151  {day} {tm}  -  layer {L} mm  ({len(sel)})",
                         "projected with that layer's H(d)"))
        for cls in ("잎", "꽃", "딸기"):
            sel = [o for o in objs if o["cls"] == cls]
            if not sel:
                continue
            SG.imwrite(
                os.path.join(d151, f"th151_{day}_03_분류_{cls}_{len(sel)}개.png"),
                SG.label(draw_masks(base, [(o["mask"], CLS_COL[cls])
                                           for o in sel]),
                         f".151  {day} {tm}  -  {CLS_EN[cls]}  ({len(sel)})",
                         ("flower/fruit are 4-9 px - drawn in RGB, "
                          "not separable here")))
        print(f"  {day} {tm}  화면평균 {cels.mean():.2f}℃  ->  "
              f"원본+전체+층3+분류3")

    # ── B. `.152` 추정 ─────────────────────────────────────────
    print(f"\n── B. .152 추정 (RGB 없음) " + "─"*46)
    for day in args.days_152.split(","):
        p, cels = pick_slot("152", day, range(10, 17))
        if p is None:
            print(f"  {day} 점등 슬롯 없음")
            continue
        g = re.search(r"_(\d{2})(\d{2})\d{2}\.y16raw$", p)
        tm = f"{g.group(1)}:{g.group(2)}"
        base = th_base(cels)
        SG.imwrite(os.path.join(d152, f"th152_{day}_00_원본.png"),
                   SG.label(base, f".152  {day} {tm}  -  no mask",
                            f"deviation from frame mean {cels.mean():.2f}C"))

        blobs = estimate_leaves(cels)
        SG.imwrite(
            os.path.join(d152, f"th152_{day}_01_잎추정_{len(blobs)}개.png"),
            SG.label(draw_masks(base, [(m, (90, 220, 90)) for m, _, _ in blobs]),
                     f".152  {day} {tm}  -  ESTIMATED leaves ({len(blobs)})",
                     "colder than local background - NOT verified, no RGB"))

        # 개체 후보에 번호를 붙여 육안으로 짚을 수 있게 합니다
        num = draw_masks(base, [(m, (90, 220, 90)) for m, _, _ in blobs])
        for i, (_, a, c) in enumerate(sorted(blobs, key=lambda b: -b[1]), 1):
            x, y = int(c[0]*UP), int(c[1]*UP)
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(num, str(i), (x-6, y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
        SG.imwrite(os.path.join(d152, f"th152_{day}_02_번호.png"),
                   SG.label(num, f".152  {day} {tm}  -  candidates numbered "
                            "(large first)",
                            "use these numbers to tell me which are wrong"))

        areas = np.array([a for _, a, _ in blobs]) if blobs else np.array([0])
        print(f"  {day} {tm}  화면평균 {cels.mean():.2f}℃  "
              f"잎 후보 {len(blobs)}개  면적 중앙 {np.median(areas):.0f}px "
              f"(최소 {areas.min()} 최대 {areas.max()})")

    n = sum(len(glob.glob(os.path.join(d, "*.png"))) for d in (d151, d152))
    print(f"\n  저장  {args.out}/  ({n}장)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
