#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정합에 쓴 이미지 쌍을 저장하고, RGB 화질·화각 겹침을 점검한다.

    python3 scripts/export_pairs.py output/pair_tight/stereo_final.npz \
        calib/th350 calib/th400 calib/th450 --rgb-dir calib/rgb \
        --out output/calib_pairs

내는 것
    raw/      원본 해상도 그대로 (다른 도구에 재사용)
    marked/   코너를 표시한 확인용
    pairs_sheet.png     쌍을 나란히
    thermal_fov_on_rgb.png   RGB 위에 열화상 화각을 그린 것
    rgb_quality.png     노출·포화·ExG 분포
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from calibrate_thermal import object_points, gather            # noqa: E402


def font(sz, bold=False):
    for p in ((r"C:\Windows\Fonts\malgunbd.ttf" if bold else
               r"C:\Windows\Fonts\malgun.ttf"),
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_T, F_B, F_S = font(20, True), font(15), font(13)
INK, INK2, INK3 = (20, 24, 29), (72, 82, 93), (121, 131, 143)
OK_C, ACC_C, NG_C = (27, 104, 69), (189, 77, 14), (161, 35, 23)


def th_frame(path, idx):
    from read_y16 import (load_meta, load_raw, temperature_converter)
    from check_board import to8
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    cel = conv(np.asarray(arr[min(idx, arr.shape[0]-1)])).astype(np.float64)
    return to8(cel), cel


def rgb_frame(vids, t):
    v = next((x for x in vids if x["s0"] <= t <= x["s0"]+x["dur"]), None)
    if v is None:
        return None, None
    cap = cv2.VideoCapture(v["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t-v["s0"])*v["fps"])))
    ok, fr = cap.read()
    cap.release()
    return (fr if ok else None), v["name"]


def main():
    ap = argparse.ArgumentParser(description="정합 이미지 쌍 추출·점검")
    ap.add_argument("stereo")
    ap.add_argument("th_dir", nargs="+")
    ap.add_argument("--rgb-dir", default="calib/rgb")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--depth", type=float, default=450.0,
                    help="열화상 화각을 그릴 깊이 mm")
    ap.add_argument("--out", default="output/calib_pairs")
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    obj = object_points(pat, args.cell)
    for d in ("raw", "marked"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    z = np.load(args.stereo, allow_pickle=True)
    R, T = z["R"], z["T"]
    Kt, Dt, Kr, Dr = z["Kt"], z["Dt"], z["Kr"], z["Dr"]
    names, rgbt = z["names"], z["rgb_t"]
    THC, RGC = z["th_corners"], z["rgb_corners"]
    W, H = (int(v) for v in z["size_rgb"])
    n = len(names)

    files = {}
    for d in args.th_dir:
        files.update(gather(d))
    vids = []
    for f in sorted(os.listdir(args.rgb_dir)):
        m = re.match(r"(\d{2})-(\d{2})-(\d{2})", f)
        if not m or not f.lower().endswith((".mp4", ".avi", ".mkv")):
            continue
        p = os.path.join(args.rgb_dir, f)
        c = cv2.VideoCapture(p)
        fps = c.get(cv2.CAP_PROP_FPS)
        nf = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        c.release()
        s0 = sum(int(v)*k for v, k in zip(m.groups(), (3600, 60, 1)))
        vids.append(dict(path=p, name=f, s0=s0, fps=fps, dur=nf/fps))

    print("=" * 86)
    print(f"정합 이미지 쌍 {n}개 추출")
    print("=" * 86)
    tiles = []
    for i in range(n):
        nm = str(names[i])
        stem, _, fi = nm.partition("#f")
        idx = int(fi) if fi else 0
        path = files.get(stem)
        if path is None:
            print(f"  {stem[:34]:<36}열화상 파일을 못 찾음")
            continue
        g8, cel = th_frame(path, idx)
        fr, vn = rgb_frame(vids, float(rgbt[i]))
        if fr is None:
            print(f"  {stem[:34]:<36}RGB 프레임을 못 읽음")
            continue
        tag = f"pair{i+1:02d}"
        # ── 원본 그대로 ──
        cv2.imwrite(os.path.join(args.out, "raw", f"{tag}_thermal.png"), g8)
        cv2.imwrite(os.path.join(args.out, "raw", f"{tag}_rgb.png"), fr)
        np.save(os.path.join(args.out, "raw", f"{tag}_thermal_celsius.npy"),
                cel.astype(np.float32))
        # ── 코너 표시 ──
        SCt = 4
        vt = cv2.resize(cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR), None,
                        fx=SCt, fy=SCt, interpolation=cv2.INTER_NEAREST)
        for p2 in THC[i]*SCt:
            cv2.circle(vt, tuple(p2.astype(int)), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vt, tuple(p2.astype(int)), 3, (80, 255, 80), -1,
                       cv2.LINE_AA)
        vr = fr.copy()
        for p2 in RGC[i]:
            cv2.circle(vr, tuple(p2.astype(int)), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vr, tuple(p2.astype(int)), 5, (255, 80, 255), -1,
                       cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, "marked", f"{tag}_thermal.png"), vt)
        cv2.imwrite(os.path.join(args.out, "marked", f"{tag}_rgb.png"),
                    cv2.resize(vr, (960, 540)))
        # ── 나란히 ──
        Wt = 640
        a = cv2.resize(vt, (Wt, int(Wt*vt.shape[0]/vt.shape[1])))
        b = cv2.resize(vr, (Wt, int(Wt*vr.shape[0]/vr.shape[1])))
        im = Image.new("RGB", (Wt, a.shape[0]+b.shape[0]+52), (255, 255, 255))
        im.paste(Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)), (0, 0))
        im.paste(Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)),
                 (0, a.shape[0]))
        d = ImageDraw.Draw(im)
        y = a.shape[0]+b.shape[0]+4
        d.text((8, y), f"{tag}  ·  열화상 {stem[-6:]} f{idx}", font=F_B,
               fill=INK)
        d.text((8, y+21), f"RGB {dt.timedelta(seconds=int(rgbt[i]))} "
               f"[{vn}]  ·  정합 오차 {z['errs'][i]:.2f} px", font=F_S,
               fill=OK_C if z["errs"][i] < 1.5 else ACC_C)
        tiles.append(im)
        print(f"  {tag}  열화상 {stem[-6:]} f{idx:<5}"
              f"RGB {dt.timedelta(seconds=int(rgbt[i]))}  "
              f"오차 {z['errs'][i]:.2f} px")

    if tiles:
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (len(tiles)*(tw+10)+10, th+50),
                          (247, 248, 249))
        d = ImageDraw.Draw(sheet)
        d.text((12, 14), f"정합 이미지 쌍 {len(tiles)}개  ·  위=열화상(초록 코너)"
               f"  아래=RGB(자홍 코너)", font=F_T, fill=INK)
        for k, t in enumerate(tiles):
            sheet.paste(t, (10+k*(tw+10), 44))
        sheet.save(os.path.join(args.out, "pairs_sheet.png"))

    # ── 열화상 화각을 RGB 위에 ──
    # H(d) 는 RGB → 열화상. 역으로 열화상 테두리를 RGB 로 보낸다.
    nv = np.array([[0.], [0.], [1.]])
    Hd = Kt @ (R + (T @ nv.T)/args.depth) @ np.linalg.inv(Kr)
    Hi = np.linalg.inv(Hd)
    border = np.array([[[0, 0]], [[159, 0]], [[159, 119]], [[0, 119]]],
                      np.float32)
    poly = cv2.perspectiveTransform(border, Hi).reshape(-1, 2)
    # 판이 없는 프레임(식물만)을 골라 배경으로
    plain, vn = rgb_frame(vids, vids[-1]["s0"]+10)
    if plain is not None:
        vis = plain.copy()
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 0, 0), 7,
                      cv2.LINE_AA)
        cv2.polylines(vis, [poly.astype(np.int32)], True, (60, 220, 255), 3,
                      cv2.LINE_AA)
        cv2.putText(vis, f"THERMAL FOV @ {args.depth:.0f}mm", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(vis, f"THERMAL FOV @ {args.depth:.0f}mm", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (60, 220, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, "thermal_fov_on_rgb.png"), vis)
        ar = cv2.contourArea(poly.astype(np.float32))/(W*H)*100
        print(f"\n  열화상 화각이 RGB 화면의 {ar:.0f} % 를 차지합니다 "
              f"(깊이 {args.depth:g} mm)")
        print(f"    → thermal_fov_on_rgb.png")

    # ── RGB 화질 점검 ──
    if plain is not None:
        b, g, r = (plain[:, :, k].astype(np.float32) for k in range(3))
        sat = ((plain.max(axis=2) >= 250).mean())*100
        exg = 2*g - r - b
        veg = exg > 20
        print(f"\n  RGB 화질 (판 없는 프레임)")
        print(f"    포화 화소 {sat:.1f} %   평균 밝기 "
              f"{plain.mean():.0f}/255   대비(표준편차) {plain.std():.0f}")
        print(f"    ExG > 20 (식물로 보이는) 화소 {veg.mean()*100:.0f} %")
        if sat > 5:
            print(f"    ⚠ 포화가 {sat:.0f} % 입니다. 밝은 잎이 흰색으로 뭉개져"
                  f" ExG 분할이 어려울 수 있습니다.")
        qq = np.zeros((H, W, 3), np.uint8)
        qq[veg] = (60, 200, 60)
        qq[plain.max(axis=2) >= 250] = (60, 60, 240)
        mix = cv2.addWeighted(plain, 0.55, qq, 0.45, 0)
        cv2.putText(mix, "green=ExG>20 (plant)   red=saturated",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6,
                    cv2.LINE_AA)
        cv2.putText(mix, "green=ExG>20 (plant)   red=saturated",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, "rgb_quality.png"), mix)
        print(f"    → rgb_quality.png")

    print(f"\n  저장: {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
