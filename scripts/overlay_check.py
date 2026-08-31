#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정합 결과를 눈으로 확인 — RGB 를 열화상 위에 겹쳐 그린다.

    python3 scripts/overlay_check.py output/pair_match/match.npz \
        calib/rgb/xxx.mp4 --out output/overlay
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

import check_board as CB                                       # noqa: E402
from check_board import detect, load_candidates          # noqa: E402
from calibrate_thermal import object_points, gather            # noqa: E402

SC = 4


def font(sz, bold=False):
    for p in ((r"C:\Windows\Fonts\malgunbd.ttf" if bold else
               r"C:\Windows\Fonts\malgun.ttf"),
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description="정합 육안 확인")
    ap.add_argument("match")
    ap.add_argument("video")
    ap.add_argument("--th-dir", default="calib/_all")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--out", default="output/overlay")
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    obj = object_points(pat, args.cell)
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.match, allow_pickle=True)
    s = np.load(os.path.join(os.path.dirname(args.match), "stereo.npz"))
    R, T, Kt, Dt, Kr, Dr = s["R"], s["T"], s["Kt"], s["Dt"], s["Kr"], s["Dr"]
    names, RG, flip = z["names"], z["rgb_corners"], z["flip"]
    TH = z["th_corners"].astype(np.float32)
    times = z["rgb_time"]

    CB.NFRAMES = 1
    files = gather(args.th_dir)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    F_T, F_S = font(17, True), font(13)
    tiles, errs = [], []

    for i, nm in enumerate(names):
        nm = str(nm)
        # ★ 저장된(정밀화된) 코너를 써야 한다. 여기서 다시 검출하면 '가장
        #   선명한' 프레임이 잡혀 매칭에 쓴 프레임과 달라진다.
        c = TH[i].copy()
        if flip[i]:
            c = c.reshape(pat[1], pat[0], 2)[::-1, ::-1].reshape(-1, 2)
        # 그 코너가 나온 프레임을 되찾아 배경으로 쓴다
        gray = None
        keep, CB.PROBE = CB.PROBE, 40
        try:
            cands = load_candidates(files[nm])
        except Exception:
            cands = []
        CB.PROBE = keep
        bd = 1e9
        for _, g8 in cands:
            ok2, c2, _, _ = detect(g8, [pat])
            if not ok2:
                continue
            d2 = float(np.abs(c2.reshape(-1, 2)-TH[i]).mean())
            if d2 < bd:
                bd, gray = d2, g8
        if gray is None:
            continue

        # 그 자세의 실제 보드 평면으로 호모그래피
        rg = RG[i].reshape(-1, 1, 2).astype(np.float32)
        _, rv, tv = cv2.solvePnP(obj, rg, Kr, Dr)
        Rb = cv2.Rodrigues(rv)[0]
        nvec = Rb[:, 2].reshape(3, 1)
        d = float(nvec.ravel() @ tv.reshape(3))
        if d < 0:
            nvec, d = -nvec, -d
        H = Kt @ (R + (T @ nvec.T)/d) @ np.linalg.inv(Kr)

        und = cv2.undistortPoints(rg, Kr, Dr, P=Kr)
        proj = cv2.perspectiveTransform(und, H).reshape(-1, 2)
        undt = cv2.undistortPoints(c.reshape(-1, 1, 2).astype(np.float32),
                                   Kt, Dt, P=Kt).reshape(-1, 2)
        e = float(np.sqrt(((undt-proj)**2).sum(axis=1).mean()))
        errs.append(e)

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, None, fx=SC, fy=SC,
                         interpolation=cv2.INTER_NEAREST)
        # 열화상 실제 코너 = 초록, RGB 를 투영한 것 = 자홍
        for p in undt*SC:
            cv2.circle(vis, tuple(p.astype(int)), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(p.astype(int)), 3, (80, 255, 80), -1,
                       cv2.LINE_AA)
        for p in proj*SC:
            cv2.drawMarker(vis, tuple(p.astype(int)), (0, 0, 0),
                           cv2.MARKER_TILTED_CROSS, 11, 3, cv2.LINE_AA)
            cv2.drawMarker(vis, tuple(p.astype(int)), (255, 80, 255),
                           cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA)
        for a, b in zip(undt*SC, proj*SC):
            cv2.line(vis, tuple(a.astype(int)), tuple(b.astype(int)),
                     (0, 200, 255), 1, cv2.LINE_AA)

        # 짝지어진 RGB 프레임
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(times[i]*fps)))
        okv, fr = cap.read()
        rgb = cv2.resize(fr, (vis.shape[1], int(vis.shape[1]*fr.shape[0] /
                                                fr.shape[1]))) if okv else None
        if rgb is not None:
            for p in RG[i]:
                cv2.circle(rgb, tuple((p*rgb.shape[1]/fr.shape[1])
                                      .astype(int)), 3, (255, 80, 255), -1,
                           cv2.LINE_AA)

        hh = vis.shape[0] + (rgb.shape[0] if rgb is not None else 0) + 62
        im = Image.new("RGB", (vis.shape[1], hh), (255, 255, 255))
        im.paste(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), (0, 0))
        if rgb is not None:
            im.paste(Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)),
                     (0, vis.shape[0]))
        dr = ImageDraw.Draw(im)
        y = hh-58
        dr.text((10, y), nm, font=F_T, fill=(20, 24, 29))
        dr.text((10, y+22), f"RGB {times[i]:.2f}s · 정합 오차 {e:.2f} px",
                font=F_S, fill=(27, 104, 69) if e < 1.5 else (189, 77, 14))
        dr.text((10, y+38), "초록=열화상 실제코너 · 자홍=RGB 투영",
                font=F_S, fill=(121, 131, 143))
        im.save(os.path.join(args.out, f"overlay_{nm}.png"))
        tiles.append(im)
        print(f"  {nm:<12}{times[i]:>8.2f}s{e:>9.2f} px")

    cap.release()
    if tiles:
        tw = max(t.size[0] for t in tiles)
        th = max(t.size[1] for t in tiles)
        cols = 3
        rows = (len(tiles)+cols-1)//cols
        sheet = Image.new("RGB", (cols*tw+(cols+1)*12, 52+rows*(th+12)),
                          (247, 248, 249))
        d = ImageDraw.Draw(sheet)
        d.text((14, 12), f"RGB → 열화상 정합 확인 · 짝 {len(tiles)}개 · "
               f"평균 오차 {np.mean(errs):.2f} px", font=F_T,
               fill=(20, 24, 29))
        d.text((14, 32), "두 표시가 겹칠수록 정합이 정확합니다. "
               "노란 선이 오차입니다.", font=F_S, fill=(72, 82, 93))
        for k, t in enumerate(tiles):
            r, c = divmod(k, cols)
            sheet.paste(t, (12+c*(tw+12), 52+r*(th+12)))
        sheet.save(os.path.join(args.out, "overlay_sheet.png"))
        print(f"\n  평균 {np.mean(errs):.2f} px · {args.out}/overlay_sheet.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
