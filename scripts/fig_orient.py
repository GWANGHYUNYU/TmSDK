# -*- coding: utf-8 -*-
"""왜 눕혀 들어야 하는가 — 방향별로 담을 수 있는 최대 크기를 그림으로.

각 방향에서 보드가 화면(160x120)에 들어가는 최대 크기를 계산해서 그립니다.
초점거리는 줄자 실측으로 확정된 147.5 px 를 씁니다.
"""
import os

import cairosvg
import numpy as np

OUT = "D:/Coding_2025/TmSDK/docs/confluence"
INK, INK2, INK3 = "#14181d", "#48525d", "#79838f"
RULE, ACC, PASS_, FAIL = "#c5ccd3", "#bd4d0e", "#1b6845", "#a12317"
FONT = "Malgun Gothic, Segoe UI, sans-serif"

NC, NR, CELL = 8, 5, 30.0
W, H, F = 160, 120, 147.5           # ★ 줄자로 확정된 초점거리
MARGIN = 2


def max_square(rot_deg):
    """면내회전 rot 일 때 화면에 들어가는 최대 칸 크기(px)."""
    a = np.radians(rot_deg)
    # 보드 8s x 5s 를 rot 만큼 돌린 외접 사각형
    wf = abs(NC*np.cos(a)) + abs(NR*np.sin(a))
    hf = abs(NC*np.sin(a)) + abs(NR*np.cos(a))
    return min((W-2*MARGIN)/wf, (H-2*MARGIN)/hf)


def grid(s, rot_deg):
    """칸 s px, 면내회전 rot 인 격자점 (NR+1, NC+1, 2)."""
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    gx, gy = np.meshgrid((np.arange(NC+1)-NC/2)*s, (np.arange(NR+1)-NR/2)*s)
    p = np.stack([gx, gy], -1) @ R.T
    return p + np.array([W/2, H/2])


def draw(g, k, ox, oy, color):
    out = []
    for j in range(NR):
        for i in range(NC):
            if (i+j) % 2:
                continue
            pts = [g[j, i], g[j, i+1], g[j+1, i+1], g[j+1, i]]
            d = " ".join(f"{ox+p[0]*k:.1f},{oy+p[1]*k:.1f}" for p in pts)
            out.append(f'<polygon points="{d}" fill="{color}" '
                       f'fill-opacity="0.55"/>')
    c = [g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]]
    d = " ".join(f"{ox+p[0]*k:.1f},{oy+p[1]*k:.1f}" for p in c)
    out.append(f'<polygon points="{d}" fill="none" stroke="{color}" '
               f'stroke-width="1.6"/>')
    return "".join(out)


K, GAP, TOP = 2.0, 34, 46
FW, FH = W*K, H*K
CASES = [(0, "눕혀서 — 긴 변 수평", PASS_, "권장"),
         (45, "45° 돌려서", FAIL, "1차에서 6/10건"),
         (90, "세워서", ACC, "")]

b = []
for idx, (rot, title, col, note) in enumerate(CASES):
    ox = idx*(FW+GAP)
    s = max_square(rot)
    g = grid(s, rot)
    area = (NC*s)*(NR*s)/(W*H)*100
    d_mm = F*CELL/s
    b.append(f'<rect x="{ox}" y="{TOP}" width="{FW}" height="{FH}" '
             f'fill="#fbfcfc" stroke="{INK}" stroke-width="1.3"/>')
    b.append(draw(g, K, ox, TOP, col))
    b.append(f'<text x="{ox+FW/2}" y="{TOP-24}" fill="{col}" font-size="17" '
             f'font-weight="bold" text-anchor="middle">{title}</text>')
    if note:
        b.append(f'<text x="{ox+FW/2}" y="{TOP-7}" fill="{INK3}" '
                 f'font-size="12" text-anchor="middle">{note}</text>')
    y = TOP+FH+26
    b.append(f'<text x="{ox+FW/2}" y="{y}" fill="{INK}" font-size="21" '
             f'font-weight="bold" text-anchor="middle">칸 {s:.1f} px</text>')
    b.append(f'<text x="{ox+FW/2}" y="{y+22}" fill="{INK2}" font-size="14" '
             f'text-anchor="middle">보드가 화면의 {area:.0f} %</text>')
    b.append(f'<text x="{ox+FW/2}" y="{y+41}" fill="{INK3}" font-size="12.5" '
             f'text-anchor="middle">이때 카메라~보드 {d_mm:.0f} mm</text>')

# cairosvg 는 tspan 을 부모 text 의 x 에 겹쳐 그린다. 줄마다 별도 text 로.
TW = 3*FW+2*GAP
b.append(f'<text x="{TW/2}" y="{TOP+FH+96}" fill="{INK2}" font-size="13.5" '
         f'text-anchor="middle">세 그림 모두 화면에 들어가는 최대 크기입니다. '
         f'45°로 들면 보드 대각선이 화면 세로 120 px 에 먼저 걸려 '
         f'더 가까이 갈 수가 없습니다.</text>')
b.append(f'<text x="{TW/2}" y="{TOP+FH+120}" fill="{PASS_}" font-size="16" '
         f'font-weight="bold" text-anchor="middle">눕혀 들면 칸이 55 % 크고 '
         f'보드가 담기는 면적은 2.4 배가 됩니다.</text>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{TW}" '
       f'height="{TOP+FH+140}" viewBox="0 0 {TW} {TOP+FH+140}">'
       f'<rect width="{TW}" height="{TOP+FH+140}" fill="#ffffff"/>'
       f'<g font-family="{FONT}">{"".join(b)}</g></svg>')

open(f"{OUT}/fig4_orientation.svg", "w", encoding="utf-8").write(svg)
png = cairosvg.svg2png(bytestring=svg.encode(), scale=2.0)
open(f"{OUT}/fig4_orientation.png", "wb").write(png)
print(f"  fig4_orientation.png ({len(png)/1024:.0f} KB)")
for rot, t, _, _ in CASES:
    s = max_square(rot)
    print(f"    {t:<22}칸 {s:5.1f} px · 보드 면적 "
          f"{(NC*s)*(NR*s)/(W*H)*100:4.0f} % · 거리 {F*CELL/s:3.0f} mm")
