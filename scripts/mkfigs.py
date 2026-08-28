# -*- coding: utf-8 -*-
"""Confluence 첨부용 그림 생성 — 흰 배경·고정 색상.

기울임/배치 그림은 실제 투영 계산으로 그리므로 현장 미리보기와 같은 모양이
나옵니다 (칸 17 px 기준).
"""
import os

import cairosvg
import numpy as np

OUT = "D:/Coding_2025/TmSDK/docs/confluence"
os.makedirs(OUT, exist_ok=True)

INK, INK2, INK3 = "#14181d", "#48525d", "#79838f"
RULE, ACC, PASS_, FAIL = "#c5ccd3", "#bd4d0e", "#1b6845", "#a12317"
FONT = "Malgun Gothic, Segoe UI, sans-serif"

NC, NR, CELL = 8, 5, 30.0          # 칸 8x5, 30 mm
BW, BH = NC*CELL, NR*CELL          # 240 x 150 mm
F, W, H = 208.4, 160, 120
D0 = F*CELL/17.0                   # 칸 17 px 이 되는 거리 ≈ 368 mm


def project(theta_y=0.0, theta_x=0.0, tx=0.0, ty=0.0, d=D0):
    """보드 격자점을 열화상 화소 좌표로 투영한다 → (NR+1, NC+1, 2)."""
    cy_, sy = np.cos(theta_y), np.sin(theta_y)
    cx_, sx = np.cos(theta_x), np.sin(theta_x)
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
    R = Ry @ Rx
    gx, gy = np.meshgrid(np.arange(NC+1)*CELL - BW/2,
                         np.arange(NR+1)*CELL - BH/2)
    p = np.stack([gx, gy, np.zeros_like(gx)], -1)
    q = p @ R.T + np.array([tx, ty, d])
    u = F*q[..., 0]/q[..., 2] + W/2
    v = F*q[..., 1]/q[..., 2] + H/2
    return np.stack([u, v], -1)


def frame(g, k, ox, oy, color, fill_op=.55):
    """투영된 격자를 SVG 조각으로. k = 확대율."""
    s = []
    for j in range(NR):
        for i in range(NC):
            if (i+j) % 2:
                continue                       # 검정 칸만 칠한다
            pts = [g[j, i], g[j, i+1], g[j+1, i+1], g[j+1, i]]
            d = " ".join(f"{ox+p[0]*k:.1f},{oy+p[1]*k:.1f}" for p in pts)
            s.append(f'<polygon points="{d}" fill="{color}" '
                     f'fill-opacity="{fill_op}"/>')
    out = [g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]]
    d = " ".join(f"{ox+p[0]*k:.1f},{oy+p[1]*k:.1f}" for p in out)
    s.append(f'<polygon points="{d}" fill="none" stroke="{color}" '
             f'stroke-width="1.4"/>')
    return "".join(s)


def screen(ox, oy, k, label, thirds=False):
    s = [f'<rect x="{ox}" y="{oy}" width="{W*k}" height="{H*k}" fill="#fbfcfc" '
         f'stroke="{INK}" stroke-width="1.2"/>']
    if thirds:
        for i in (1, 2):
            s.append(f'<line x1="{ox+W*k*i/3}" y1="{oy}" x2="{ox+W*k*i/3}" '
                     f'y2="{oy+H*k}" stroke="{RULE}" stroke-dasharray="3 3"/>')
            s.append(f'<line x1="{ox}" y1="{oy+H*k*i/3}" x2="{ox+W*k}" '
                     f'y2="{oy+H*k*i/3}" stroke="{RULE}" stroke-dasharray="3 3"/>')
    s.append(f'<text x="{ox+W*k/2}" y="{oy+H*k+17}" fill="{INK2}" '
             f'font-size="12" text-anchor="middle">{label}</text>')
    return "".join(s)


def wrap(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" '
            f'fill="#ffffff"/><g font-family="{FONT}">{body}</g></svg>')


def save(name, svg, scale=2.0):
    # 파일명은 ASCII 로 — Confluence 첨부와 cairosvg 경로 처리가 모두 안전합니다.
    p_svg, p_png = f"{OUT}/{name}.svg", f"{OUT}/{name}.png"
    open(p_svg, "w", encoding="utf-8").write(svg)
    png = cairosvg.svg2png(bytestring=svg.encode(), scale=scale)
    open(p_png, "wb").write(png)
    print(f"  {name}.png  ({len(png)/1024:.0f} KB)  + .svg")


# ── 그림 1 · 보드 배치 ───────────────────────────────────────────────
K, GAP = 1.45, 26
FW = W*K
b = []
# 화소 → mm. 보드 반폭 68 px, 반높이 42.5 px 이므로 가장자리 2 px 를 남기려면
# 가로 ±10 px, 세로 ±15.5 px 가 한계입니다. 그 이상은 칸이 잘립니다.
PX = D0/F
b.append(screen(0, 8, K, "01 중앙", True))
b.append(frame(project(), K, 0, 8, INK))
b.append(screen(FW+GAP, 8, K, "05 좌 — 왼쪽 변이 화면 끝에", True))
b.append(frame(project(tx=-10*PX), K, FW+GAP, 8, ACC))
b.append(screen(2*(FW+GAP), 8, K, "09 우하 — 오른쪽 + 아래", True))
b.append(frame(project(tx=10*PX, ty=15.5*PX), K, 2*(FW+GAP), 8, ACC))
b.append(f'<text x="{1.5*FW+GAP}" y="{H*K+44}" fill="{INK3}" font-size="11.5" '
         f'text-anchor="middle">점선은 3×3 판정 칸 · 칸 17 px · '
         f'검정 칸 40개가 전부 온전히 보여야 합니다 (판 여백은 잘려도 됨)</text>')
save("fig1_board_layout", wrap(3*FW+2*GAP, H*K+56, "".join(b)))

# ── 그림 2 · 기울임 ──────────────────────────────────────────────────
K2, GAP2 = 1.9, 40
FW2 = W*K2
b = []
b.append(screen(0, 24, K2, "1차 평균 (8°) — 부족", False))
b.append(frame(project(theta_y=np.radians(8)), K2, 0, 24, FAIL))
b.append(f'<text x="{FW2/2}" y="16" fill="{FAIL}" font-size="13" '
         f'font-weight="bold" text-anchor="middle">× 사각형에 가깝다</text>')
b.append(screen(FW2+GAP2, 24, K2, "목표 (30°) — 충분", False))
b.append(frame(project(theta_y=np.radians(30)), K2, FW2+GAP2, 24, PASS_))
b.append(f'<text x="{FW2*1.5+GAP2}" y="16" fill="{PASS_}" font-size="13" '
         f'font-weight="bold" text-anchor="middle">○ 눈에 띄게 사다리꼴</text>')
b.append(f'<text x="{FW2+GAP2/2}" y="{24+H*K2+42}" fill="{INK3}" '
         f'font-size="11.5" text-anchor="middle">두 그림 모두 같은 거리에서 '
         f'실제 투영 계산으로 그린 것입니다. 30°면 왼쪽 변이 오른쪽 변보다 '
         f'뚜렷하게 큽니다.</text>')
save("fig2_tilt", wrap(2*FW2+GAP2, 24+H*K2+56, "".join(b)))

# ── 그림 3 · 캘리퍼스 ────────────────────────────────────────────────
b = [f'''
<circle cx="150" cy="104" r="36" fill="none" stroke="{RULE}" stroke-width="1.6"/>
<circle cx="150" cy="104" r="21" fill="none" stroke="{INK}" stroke-width="1.8"/>
<circle cx="150" cy="104" r="3" fill="{INK}"/>
<text x="150" y="162" fill="{INK3}" font-size="12" text-anchor="middle">RGB</text>
<circle cx="320" cy="104" r="47" fill="none" stroke="{RULE}" stroke-width="1.6"/>
<circle cx="320" cy="104" r="27" fill="none" stroke="{INK}" stroke-width="1.8"/>
<circle cx="320" cy="104" r="3" fill="{INK}"/>
<text x="320" y="172" fill="{INK3}" font-size="12" text-anchor="middle">열화상</text>

<g stroke="{ACC}" stroke-width="1.6" fill="none">
  <path d="M129 34 L129 96 M347 34 L347 96"/><path d="M129 44 L347 44"/>
  <path d="M134 39 L129 44 L134 49 M342 39 L347 44 L342 49"/></g>
<rect x="205" y="32" width="66" height="20" rx="2" fill="#fff"/>
<text x="212" y="47" fill="{ACC}" font-size="14" font-weight="bold">L_out</text>

<g stroke="{ACC}" stroke-width="1.6" fill="none">
  <path d="M171 112 L171 180 M293 112 L293 180"/><path d="M171 170 L293 170"/>
  <path d="M176 165 L171 170 L176 175 M288 165 L293 170 L288 175"/></g>
<rect x="205" y="158" width="54" height="20" rx="2" fill="#fff"/>
<text x="212" y="173" fill="{ACC}" font-size="14" font-weight="bold">L_in</text>

<path d="M150 104 L320 104" stroke="{INK}" stroke-width="1.6"
      stroke-dasharray="5 4"/>
<rect x="220" y="93" width="30" height="22" rx="2" fill="#fff"/>
<text x="228" y="110" fill="{INK}" font-size="16" font-weight="bold">b</text>

<text x="390" y="98" fill="{INK}" font-size="15" font-weight="bold">b = (L_out + L_in) / 2</text>
<text x="390" y="120" fill="{INK2}" font-size="12">두 렌즈 반지름이 소거되므로</text>
<text x="390" y="138" fill="{INK2}" font-size="12">렌즈 지름도 케이스 간격도</text>
<text x="390" y="156" fill="{INK2}" font-size="12">필요 없습니다.</text>
''']
save("fig3_caliper", wrap(620, 200, "".join(b)))
print(f"\n  → {OUT}")
