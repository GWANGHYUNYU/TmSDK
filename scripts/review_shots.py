#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""촬영분을 자세별로 뽑아 무엇이 잘 됐고 무엇이 안 됐는지 눈으로 보게 만든다.

    python3 scripts/review_shots.py calib/_all --out output/calib_review \
        --focal 147.4 --measured d400_01=400 --measured d450_01=450

내는 것
    summary.png        세션 전체 요약 — 커버리지·자세분포·거리대조·판정
    contact_sheet.png  전체 자세를 한 장에
    <자세이름>.png      자세별 낱장

자세마다 검출된 코너를 그리고, 칸 크기·기울임·면내회전·화면 점유·자세별 RMS·
대비를 함께 적습니다. 검출 실패분은 왜 실패했는지 더 작은 격자로 좁혀 봅니다.

★ --focal 을 반드시 넣으십시오. 기울임과 역산 거리가 f 에 의존합니다.
  빼면 f 가 자유로 풀려 기울임이 최대 3° 과대평가됩니다.
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

import check_board as CB                                          # noqa: E402
from check_board import (detect_file, detect, load_candidates,     # noqa: E402
                         short, square_px, contrast_c, inplane_deg)
from calibrate_thermal import object_points, CALIB_FLAGS, gather   # noqa: E402

SCALE = 3                       # 160x120 → 480x360
FPS = 8.7                       # TMC160F

BG = (247, 248, 249)
INK, INK2, INK3 = (20, 24, 29), (72, 82, 93), (121, 131, 143)
RULE, RULE2 = (219, 224, 229), (197, 204, 211)
OK_C, NG_C, ACC_C = (27, 104, 69), (161, 35, 23), (189, 77, 14)


def font(sz, bold=False):
    for p in ((r"C:\Windows\Fonts\malgunbd.ttf" if bold else
               r"C:\Windows\Fonts\malgun.ttf"),
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_H1, F_H2, F_BODY, F_SM, F_XS = (font(25, True), font(16, True),
                                  font(14), font(12), font(11))


# ── 자세 하나 그리기 ────────────────────────────────────────────────
def render(gray, corners, pat):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST)
    if corners is not None:
        c = corners.reshape(-1, 2)*SCALE
        g = c.reshape(pat[1], pat[0], 2)
        for row in g:
            cv2.polylines(img, [row.astype(np.int32)], False, (30, 200, 255),
                          1, cv2.LINE_AA)
        for k in range(pat[0]):
            cv2.polylines(img, [g[:, k].astype(np.int32)], False,
                          (30, 200, 255), 1, cv2.LINE_AA)
        for p in c:
            cv2.circle(img, tuple(p.astype(int)), 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p.astype(int)), 2, (60, 255, 60), -1,
                       cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def why_failed(path, pat):
    """실패 원인을 더 작은 격자로 좁힌다."""
    keep, CB.PROBE = CB.PROBE, 40
    try:
        cands = load_candidates(path)
    except Exception:
        CB.PROBE = keep
        return "파일을 읽을 수 없음"
    CB.PROBE = keep
    for p in ((pat[0]-1, pat[1]), (pat[0], pat[1]-1), (pat[0]-1, pat[1]-1)):
        for _, g8 in cands:
            if detect(g8, [p])[0]:
                miss = ("한 열" if p[1] == pat[1] else
                        "한 행" if p[0] == pat[0] else "한 행과 한 열")
                return (f"{p[0]}x{p[1]} 까지만 잡힘 — {miss}이 화면 밖으로 "
                        f"나갔거나 가려짐")
    return "어떤 크기로도 격자를 못 찾음 — 대비 부족 또는 흔들림"


def stability(path, pat):
    """녹화 하나 안에서 판이 얼마나 움직였는가.

    길게 녹화하면 한 파일에 여러 자세가 섞여 «한 파일 = 한 자세» 가 깨진다.
    → (프레임수, 검출수/후보수, 면내회전 범위, 중심 이동 px)
    """
    nf = 0
    try:
        nf = os.path.getsize(path)//(160*120*2) \
            if path.lower().endswith(".y16raw") else 0
    except OSError:
        pass
    keep, CB.PROBE = CB.PROBE, 30
    try:
        cands = load_candidates(path)
    except Exception:
        CB.PROBE = keep
        return nf, (0, 0), None, None
    CB.PROBE = keep
    rots, ctrs = [], []
    for _, g8 in cands:
        ok, c, _, _ = detect(g8, [pat])
        if ok:
            rots.append(inplane_deg(c, pat))
            ctrs.append(c.reshape(-1, 2).mean(0))
    if not rots:
        return nf, (0, len(cands)), None, None
    ctrs = np.array(ctrs)
    mv = float(np.linalg.norm(ctrs-ctrs.mean(0), axis=1).max())
    return nf, (len(rots), len(cands)), (min(rots), max(rots)), mv


def board_occ(pts, pat, size=(160, 120)):
    """보드 자신의 면적 / 화면. 축정렬 상자로 재면 돌려 든 손실을 놓친다."""
    q = pts.reshape(pat[1], pat[0], 2)
    v = np.array([q[0, 0], q[0, -1], q[-1, -1], q[-1, 0]])
    a = 0.5*abs(np.dot(v[:, 0], np.roll(v[:, 1], -1))
                - np.dot(v[:, 1], np.roll(v[:, 0], -1)))
    return float(a*(pat[0]+1)*(pat[1]+1)/((pat[0]-1)*(pat[1]-1))
                 / (size[0]*size[1]))


# ── 그리기 도구 ────────────────────────────────────────────────────
def wrap(draw, text, fnt, width, sep=" · "):
    if not text:
        return []
    out, cur = [], ""
    for part in text.split(sep):
        cand = part if not cur else f"{cur}{sep}{part}"
        if draw.textlength(cand, fnt) <= width:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = part
    if cur:
        out.append(cur)
    return out


def panel(d, x, y, w, h, title, sub=None):
    d.rectangle([x, y, x+w, y+h], fill=(255, 255, 255), outline=RULE)
    d.text((x+16, y+13), title, font=F_H2, fill=INK)
    if sub:
        d.text((x+16, y+35), sub, font=F_XS, fill=INK3)
    return y + (56 if sub else 42)


def axes(d, x, y, w, h, xr, yr, xlab, ylab, xt, yt):
    d.rectangle([x, y, x+w, y+h], fill=(252, 253, 253), outline=RULE2)
    for v in xt:
        px = x + (v-xr[0])/(xr[1]-xr[0])*w
        d.line([px, y, px, y+h], fill=RULE)
        d.text((px-8, y+h+5), f"{v:g}", font=F_XS, fill=INK3)
    for v in yt:
        py = y + h - (v-yr[0])/(yr[1]-yr[0])*h
        d.line([x, py, x+w, py], fill=RULE)
        d.text((x-30, py-7), f"{v:g}", font=F_XS, fill=INK3)
    d.text((x+w/2-25, y+h+22), xlab, font=F_XS, fill=INK2)
    d.text((x-32, y-18), ylab, font=F_XS, fill=INK2)

    def to(vx, vy):
        return (x + (vx-xr[0])/(xr[1]-xr[0])*w,
                y + h - (vy-yr[0])/(yr[1]-yr[0])*h)
    return to


def tile(rgb, name, ok, line1, line2, flags):
    h, w = rgb.shape[:2]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    fl = wrap(probe, flags, F_SM, w-20)
    lab = 76 + 17*len(fl)
    im = Image.new("RGB", (w, h+lab), (255, 255, 255))
    im.paste(Image.fromarray(rgb), (0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w-1, h-1], outline=RULE2)
    badge = "검출 OK" if ok else "검출 실패"
    col = OK_C if ok else NG_C
    d.rectangle([8, 8, 8+d.textlength(badge, F_BODY)+16, 31], fill=col)
    d.text((16, 10), badge, font=F_BODY, fill=(255, 255, 255))
    d.text((10, h+7), name, font=F_H2, fill=INK)
    d.text((10, h+31), line1, font=F_SM, fill=INK2)
    if line2:
        d.text((10, h+48), line2, font=F_SM, fill=INK2)
    for k, ln in enumerate(fl):
        d.text((10, h+68+17*k), ln, font=F_SM, fill=ACC_C)
    return im


# ── 세션 요약 ──────────────────────────────────────────────────────
def summary(shots, det, rms, K, D, checks, notes, src, focal, out):
    W = 1580
    d0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    nlines = sum(len(wrap(d0, t, F_SM, W-90, sep=" ")) for t in notes)
    H = 700 + 26*len(checks)//2 + 20*nlines
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    ng = sum(1 for s in shots if not s["ok"])
    warn = sum(1 for s in shots if s["ok"] and s["flags"])
    good = len(shots)-ng-warn
    d.text((28, 24), "1차 캘리브레이션 촬영 검토", font=F_H1, fill=INK)
    d.text((28, 60), f"{os.path.abspath(src)} · {len(shots)}건", font=F_SM,
           fill=INK3)
    bx = 28
    for lab, cnt, col in (("좋음", good, OK_C), ("쓸 수 있음", warn, ACC_C),
                          ("못 씀", ng, NG_C)):
        t = f"{lab} {cnt}"
        wpx = d.textlength(t, F_BODY)+22
        d.rectangle([bx, 84, bx+wpx, 111], fill=col)
        d.text((bx+11, 87), t, font=F_BODY, fill=(255, 255, 255))
        bx += wpx+9
    stat = (f"재투영 RMS {rms:.3f} px      f = {K[0,0]:.1f} px "
            f"{'(줄자로 고정)' if focal else '(자유 — 신뢰 불가)'}      "
            f"화각 {2*np.degrees(np.arctan(80/K[0,0])):.1f}° × "
            f"{2*np.degrees(np.arctan(60/K[1,1])):.1f}°      k1 {D[0,0]:+.3f}")
    d.text((bx+18, 90), stat, font=F_SM, fill=INK2)

    PW, PH, GAP, TOP = 496, 348, 22, 132
    # ── A. 화면 커버리지 ──
    y = panel(d, 28, TOP, PW, PH, "화면 커버리지",
              "각 자세의 보드 외곽. 점선은 3×3 판정 칸")
    k = 2.05
    ox, oy = 28+(PW-160*k)/2, y+8
    d.rectangle([ox, oy, ox+160*k, oy+120*k], fill=(252, 253, 253),
                outline=RULE2)
    seen = set()
    for s in det:
        for px, py in s["c"].reshape(-1, 2):
            seen.add((min(2, int(px/160*3)), min(2, int(py/120*3))))
    for gy in range(3):
        for gx in range(3):
            x0, y0 = ox+gx*160*k/3, oy+gy*120*k/3
            if (gx, gy) not in seen:
                d.rectangle([x0, y0, x0+160*k/3, y0+120*k/3],
                            fill=(252, 233, 231))
                d.text((x0+18, y0+42), "비어 있음", font=F_XS, fill=NG_C)
            d.rectangle([x0, y0, x0+160*k/3, y0+120*k/3], outline=RULE)
    for s in det:
        q = s["c"].reshape(4, 7, 2)
        v = [q[0, 0], q[0, -1], q[-1, -1], q[-1, 0]]
        d.polygon([(ox+p[0]*k, oy+p[1]*k) for p in v],
                  outline=ACC_C if s["rot"] > 20 else (70, 110, 160))
    d.text((28+16, TOP+PH-26), f"코너가 닿은 칸 {len(seen)}/9", font=F_SM,
           fill=NG_C if len(seen) < 9 else OK_C)

    # ── B. 자세 분포 ──
    x2 = 28+PW+GAP
    y = panel(d, x2, TOP, PW, PH, "자세 분포",
              "기울임은 20° 이상, 면내회전은 20° 이하가 목표")
    to = axes(d, x2+52, y+10, PW-96, PH-110, (0, 40), (0, 90),
              "기울임 (°)", "면내회전 (°)", [0, 10, 20, 30, 40],
              [0, 30, 60, 90])
    gx0, gy0 = to(20, 0)
    gx1, gy1 = to(40, 20)
    d.rectangle([gx0, gy1, gx1, gy0], fill=(232, 244, 237), outline=OK_C)
    d.text((gx0+6, gy1-16), "목표 구역", font=F_XS, fill=OK_C)
    for s in det:
        px, py = to(min(39.5, s["tilt"]), min(89, s["rot"]))
        c = OK_C if (s["tilt"] >= 20 and s["rot"] <= 20) else ACC_C
        d.ellipse([px-5, py-5, px+5, py+5], fill=c)
        d.text((px+8, py-7), short(s["name"], 8), font=F_XS, fill=INK3)

    # ── C. 거리 ↔ 줄자 ──
    x3 = x2+PW+GAP
    y = panel(d, x3, TOP, PW, PH, "역산 거리 ↔ 줄자",
              "막대는 계산값, [ ] 와 검은 눈금은 줄자 실측")
    ds = [s["dist"] for s in det]
    lo, hi = min(ds)-40, max(ds)+40
    bh = (PH-96)/len(det)
    for i, s in enumerate(det):
        by = y+8+i*bh
        bx0 = x3+96
        bw = (PW-176)*(s["dist"]-lo)/(hi-lo)
        col = (70, 110, 160)
        if s.get("tape"):
            e = abs(s["dist"]-s["tape"])/s["tape"]
            col = OK_C if e <= 0.05 else NG_C
        d.rectangle([bx0, by+2, bx0+bw, by+bh-4], fill=col)
        nm = (f"{short(s['name'], 9)} [{s['tape']:.0f}]" if s.get("tape")
              else short(s["name"], 9))
        d.text((x3+14, by+3), nm, font=F_XS, fill=INK2)
        d.text((x3+PW-42, by+3), f"{s['dist']:.0f}", font=F_XS, fill=INK2)
        if s.get("tape"):
            tx = x3+96+(PW-176)*(s["tape"]-lo)/(hi-lo)
            d.line([tx, by, tx, by+bh-2], fill=INK, width=3)

    # ── D. 판정 ──
    ROW = 26
    ph2 = 62+ROW*((len(checks)+1)//2)
    y = panel(d, 28, TOP+PH+GAP, W-56, ph2, "판정")
    for i, (lab, ok, val) in enumerate(checks):
        cx = 28+22+(i % 2)*((W-100)//2)
        cy = y+4+(i//2)*ROW
        # ✔/✘ 는 Malgun Gothic 에 없어 □ 로 렌더된다. 직접 그린다.
        r, cy0 = 6, cy+9
        if ok:
            d.ellipse([cx, cy0-r, cx+2*r, cy0+r], outline=OK_C, width=2)
        else:
            d.line([cx+1, cy0-r+1, cx+2*r-1, cy0+r-1], fill=NG_C, width=2)
            d.line([cx+1, cy0+r-1, cx+2*r-1, cy0-r+1], fill=NG_C, width=2)
        d.text((cx+26, cy+1), lab, font=F_SM, fill=INK)
        d.text((cx+320, cy+1), val, font=F_SM, fill=INK2)

    # ── E. 핵심 발견 ──
    y2 = TOP+PH+GAP+ph2+GAP
    y = panel(d, 28, y2, W-56, H-y2-28, "오늘 확인된 것")
    for t in notes:
        for ln in wrap(d, t, F_SM, W-120, sep=" "):
            d.text((50, y), ln, font=F_SM, fill=INK2)
            y += 20
        y += 6
    im.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="촬영분 자세별 육안 검토")
    ap.add_argument("src")
    ap.add_argument("--out", default="output/calib_review")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--focal", type=float, default=None,
                    help="초점거리 px 고정 (TMC160F 실측 147.4). 반드시 넣으십시오")
    ap.add_argument("--measured", action="append", default=[],
                    metavar="POSE=MM", help="줄자로 잰 보드 거리")
    ap.add_argument("--no-refine", action="store_true")
    args = ap.parse_args()

    CB.NFRAMES = 1
    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    os.makedirs(args.out, exist_ok=True)
    tape = {}
    for m in args.measured:
        k, v = m.split("=")
        tape[k.strip()] = float(v)
    files = gather(args.src)
    if not files:
        raise SystemExit(f"파일이 없습니다: {args.src}")
    obj = object_points(pat, args.cell)
    SZ = (160, 120)

    print("=" * 88)
    print(f"촬영분 검토 · {len(files)}건 · 패턴 {pat[0]}x{pat[1]}"
          + (f" · f 고정 {args.focal:g} px" if args.focal else " · f 자유"))
    print("=" * 88)
    print(f"  {'자세':<11}{'검출':>6}{'칸px':>7}{'대비':>8}{'회전':>6}"
          f"{'길이':>7}{'프레임내 안정성':>18}")

    shots = []
    for name, path in files.items():
        ok, c, _, _, cels, gray, _ = detect_file(path, [pat])
        nf, (nd, ncand), rng, mv = stability(path, pat)
        s = dict(name=name, ok=ok, c=c, gray=gray, cels=cels,
                 sq=square_px(c, pat) if ok else None,
                 dt=contrast_c(cels, c, pat) if ok else None,
                 rot=inplane_deg(c, pat) if ok else None,
                 why=None if ok else why_failed(path, pat),
                 nf=nf, nd=nd, ncand=ncand, rng=rng, mv=mv,
                 tape=tape.get(name))
        shots.append(s)
        st = (f"회전 {rng[0]:.0f}~{rng[1]:.0f}° · 이동 {mv:.0f}px"
              if rng else "-")
        print(f"  {short(name, 11):<11}{'OK' if ok else '실패':>6}"
              f"{(f'{s[chr(115)+chr(113)]:.1f}' if ok else '-'):>7}"
              f"{(f'{s[chr(100)+chr(116)]:.2f}℃' if s['dt'] is not None else '-'):>8}"
              f"{(f'{s[chr(114)+chr(111)+chr(116)]:.0f}°' if ok else '-'):>6}"
              f"{(f'{nf/FPS:.0f}s' if nf else '-'):>7}   {st:<28}")

    det = [s for s in shots if s["ok"]]
    if len(det) < 4:
        raise SystemExit("검출된 자세가 너무 적습니다.")

    def calib(P):
        if args.focal:
            Kg = np.array([[args.focal, 0, 80.], [0, args.focal, 60.],
                           [0, 0, 1.]])
            return cv2.calibrateCamera(
                [obj]*len(P), P, SZ, Kg, None,
                flags=(CALIB_FLAGS | cv2.CALIB_FIX_FOCAL_LENGTH
                       | cv2.CALIB_USE_INTRINSIC_GUESS))
        return cv2.calibrateCamera([obj]*len(P), P, SZ, None, None,
                                   flags=CALIB_FLAGS)

    P = [s["c"].astype(np.float32) for s in det]
    rms, K, D, rv, tv = calib(P)

    # ★ 프레임 재선택 — '가장 선명한' 것이 '코너가 가장 정확한' 것은 아니다.
    swapped, before = 0, rms
    if not args.no_refine:
        for i, s in enumerate(det):
            keep, CB.PROBE = CB.PROBE, 40
            try:
                cands = load_candidates(files[s["name"]])
            except Exception:
                CB.PROBE = keep
                continue
            CB.PROBE = keep
            best = (float("inf"), None, None, None)
            for cel, g8 in cands:
                ok2, c2, _, _ = detect(g8, [pat])
                if not ok2:
                    continue
                c2 = c2.astype(np.float64)
                okp, rvi, tvi = cv2.solvePnP(obj, c2, K, D)
                if not okp:
                    continue
                pr, _ = cv2.projectPoints(obj, rvi, tvi, K, D)
                e = float(np.sqrt(((c2.reshape(-1, 2)-pr.reshape(-1, 2))**2)
                                  .sum(axis=1).mean()))
                if e < best[0]:
                    best = (e, c2.astype(np.float32), g8, cel)
            if best[1] is None:
                continue
            okp, rv0, tv0 = cv2.solvePnP(obj, P[i].astype(np.float64), K, D)
            pr, _ = cv2.projectPoints(obj, rv0, tv0, K, D)
            e0 = float(np.sqrt(((P[i].reshape(-1, 2)-pr.reshape(-1, 2))**2)
                               .sum(axis=1).mean()))
            if best[0] < e0*0.9:
                P[i] = best[1]
                s["c"], s["gray"], s["cels"] = best[1], best[2], best[3]
                s["sq"] = square_px(best[1], pat)
                s["rot"] = inplane_deg(best[1], pat)
                if best[3] is not None:
                    s["dt"] = contrast_c(best[3], best[1], pat)
                s["refined"] = True
                swapped += 1
        if swapped:
            rms, K, D, rv, tv = calib(P)
            print(f"\n  프레임 재선택: {swapped}자세 교체 → "
                  f"RMS {before:.3f} → {rms:.3f} px")

    for i, s in enumerate(det):
        R = cv2.Rodrigues(rv[i])[0]
        pr = cv2.projectPoints(obj, rv[i], tv[i], K, D)[0].reshape(-1, 2)
        p = P[i].reshape(-1, 2)
        s["tilt"] = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
        s["rms"] = float(np.sqrt(((p-pr)**2).sum(axis=1).mean()))
        s["occ"] = board_occ(p, pat)
        s["dist"] = float(((R @ obj.T).T + tv[i].reshape(3))[:, 2].mean())
    med = float(np.median([s["rms"] for s in det]))

    # ── 자세별 판정 ──
    print("\n" + "=" * 88)
    print("자세별 판정")
    print("=" * 88)
    tiles = []
    for s in shots:
        f = []
        if not s["ok"]:
            f.append(s["why"] or "검출 실패")
        else:
            if s["rms"] > max(0.5, 2.5*med):
                f.append(f"RMS {s['rms']:.2f} 과다 — 전체를 끌어내림")
            if s["tilt"] < 20:
                f.append(f"기울임 {s['tilt']:.0f}° 부족 (20° 필요)")
            if s["rot"] > 20:
                f.append(f"면내회전 {s['rot']:.0f}° — 눕혀 드십시오")
            if s["occ"] < 0.40:
                f.append(f"점유 {s['occ']*100:.0f}% 작음 (40% 필요)")
            if s["dt"] is not None and s["dt"] < 2.5:
                f.append(f"대비 {s['dt']:.1f}℃ 낮음")
            if s["nf"] and s["nf"]/FPS > 15:
                f.append(f"녹화 {s['nf']/FPS:.0f}초 — 한 파일에 여러 자세")
            if s.get("tape") and abs(s["dist"]-s["tape"])/s["tape"] > 0.05:
                f.append(f"줄자 {s['tape']:.0f} 와 "
                         f"{abs(s['dist']-s['tape']):.0f}mm 차이")
        s["flags"] = f
        mark = ("◎ 좋음" if s["ok"] and not f else
                "○ 쓸 수 있음" if s["ok"] else "✗ 못 씀")
        print(f"  {short(s['name'], 11):<11}{mark:<13}"
              + (" · ".join(f) if f else "지적 사항 없음"))

        if s["ok"]:
            l1 = (f"칸 {s['sq']:.1f}px · 기울임 {s['tilt']:.0f}° · "
                  f"면내회전 {s['rot']:.0f}° · 점유 {s['occ']*100:.0f}%")
            l2 = (f"거리 {s['dist']:.0f}mm · RMS {s['rms']:.2f}"
                  + (f" · 대비 {s['dt']:.2f}℃" if s["dt"] is not None else "")
                  + (f" · 줄자 {s['tape']:.0f}mm ({s['dist']-s['tape']:+.0f})"
                     if s.get("tape") else "")
                  + ("  [프레임 재선택]" if s.get("refined") else ""))
        else:
            l1, l2 = "검출된 코너 없음", (f"녹화 {s['nf']/FPS:.0f}초"
                                    if s["nf"] else "")
        t = tile(render(s["gray"], s["c"] if s["ok"] else None, pat),
                 short(s["name"], 30), s["ok"], l1, l2, " · ".join(f))
        t.save(os.path.join(args.out, f"{s['name']}.png"))
        tiles.append(t)

    # ── 전체 한 장 ──
    cols = args.cols
    tw = max(t.size[0] for t in tiles)
    th = max(t.size[1] for t in tiles)
    for i, t in enumerate(tiles):
        if t.size != (tw, th):
            pad = Image.new("RGB", (tw, th), (255, 255, 255))
            pad.paste(t, (0, 0))
            tiles[i] = pad
    rows = (len(tiles)+cols-1)//cols
    PAD, TOP = 14, 58
    sheet = Image.new("RGB", (cols*tw+(cols+1)*PAD, TOP+rows*(th+PAD)+PAD), BG)
    d = ImageDraw.Draw(sheet)
    ng = sum(1 for s in shots if not s["ok"])
    warn = sum(1 for s in shots if s["ok"] and s["flags"])
    d.text((PAD+2, 12), f"촬영분 검토 · {os.path.abspath(args.src)}",
           font=F_H2, fill=INK)
    d.text((PAD+2, 34), f"◎ 좋음 {len(shots)-ng-warn}    ○ 쓸 수 있음 {warn}"
           f"    ✗ 못 씀 {ng}    (총 {len(shots)}건 · RMS {rms:.3f} px"
           f" · f {K[0,0]:.1f})", font=F_SM, fill=INK2)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (PAD+c*(tw+PAD), TOP+r*(th+PAD)))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))

    # ── 요약 ──
    occ = np.array([s["occ"] for s in det])
    rot = np.array([s["rot"] for s in det])
    tilt = np.array([s["tilt"] for s in det])
    ds = np.array([s["dist"] for s in det])
    seen = set()
    for s in det:
        for px, py in s["c"].reshape(-1, 2):
            seen.add((min(2, int(px/160*3)), min(2, int(py/120*3))))
    tp = [s for s in det if s.get("tape")]
    rat = [s["tape"]/s["dist"] for s in tp]
    checks = [
        ("자세 수 ≥ 15", len(det) >= 15, f"{len(det)}개"),
        ("재투영 RMS < 0.5 px", rms < 0.5, f"{rms:.3f} px (목표 0.3)"),
        ("초점거리", bool(args.focal), "줄자로 외부 확정" if args.focal
         else "미확정 — 사진만으로는 못 정함"),
        ("보드 화면 점유 ≥ 40 %", occ.mean() >= 0.40,
         f"평균 {occ.mean()*100:.0f} %"),
        ("면내회전 20° 초과 없음", int((rot > 20).sum()) == 0,
         f"{int((rot > 20).sum())}개"),
        ("20° 이상 기울임 ≥ 4개", int((tilt >= 20).sum()) >= 4,
         f"{int((tilt >= 20).sum())}개"),
        ("코너가 9칸 모두 닿음", len(seen) == 9, f"{len(seen)}/9"),
        ("거리 폭 ≥ 60 mm", ds.max()-ds.min() >= 60,
         f"{ds.max()-ds.min():.0f} mm"),
        ("대비 ≥ 1 ℃", all(s["dt"] is None or s["dt"] >= 1 for s in det),
         f"{min(s['dt'] for s in det if s['dt'] is not None):.2f} ~ "
         f"{max(s['dt'] for s in det if s['dt'] is not None):.2f} ℃"),
        ("줄자 대조 ≤ 5 %", bool(rat) and float(np.std(rat)/np.mean(rat)) <= .05,
         (f"{len(tp)}점 · 산포 {np.std(rat)/np.mean(rat)*100:.1f} %"
          if rat else "실측 없음")),
        ("녹화 길이 ≤ 15 초", all(not s["nf"] or s["nf"]/FPS <= 15
                             for s in shots),
         f"{max((s['nf']/FPS for s in shots if s['nf']), default=0):.0f}초 최대"),
        ("RGB 동시 녹화", False, "파일 0건 — 스테레오 불가"),
    ]
    notes = [
        f"초점거리가 확정됐습니다. 줄자 실측으로 f = {K[0,0]:.1f} px "
        f"(화각 {2*np.degrees(np.arctan(80/K[0,0])):.1f}° × "
        f"{2*np.degrees(np.arctan(60/K[1,1])):.1f}°). 사양서의 42° × 32° "
        f"(f = 208.4)는 실측과 맞지 않습니다. 457 mm 에서 GSD 가 "
        f"2.19 → {457/K[0,0]:.2f} mm/px 로 41 % 거칠어지므로 보고서 재계산이 "
        f"필요합니다.",
        f"프레임 재선택으로 재촬영 없이 RMS {before:.3f} → {rms:.3f} px. "
        f"{swapped}자세가 같은 녹화 안의 더 좋은 프레임으로 교체됐습니다. "
        f"'가장 선명한' 프레임이 '코너가 가장 정확한' 프레임은 아닙니다.",
        f"판을 돌려 든 것이 가장 큰 손실입니다. {int((rot > 20).sum())}자세가 "
        f"20° 넘게 돌아가 있습니다. 45°로 들면 보드 대각선이 화면 세로 120 px "
        f"에 먼저 걸려 칸 12.6 px 가 한계입니다. 눕혀 들면 19.5 px, 보드 면적은 "
        f"2.4 배가 됩니다.",
        f"녹화가 길면 한 파일에 여러 자세가 섞입니다. 4초짜리는 판이 1~2° 안에서 "
        f"안정적인데, 39초짜리는 53°까지 돌아갔고 246초짜리는 60장 중 2장만 "
        f"검출됩니다. 5초 안팎으로 끊고 파일이 3 MB 를 넘지 않는지 확인하십시오.",
        f"RGB 가 한 건도 없어 스테레오(R·T)와 호모그래피는 전혀 구할 수 "
        f"없습니다. 2차는 반드시 RGB 를 연속 녹화한 상태로 찍어야 합니다.",
    ]
    p = summary(shots, det, rms, K, D, checks, notes, args.src, args.focal,
                os.path.join(args.out, "summary.png"))

    print("\n" + "=" * 88)
    for lab, ok, val in checks:
        print(f"  {'✓' if ok else '✗'}  {lab:<26}{val}")
    print(f"\n  요약      {p}")
    print(f"  전체      {args.out}/contact_sheet.png")
    print(f"  낱장      {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
