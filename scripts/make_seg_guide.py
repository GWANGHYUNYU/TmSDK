#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""분할 가이드용 대표 프레임과 «보이게 만든» 참고 영상을 냅니다.

    python3 scripts/make_seg_guide.py --out output/segguide

내는 것
    RGB   원본 · 반전(보기용) · 식생지수(ExG) · 꽃 후보 · 딸기 후보
    열화상 .151/.152 · 점등/소등 · 원본 대비 · **화면평균 대비 편차**

★ 열화상은 그냥 띄우면 잎이 안 보입니다. 화면 전체가 20~27 ℃ 안에 몰려
  있어서 대비를 1~99 % 로 펴도 공조 얼룩이 잎을 덮습니다. 「화면평균을
  뺀 영상」으로 봐야 잎 덩어리가 드러납니다 — 가이드가 이 영상을 쓰라고
  하는 이유입니다.

★ 규칙 기반 후보(ExG·HSV)는 **초안**입니다. 그대로 쓰라는 뜻이 아니라,
  손으로 고칠 출발점을 주는 것입니다. 정답은 사람이 찍습니다.
"""
from __future__ import annotations

import argparse
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

_spec = importlib.util.spec_from_file_location(
    "lts", os.path.join(HERE, "scripts", "leaf_temp_series.py"))
LTS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LTS)

TH_W, TH_H = 160, 120
UP = 6


def imwrite(path, img):
    """★ cv2.imwrite 는 Windows 에서 «한글 경로에 조용히 실패**합니다.**»
    반환값을 안 보면 0장 저장하고도 성공한 줄 압니다. imencode 로 씁니다."""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(path)


def label(img, text, sub=""):
    h = 58 if sub else 38
    out = np.zeros((img.shape[0]+h, img.shape[1], 3), np.uint8)
    out[h:] = img
    cv2.putText(out, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (170, 170, 170), 1, cv2.LINE_AA)
    return out


def grab(mp4, frame=15):
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    cap.release()
    if not ok:
        # ★ SystemExit 를 던지면 안 됩니다 — except Exception 을 빠져나가서,
        #   클립 하나가 깨졌을 뿐인데 훑기 전체가 멈춥니다.
        raise IOError(f"프레임을 못 읽었습니다: {mp4}")
    return im


def exg(bgr):
    """ExG = 2G − R − B. 조명 세기에 둔감해 실내 LED 아래서도 씁니다."""
    b, g, r = [x.astype(np.float32) for x in cv2.split(bgr)]
    s = b + g + r + 1e-6
    return 2*(g/s) - (r/s) - (b/s)


def flower_mask(bgr, k=38, amin=60, amax=9000):
    """꽃 = **주위보다** 밝고 채도 낮은 작은 덩어리.

    ★ 전역 임계(V>190 & S<60)로는 안 됩니다 — 이 현장의 RGB 는 김이 서려
      화면 전체가 뿌옇고, 그 임계로 17 % 가 «꽃» 이 됩니다. 꽃은 화면의
      1 % 미만입니다. 주위 평균 대비로 바꾸고 덩어리 크기로 거릅니다.
    """
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    loc = cv2.blur(v, (101, 101)).astype(np.int16)
    m = ((v.astype(np.int16) - loc > k) & (s < 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if amin <= st[i, cv2.CC_STAT_AREA] <= amax:
            out[lab == i] = 1
    return out


def fruit_mask(bgr):
    """열매 = 붉은 계열. 색상환이 0 에서 감기므로 양끝을 모두 봅니다."""
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    return (((h < 12) | (h > 168)) & (s > 90) & (v > 70)).astype(np.uint8)


def paint(bgr, m, col, a=0.45):
    out = bgr.copy()
    lay = np.zeros_like(bgr)
    lay[m > 0] = col
    return cv2.addWeighted(out, 1-a, lay, a, 0, out, -1) if False else \
        np.where(m[..., None] > 0, (out*(1-a)+np.array(col)*a).astype(np.uint8),
                 out)


def diverging_lut():
    """파랑(차가움) → 흰색(화면평균) → 빨강(따뜻함).

    ★ OpenCV 기본 맵은 전부 «순서형» 이라 0 이 어디인지 안 보입니다.
      COOL 로 그렸더니 분홍·청록만 요란하고 «평균과 같은 곳» 을 못 찾습니다.
      잎은 평균보다 «차가운» 쪽이므로, 0 을 흰색으로 두는 발산형이 맞습니다.
    """
    lut = np.zeros((256, 1, 3), np.uint8)
    for i in range(256):
        t = (i-127.5)/127.5                      # −1 … +1
        if t < 0:                                 # 파랑 쪽
            k = -t
            lut[i, 0] = (255, int(255*(1-k*0.75)), int(255*(1-k)))
        else:                                     # 빨강 쪽
            lut[i, 0] = (int(255*(1-t)), int(255*(1-t*0.75)), 255)
    return lut


def th_render(cels, mode="raw"):
    if mode == "dev":
        x = cels - cels.mean()
        # ★ 배율을 |편차| 의 98 분위로 잡으면, 조명 같은 «뜨거운» 물체가
        #   화면에 들어온 프레임에서 배율이 통째로 끌려가 잎이 창백해집니다.
        #   (.152 09-21 10:00 은 온도폭이 18.3 ℃ 였습니다.)
        #   우리가 그리려는 것은 «찬» 쪽이므로 찬 쪽으로 배율을 잡고,
        #   뜨거운 쪽은 빨강으로 포화시켜 버립니다.
        lim = max(abs(np.percentile(x, 2)), 0.3)
        u = np.clip((x+lim)/(2*lim)*255, 0, 255).astype(np.uint8)
        im = cv2.LUT(cv2.cvtColor(u, cv2.COLOR_GRAY2BGR), diverging_lut())
    else:
        lo, hi = np.percentile(cels, [1, 99])
        u = np.clip((cels-lo)/(hi-lo+1e-9)*255, 0, 255).astype(np.uint8)
        im = cv2.applyColorMap(u, cv2.COLORMAP_INFERNO)
    return cv2.resize(im, (TH_W*UP, TH_H*UP), interpolation=cv2.INTER_NEAREST)


def find_slot(cam, day, hours):
    """그 날, 주어진 «시간대» 안에서 실제로 읽히는 슬롯 → (경로, 온도맵).

    ★ 파일 크기만 보고 고르면 안 됩니다. `.y16raw` 는 멀쩡한데 짝인
      `.y16meta` 가 0 바이트인 녹화가 있어서, 크기 검사를 통과하고도
      읽기에서 터집니다. 그래서 «읽어 보고» 되는 것을 고릅니다.
    """
    for p in sorted(glob.glob(os.path.join(
            HERE, "calib/raw/th/raw_output",
            f"192_168_0_{cam}_{day}_*.y16raw"))):
        if os.path.getsize(p) < 1e6:
            continue
        # 자리수로 자르지 말 것 — 확장자가 .y16raw(7자)라 끝에서 센 인덱스가
        # 어긋나 «시» 대신 «초» 를 읽습니다. 처음에 그래서 점등 슬롯을
        # 하나도 못 찾았습니다.
        m = re.search(r"_(\d{2})(\d{2})(\d{2})\.y16raw$", p)
        if not (m and int(m.group(1)) in hours):
            continue
        try:
            return p, LTS.slot_mean(p)[0]
        except Exception:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser(description="분할 가이드용 대표 프레임")
    ap.add_argument("--rgb", default="calib/raw/rgb/26-09-22/12-00-00.mp4")
    ap.add_argument("--rgb-dir", default="calib/raw/rgb/26-09-22")
    ap.add_argument("--day", default="20260918")
    ap.add_argument("--out", default="output/segguide")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    O = lambda n: os.path.join(args.out, n)

    # ── RGB ──────────────────────────────────────────────────────────
    # ★ 아무 클립이나 쓰면 안 됩니다. 이 현장 RGB 는 08:25 이후 노출이
    #   무너져 화소의 16~20 % 가 포화합니다(하얗게 날아감). 포화율로 골라야
    #   합니다 — 사람이 파일명만 보고 고르면 «점심때 밝은 거» 를 집습니다.
    mp4 = args.rgb if os.path.exists(args.rgb) else None
    if mp4 is None:
        best = None
        for p in sorted(glob.glob(os.path.join(HERE, args.rgb_dir, "*.mp4"))):
            try:
                V = cv2.cvtColor(grab(p), cv2.COLOR_BGR2HSV)[:, :, 2]
            except Exception:
                continue
            sat, med = float((V >= 254).mean()), float(np.median(V))
            if med < 40:          # 소등 클립
                continue
            if best is None or sat < best[0]:
                best = (sat, p, med)
        if best is None:
            raise SystemExit("쓸 수 있는 RGB 클립이 없습니다.")
        mp4 = best[1]
        print(f"  RGB 자동 선택 — 포화 {100*best[0]:.1f} % · V중앙 "
              f"{best[2]:.0f} (같은 폴더에서 가장 덜 날아간 클립)")
    print(f"  RGB 대표: {os.path.relpath(mp4, HERE)}")
    raw = grab(mp4)
    view = cv2.rotate(raw, cv2.ROTATE_180)     # ★ RGB 가 뒤집힌 쪽입니다
    imwrite(O("rgb_01_원본_as_recorded.png"), raw)
    imwrite(O("rgb_02_반전_보기용.png"), view)

    e = exg(view)
    veg = ((e > 0.06)*255).astype(np.uint8)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    imwrite(O("rgb_03_식생ExG.png"),
                label(paint(view, veg, (60, 220, 60)), "ExG > 0.06  (잎 후보)",
                      "2G-R-B. 초안입니다 - 손으로 고칠 출발점"))
    fl = flower_mask(view)
    imwrite(O("rgb_04_꽃후보.png"),
                label(paint(view, fl, (255, 255, 255)),
                      "주위보다 밝고 채도 낮은 작은 덩어리  (꽃 후보)",
                      "흰 반사면도 같이 잡힙니다 - 반드시 손으로 거를 것"))
    fr = cv2.morphologyEx(fruit_mask(view), cv2.MORPH_OPEN,
                          np.ones((3, 3), np.uint8))
    imwrite(O("rgb_05_딸기후보.png"),
                label(paint(view, fr, (40, 40, 240)),
                      "붉은 계열  (딸기 후보)",
                      "익은 것만 잡힙니다. 흰 미숙과는 꽃 후보로 갑니다"))
    print(f"    식생 {100*veg.mean()/255:.1f} % · 꽃후보 {100*fl.mean():.2f} % "
          f"· 딸기후보 {100*fr.mean():.2f} %")

    # ── 열화상 ───────────────────────────────────────────────────────
    made, dev_imgs = [], []
    for cam in ("151", "152"):
        for tag, hours in (("점등", range(10, 17)), ("소등", range(0, 7))):
            p, cels = find_slot(cam, args.day, hours)
            if p is None:
                print(f"    .{cam} {tag} 슬롯 없음")
                continue
            g = re.search(r"_(\d{8})_(\d{2})(\d{2})\d{2}\.y16raw$", p)
            sub = (f"{g.group(1)} {g.group(2)}:{g.group(3)}  "
                   f"화면평균 {cels.mean():.2f}C")
            imwrite(O(f"th_{cam}_{tag}_1_원본대비.png"),
                        label(th_render(cels, "raw"),
                              f".{cam} {tag} - 1~99% 대비", sub))
            dv = label(th_render(cels, "dev"),
                       f".{cam} {tag} - 화면평균 대비 편차  <- 이걸 보고 그리세요",
                       "파랑=차가움(잎)  흰색=화면평균  빨강=따뜻함(배경/구조물)")
            imwrite(O(f"th_{cam}_{tag}_2_화면평균대비편차.png"), dv)
            dev_imgs.append((cam, tag, dv))
            made.append((cam, tag, float(cels.mean()), float(np.ptp(cels))))
    for cam, tag, mu, pk in made:
        print(f"    .{cam} {tag}  화면평균 {mu:.2f}℃  온도폭 {pk:.2f}℃")

    # ★ 점등/소등을 나란히 놓는 것이 이 가이드에서 제일 중요한 그림입니다.
    #   «열화상으로 잎을 그리려면 점등 중이어야 한다» 를 말로 설명하는 것보다
    #   두 장을 붙여 보여 주는 편이 빠릅니다. 소등 쪽은 잎이 아예 없습니다.
    for cam in ("151", "152"):
        pair = [d for d in dev_imgs if d[0] == cam]
        if len(pair) == 2:
            imwrite(O(f"th_{cam}_6_점등_소등_비교.png"),
                    np.hstack([p[2] for p in sorted(pair, key=lambda x: x[1])]))

    json.dump(dict(rgb=os.path.relpath(mp4, HERE), day=args.day,
                   note="분할 가이드용 대표 프레임", frames=[
                       f for f in sorted(os.listdir(args.out))
                       if f.endswith(".png")]),
              open(O("index.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  저장  {args.out}/  "
          f"({len([f for f in os.listdir(args.out) if f.endswith('.png')])}장)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
