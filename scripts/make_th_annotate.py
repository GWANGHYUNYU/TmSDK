#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""열화상에 직접 라벨링하기 위한 패키지 — `.152` 처럼 짝 RGB 가 없는 카메라용.

    python3 scripts/make_th_annotate.py --cam 152 --out output/annotate_th152

`.151` 은 RGB 어노테이션을 H(d) 로 투영하면 되지만, `.152` 는 RGB 와의
스테레오가 없어 투영할 수 없습니다. 열화상 위에 직접 그리는 수밖에 없습니다.

내는 것
    frames/<날짜>_<시각>.png   그릴 대상. 화면평균 대비 편차 + ×6 확대
    ref/<...>_원본.png         원본 대비(1~99 %) — 참고용
    ref/<...>_밴드.png         따뜻한 가로 밴드 = 선반 추정 (층 힌트)
    index.json                 좌표를 되돌리는 데 필요한 값

★ 좌표 규약 — 나중에 되돌릴 수 있게 반드시 지킵니다.
    PNG 좌표 = (프로젝트 규약의 rotated 열화상 좌표) × upscale
    되돌리기:  열화상 화소 = PNG 좌표 / upscale
  도구의 «반전» 체크는 보기에만 쓰고, 저장은 PNG 원본 좌표로 됩니다.

★ 그릴 때
  - **점등 슬롯만** 씁니다. 소등 중에는 잎이 공기와 평형이라 그릴 것이 없습니다.
  - **분류는 «잎» 만** 찍습니다. 꽃·딸기는 열화상에서 안 갈립니다
    (꽃 66 % · 딸기 44 %, docs/segmentation_guide.md §0).
  - **층은 정하지 마십시오.** `.152` 는 층 대역을 확인할 근거가 없습니다.
    RGB 가 붙거나 `.152` 스테레오를 잡기 전까지는 층별 분석이 안 됩니다.
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


def _load(name, fn):
    s = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, "scripts", fn))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


LTS = _load("lts", "leaf_temp_series.py")
SG = _load("sg", "make_seg_guide.py")

TH_W, TH_H = 160, 120


def band_hint(cels, up):
    """따뜻한 가로 밴드를 표시합니다 — 선반·홈통일 가능성이 높은 자리.

    ★ «층 경계» 라고 단정하지 않습니다. `.151` 에서는 어노테이션으로 층을
      알고 있어서 검증할 수 있었지만, `.152` 는 맞춰 볼 정답이 없습니다.
      어디까지나 눈으로 층을 나눌 때 참고하라는 힌트입니다.
    """
    im = SG.th_render(cels, "dev")
    r = cels.mean(1) - cels.mean()
    pk = [y for y in range(2, TH_H-2)
          if r[y] > r[y-1] and r[y] >= r[y+1] and r[y] > 0.15]
    for y in pk:
        yy = int((y+0.5)*up)
        cv2.line(im, (0, yy), (im.shape[1], yy), (0, 0, 0), 3)
        cv2.line(im, (0, yy), (im.shape[1], yy), (40, 160, 255), 1)
        cv2.putText(im, f"y={y}", (6, yy-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (40, 160, 255), 1, cv2.LINE_AA)
    return im, pk


def main():
    ap = argparse.ArgumentParser(description="열화상 직접 라벨링 패키지")
    ap.add_argument("--cam", default="152")
    ap.add_argument("--days", default="20260919,20260921,20260923")
    ap.add_argument("--hours", default="10,12,14",
                    help="각 날짜에서 고를 시각(점등 중)")
    ap.add_argument("--upscale", type=int, default=6)
    ap.add_argument("--raw", default="calib/raw/th/raw_output")
    ap.add_argument("--out", default="output/annotate_th152")
    args = ap.parse_args()

    fr = os.path.join(args.out, "frames")
    rf = os.path.join(args.out, "ref")
    for d in (fr, rf):
        os.makedirs(d, exist_ok=True)
    SG.UP = args.upscale

    print("=" * 84)
    print(f"열화상 직접 라벨링 패키지 · 카메라 .{args.cam}")
    print("=" * 84)

    hours = [int(h) for h in args.hours.split(",")]
    frames, skipped = [], []
    for day in args.days.split(","):
        for hh in hours:
            p, cels = SG.find_slot(args.cam, day, [hh])
            if p is None:
                skipped.append(f"{day} {hh:02d}시")
                continue
            g = re.search(r"_(\d{2})(\d{2})\d{2}\.y16raw$", p)
            name = f"{day}_{g.group(1)}{g.group(2)}"
            SG.imwrite(os.path.join(fr, name + ".png"),
                       SG.th_render(cels, "dev"))
            SG.imwrite(os.path.join(rf, name + "_원본.png"),
                       SG.th_render(cels, "raw"))
            im, pk = band_hint(cels, args.upscale)
            SG.imwrite(os.path.join(rf, name + "_밴드.png"), im)
            frames.append(dict(name=name, source=os.path.basename(p),
                               day=day, time=f"{g.group(1)}:{g.group(2)}",
                               frame_mean_C=round(float(cels.mean()), 2),
                               span_C=round(float(np.ptp(cels)), 2),
                               warm_bands_y=pk))
            print(f"  {name}  화면평균 {cels.mean():.2f}℃  "
                  f"온도폭 {np.ptp(cels):.2f}℃  따뜻한 밴드 {len(pk)}개")
    if skipped:
        print(f"  (없는 슬롯 {len(skipped)}개: {', '.join(skipped)})")
    if not frames:
        raise SystemExit("쓸 수 있는 슬롯이 없습니다.")

    idx = dict(
        camera=f"192.168.0.{args.cam}",
        thermal_size=[TH_W, TH_H],
        upscale=args.upscale,
        png_size=[TH_W*args.upscale, TH_H*args.upscale],
        orientation="rotated_180 (프로젝트 규약과 동일)",
        coord_note=("PNG 좌표 / upscale = 열화상 화소 좌표. "
                    "도구의 «반전» 체크는 보기에만 영향을 줍니다."),
        render="화면평균 대비 편차 (파랑=차가움=잎, 빨강=따뜻함)",
        classes=["잎"],
        layer_note=("층은 정하지 마십시오. .152 는 층 대역을 확인할 근거가 "
                    "없습니다 — RGB 매칭이나 .152 스테레오가 필요합니다."),
        no_fov_polygon=("이 패키지에는 화각 경계가 없습니다. 열화상 자체가 "
                        "곧 화각이라 전체가 유효 영역입니다."),
        frames=frames,
    )
    json.dump(idx, open(os.path.join(args.out, "index.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    readme = f"""# `.152` 열화상 직접 라벨링

`frames/` 의 PNG {len(frames)}장을 [`tools/annotator.html`](../../tools/README.md)
에 끌어다 놓고 **잎**만 폴리곤으로 그리십시오.

## 왜 열화상에 직접 그리나

`.151` 은 RGB 어노테이션을 층별 H(d) 로 투영하면 됩니다. `.152` 는 RGB 와의
스테레오 캘리브레이션이 **없어서** 투영할 수 없습니다. 같은 시각 두 카메라의
상호상관이 **+0.077** 로, 서로 다른 구역을 봅니다.

## 그리는 법

| | |
|---|---|
| 볼 영상 | `frames/*.png` — **화면평균 대비 편차**. 파랑이 차가운 쪽(잎) |
| 참고 | `ref/*_원본.png` (1~99 % 대비), `ref/*_밴드.png` (선반 추정 위치) |
| 분류 | **«잎» 만** 찍으십시오 |
| 층 | **비워 두십시오** |

**분류를 «잎» 만 찍는 이유** — 열화상에서 꽃·딸기는 갈리지 않습니다.
꽃은 잎보다 +0.224 ℃(단일 임계 정확도 66 %), 딸기는 +0.064 ℃(44 %, 동전
던지기보다 낮음)이고, 1000 mm 층의 꽃은 화소 **3.7개**입니다.

**층을 비우는 이유** — `.152` 는 층 대역을 맞춰 볼 정답이 없습니다.
`ref/*_밴드.png` 의 가로선은 «따뜻한 행» 이고 선반·홈통일 가능성이 높지만,
검증된 값이 아닙니다. 층별 분석은 RGB 매칭이나 `.152` 스테레오가 생긴
뒤에 하십시오.

## 좌표 규약

```
PNG 좌표 / {args.upscale} = 열화상 화소 좌표 (160×120, 프로젝트 규약의 rotated)
```

도구의 «반전» 체크는 **보기에만** 영향을 줍니다. 저장은 항상 PNG 원본
좌표로 되므로, 편하게 켜고 쓰십시오.

## 내보낸 다음

JSON 을 주시면 `.152` 의 15일 · 약 700슬롯에 그대로 적용합니다.
지금 `.151` 에서 쓸 수 있는 것이 이틀뿐이라, 10배가 넘게 늘어납니다.

## 프레임 목록

| 이름 | 시각 | 화면평균 | 온도폭 |
|---|---|---|---|
"""
    for f in frames:
        readme += (f"| `{f['name']}.png` | {f['day'][4:6]}-{f['day'][6:]} "
                   f"{f['time']} | {f['frame_mean_C']} ℃ | "
                   f"{f['span_C']} ℃ |\n")
    open(os.path.join(args.out, "README.md"), "w",
         encoding="utf-8").write(readme)

    print(f"\n  저장  {args.out}/  (그릴 것 {len(frames)}장 + 참고 "
          f"{len(frames)*2}장 + index.json + README.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
