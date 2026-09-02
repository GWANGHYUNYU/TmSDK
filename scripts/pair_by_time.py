#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""절대 시각으로 열화상 ↔ RGB 짝짓기 (원본 RGB 용).

    python3 scripts/pair_by_time.py calib/_all calib/rgb --focal 147.4

원본 RGB 는 파일명이 시작 시각(HH-MM-SS.mp4)이고 열화상도 파일명에 시각이
들어 있어, 두 영상을 절대 시각으로 겹칠 수 있습니다. 추측이 필요 없습니다.

열화상 녹화 한 건 안에서도 판이 움직인 경우(긴 녹화)에는 서로 다른 자세를
여러 개 뽑아 각각 짝지어 자세 수를 늘립니다.

각 열화상 프레임의 시각 창 안에서 RGB 를 모두 훑어, 기하가 가장 잘 맞는
프레임을 고릅니다. fps 오차나 시계 오프셋이 있어도 창 안에서 흡수됩니다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

import check_board as CB                                       # noqa: E402
from check_board import detect, inplane_deg, to8, short        # noqa: E402
from calibrate_thermal import object_points, CALIB_FLAGS, gather  # noqa: E402


def th_start(name):
    """192_168_0_151_20260828_162609 → datetime"""
    m = re.search(r"(\d{8})_(\d{6})", name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")


def rgb_start(name):
    """16-24-53.mp4 → time"""
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})", os.path.basename(name))
    if not m:
        return None
    return dt.time(*(int(v) for v in m.groups()))


def load_th_frames(path):
    """y16raw → (배열, 변환함수, 프레임별 절대초 또는 None).

    ★ .y16meta 에 프레임별 타임스탬프(밀리초)가 들어 있다. 이것을 쓰면
      '시작 + idx/fps' 추정이 필요 없다. 추정은 프레임 드롭이 있는 녹화에서
      최대 1.6 초까지 어긋났고(161425, 실측 fps 8.17), 그만큼 짝이 틀어졌다.
      실측 fps 는 8.775 로 문서의 8.7 과도 다르다.
    """
    import json
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    secs = None
    mp = os.path.splitext(path)[0] + ".y16meta"
    try:
        ts = json.load(open(mp, encoding="utf-8")).get("timestamps") or []
        if ts:
            secs = []
            for s in ts:
                t = dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
                secs.append(t.hour*3600 + t.minute*60 + t.second
                            + t.microsecond/1e6)
            secs = np.array(secs)
    except Exception:
        secs = None
    return arr, conv, secs


def main():
    ap = argparse.ArgumentParser(description="절대 시각 기반 짝짓기")
    ap.add_argument("th_dir", nargs="+",
                    help="열화상 폴더들 (원본 파일명이어야 시각을 읽습니다)")
    ap.add_argument("rgb_dir")
    ap.add_argument("--pattern", default="7x4")
    ap.add_argument("--cell", type=float, default=30.0)
    ap.add_argument("--focal", type=float, default=147.4)
    ap.add_argument("--th-fps", type=float, default=8.7)
    ap.add_argument("--coarse-width", type=int, default=640,
                    help="탐색용 축소 폭. 원본 해상도 전수 검출은 너무 느리다")
    ap.add_argument("--frame-step", type=int, default=3,
                    help="창 안에서 몇 프레임마다 볼지")
    ap.add_argument("--window", type=float, default=2.5,
                    help="RGB 탐색 창 반경 초 (시계 오프셋 흡수)")
    ap.add_argument("--max-probe", type=int, default=120,
                    help="녹화 하나에서 훑어볼 최대 프레임 수")
    ap.add_argument("--min-move", type=float, default=10.0,
                    help="이만큼(도) 돌아가면 다른 자세로 보고 추가 추출")
    ap.add_argument("--out", default="output/pair_time")
    args = ap.parse_args()

    pat = tuple(int(v) for v in args.pattern.lower().split("x"))
    obj = object_points(pat, args.cell)
    os.makedirs(args.out, exist_ok=True)
    Kt = np.array([[args.focal, 0, 80.], [0, args.focal, 60.], [0, 0, 1.]])

    print("=" * 88)
    print("절대 시각 기반 열화상 ↔ RGB 짝짓기")
    print("=" * 88)

    # ── RGB 영상 목록 ──
    vids = []
    for f in sorted(os.listdir(args.rgb_dir)):
        if not f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        t0 = rgb_start(f)
        if t0 is None:
            print(f"  건너뜀 (시각 없는 파일명): {f}")
            continue
        p = os.path.join(args.rgb_dir, f)
        c = cv2.VideoCapture(p)
        fps = c.get(cv2.CAP_PROP_FPS)
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        c.release()
        s0 = t0.hour*3600 + t0.minute*60 + t0.second
        vids.append(dict(path=p, name=f, s0=s0, fps=fps, n=n, dur=n/fps,
                         size=(W, H)))
        print(f"  RGB  {f:<22}{W}x{H} {fps:.2f}fps  "
              f"{t0}  ~  {dt.timedelta(seconds=s0+n/fps)}")
    if not vids:
        raise SystemExit("시각이 든 RGB 파일명이 없습니다 (예: 16-24-53.mp4)")
    W, H = vids[0]["size"]

    # ── 열화상에서 자세 뽑기 ──
    print()
    files = {}
    for d in args.th_dir:
        for k, v in gather(d).items():
            files[k] = v
    th_items = []          # (이름, 절대초, 코너)
    for nm, p in files.items():
        # 원래 이름에서 시각을 얻으려면 _all 사본이 아니라 원본이 필요하다
        st = th_start(nm) or th_start(os.path.basename(p))
        if st is None:
            print(f"  {short(nm,14):<14}시각을 알 수 없어 건너뜁니다")
            continue
        try:
            arr, conv, secs = load_th_frames(p)
        except Exception as e:
            print(f"  {short(nm,14):<14}읽기 실패 {e}")
            continue
        base = st.hour*3600 + st.minute*60 + st.second
        nts = 0 if secs is None else len(secs)
        got = []
        # 초당 3장 정도, 다만 긴 녹화는 상한을 둔다 (246초짜리가 섞여 있다)
        step = max(1, int(round(args.th_fps/3)),
                   arr.shape[0]//args.max_probe + 1)
        for i in range(0, arr.shape[0], step):
            g = to8(conv(np.asarray(arr[i])).astype(np.float64))
            ok, c, _, _ = detect(g, [pat])
            if not ok:
                continue
            r = inplane_deg(c, pat)
            if got and abs(r-got[-1][2]) < args.min_move:
                continue                              # 같은 자세는 한 번만
            # 메타 타임스탬프가 있으면 그것을, 없으면 fps 추정을 쓴다
            tt = (float(secs[i]) if secs is not None and i < len(secs)
                  else base + i/args.th_fps)
            got.append((i, c.reshape(-1, 2), r, tt))
        for i, c, r, t in got:
            th_items.append((f"{nm}#f{i:04d}", t, c, r))
        src = (f"메타 {nts}개" if nts >= arr.shape[0]
               else (f"메타 {nts}개(부족)" if nts else "fps 추정"))
        print(f"  열화상 {short(nm,20):<20}{st.strftime('%H:%M:%S')}  "
              f"{arr.shape[0]:>5}프레임  시각 {src:<14}→  자세 {len(got)}개")

    if not th_items:
        raise SystemExit("열화상 자세를 하나도 못 뽑았습니다.")
    print(f"\n  열화상 자세 후보 {len(th_items)}개")

    # ── 각 자세의 시각 창에서 RGB 후보 모으기 ──
    print(f"\n  각 자세 ±{args.window:g}초 창에서 RGB 검출")
    pairs = []
    cache = {}
    for nm, t, cth, rot in th_items:
        v = next((x for x in vids if x["s0"] <= t <= x["s0"]+x["dur"]), None)
        if v is None:
            continue
        if v["path"] not in cache:
            cache[v["path"]] = cv2.VideoCapture(v["path"])
        cap = cache[v["path"]]
        lo = max(0, int((t-args.window-v["s0"])*v["fps"]))
        hi = min(v["n"]-1, int((t+args.window-v["s0"])*v["fps"]))
        # ★ 1920x1080 전수 검출은 프레임당 1초가 넘는다. 640 폭 축소본으로
        #   먼저 거르고, 통과한 프레임만 원본 해상도로 다시 잡는다.
        sc = args.coarse_width/v["size"][0]
        cands = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for fi in range(lo, hi+1):
            ok, fr = cap.read()                       # 순차 read 가 seek 보다 빠름
            if not ok:
                break
            if (fi-lo) % args.frame_step:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gs = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if not detect(gs, [pat], 1)[0]:
                continue
            ok2, c, _, _ = detect(g, [pat], 1)
            if ok2:
                cands.append((v["s0"]+fi/v["fps"], c.reshape(-1, 2)))
        if cands:
            pairs.append(dict(name=nm, t=t, th=cth, rot=rot,
                              cands=cands, vid=v["name"]))
        print(f"    {short(nm, 26):<26}{dt.timedelta(seconds=int(t))}  "
              f"RGB 후보 {len(cands):>3}개  [{v['name']}]")
    for c in cache.values():
        c.release()
    if not pairs:
        raise SystemExit("겹치는 구간이 없습니다.")

    np.savez(os.path.join(args.out, "candidates.npz"),
             names=np.array([p["name"] for p in pairs]),
             th=np.array([p["th"] for p in pairs]),
             tt=np.array([p["t"] for p in pairs]),
             ncand=np.array([len(p["cands"]) for p in pairs]),
             cand_t=np.array([np.array([c[0] for c in p["cands"]])
                              for p in pairs], dtype=object),
             cand_c=np.array([np.array([c[1] for c in p["cands"]])
                              for p in pairs], dtype=object),
             size=np.array([W, H]), Kt=Kt, allow_pickle=True)
    print(f"\n  후보 저장: {args.out}/candidates.npz  "
          f"({len(pairs)}자세 · RGB 후보 총 "
          f"{sum(len(p['cands']) for p in pairs)}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
