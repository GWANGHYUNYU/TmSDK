#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""탈락했던 녹화를 다시 살려 본다 — 코너 없이도 보드 대비를 잰다.

    python3 scripts/rescue.py <원본폴더>                 # 판정만
    python3 scripts/rescue.py <원본폴더> --apply          # 살릴 것을 calib/ 로

3차 촬영에서 65건이 「검출 0」으로 탈락했는데, 그 녹화들의 **보드 대비를
잰 적이 없습니다.** `contrast_c` 는 코너를 찾은 뒤에만 계산되기 때문입니다.
정작 알아야 할 때 못 재는 값이었습니다.

여기서는 두 가지를 합니다.

① **작은 격자로 보드를 찾아 대비를 잰다**
   7×4 가 안 잡혀도 5×4·6×3·4×3 같은 작은 격자는 잡히는 경우가 많습니다
   (3차 탈락 65건 중 53건). 그것으로 **보드 위치를 알아내고**, 거기서
   검정 칸과 금속 칸의 온도차를 직접 잽니다.

   이미 7×4 가 잡히는 52건으로 검증했습니다 — 같은 프레임을 7×4 로 잰
   값과 작은 격자로 잰 값의 **상관 r = 0.915, 절대차 중앙 0.23 ℃**.
   격자가 작아 표본이 적은 만큼 ±0.2 ℃ 정도로 보시면 됩니다.

   보드를 아예 못 찾으면 마지막 수단으로 체커 커널 응답
   (`checker_amplitude`)을 냅니다. 다만 이 값은 캐노피 무늬와 LED 바가
   섞여 들어가 **대비가 아닙니다** — 검증해 보니 순위상관이 0.38 에
   그쳤습니다. 「보드가 화면에 있었는가」의 참고용으로만 쓰십시오.

② **검출을 훨씬 집요하게 시도한다**
   정규화 3종 × 확대 3종 × 격자 3종. 기존 `detect` 는 p1~p99 정규화에
   원본 배율만 썼습니다.

판정
    살림(전체)   7×4 가 잡힘 — 그대로 쓸 수 있음
    살림(부분)   6×4 또는 7×3 이 2프레임 이상 — 자세·호모그래피에 사용 가능
    못 살림      어떤 조합으로도 안 잡힘 → 그때의 보드 대비를 수치로 남김
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ubuntu_python_stream"))

from check_board import detect, square_px, contrast_c      # noqa: E402

F_PX = 147.4
CELL = 30.0
FULL = (7, 4)
SUBS = [(6, 4), (7, 3)]                       # 캘리브레이션에 써도 되는 격자
TINY = [(5, 4), (6, 3), (5, 3), (4, 4), (4, 3)]   # 대비를 재려고 보드만 찾는 격자
MIN_SQ = 6.0
MIN_CONTRAST = 0.35     # 이 아래면 무늬가 온도잡음에 묻힌 것


# ── 정규화 3종 ────────────────────────────────────────────────────
def n_pct(img, lo=1, hi=99):
    a, b = np.percentile(img, [lo, hi])
    if b - a < 1e-6:
        return np.zeros(img.shape, np.uint8)
    return np.clip((img-a)/(b-a)*255, 0, 255).astype(np.uint8)


def n_tight(img):
    """위쪽을 더 잘라낸다 — LED 바가 계조를 다 먹는 것을 막는다."""
    return n_pct(img, 1, 95)


def n_clip(img, k=2.5):
    """중앙값 기준 이상치를 자른다."""
    m = np.median(img)
    s = np.median(np.abs(img-m))*1.4826 + 1e-6
    return n_pct(np.clip(img, m-k*s, m+k*s))


NORMS = (("p1-99", n_pct), ("p1-95", n_tight), ("이상치클립", n_clip))


# ── ① 코너 없이 재는 체커 패턴 진폭 ──────────────────────────────
def checker_kernel(s, ang):
    """한 칸이 s 화소인 2×2 체커 커널을 ang 만큼 돌려서 만든다."""
    n = int(round(s*2))
    if n < 4:
        return None
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n-1)/2
    a = np.radians(ang)
    u = (x-cx)*np.cos(a) + (y-cy)*np.sin(a)
    v = -(x-cx)*np.sin(a) + (y-cy)*np.cos(a)
    k = np.sign(np.sin(np.pi*u/s)) * np.sign(np.sin(np.pi*v/s))
    k -= k.mean()
    nrm = np.abs(k).sum()
    return k/nrm if nrm > 0 else None


_KCACHE = {}


def checker_amplitude(cels, sizes=(6, 8, 10, 13, 17), angles=(0, 22.5, 45)):
    """화면 안 체커 무늬의 최대 진폭 (℃). 코너 검출이 필요 없다.

    되돌림: (진폭, 그때의 칸 크기, 그때의 각도)
    """
    f = cels.astype(np.float32)
    best = (0.0, None, None)
    for s in sizes:
        for a in angles:
            key = (s, a)
            if key not in _KCACHE:
                _KCACHE[key] = checker_kernel(s, a)
            k = _KCACHE[key]
            if k is None or k.shape[0] >= min(f.shape):
                continue
            r = cv2.filter2D(f, cv2.CV_32F, k.astype(np.float32),
                             borderType=cv2.BORDER_REPLICATE)
            amp = float(np.percentile(np.abs(r), 99.8))*2
            if amp > best[0]:
                best = (amp, s, a)
    return best


# ── ② 집요한 검출 ────────────────────────────────────────────────
MAX_SPACING_RATIO = 2.5     # 이웃 코너 간격의 최대/최소. 투시로는 이보다 안 커진다


def plausible(c, pat):
    """잡힌 격자가 기하적으로 말이 되는가.

    ★ 이 가드가 없으면 안 됩니다. 확대(×2·×3)와 CLAHE 를 걸면 검출기가
      **캐노피 잎 무늬에서 가짜 격자를 만들어 냅니다.** 실제로 1000 mm
      계열에서 칸 20.7 px(=거리 213 mm)짜리 «검출»이 나왔는데, 그려 보니
      점들이 보드가 아니라 잎 위에 흩어져 있었고 간격 비가 4.0~36.4 였습니다.
      진짜 보드는 투시가 심해도 2.5 를 넘지 않습니다.
    """
    g = c.reshape(pat[1], pat[0], 2)
    d = np.concatenate([
        np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
        np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])
    if d.min() < 1.0:
        return False
    return bool(d.max()/d.min() < MAX_SPACING_RATIO)


def hard_detect(cels, pats, upscales=(1, 2, 3)):
    """정규화 × 확대 × 격자를 모두 시도 → (ok, corners, pat, 방법).

    기하적으로 말이 안 되는 격자는 버리고 계속 찾습니다.
    """
    for nm, fn in NORMS:
        g = fn(cels)
        for up in upscales:
            ok, c, pat, how = detect(g, pats, up)
            if ok and plausible(c, pat):
                return True, c, pat, f"{nm}·x{up}·{how}"
    return False, None, None, "-"


def measure_contrast(cels):
    """보드를 찾아 검정 칸 ↔ 금속 칸 온도차를 잰다.

    캘리브레이션에 못 쓸 만큼 작은 격자라도 **보드 위치만 알면 대비는
    잴 수 있습니다.** 7×4 로 잰 값과 r = 0.915 로 일치합니다.

    → (대비 ℃, 쓴 격자) 또는 (None, None)
    """
    for pats in ([FULL], SUBS, TINY):
        ok, c, pat, _ = hard_detect(cels, pats, upscales=(1, 2))
        if ok:
            ct = contrast_c(cels, c, pat)
            if ct is not None:
                return float(ct), pat
    return None, None


def scan(path, stride, per_rec=400):
    from read_y16 import load_meta, load_raw, temperature_converter
    meta = load_meta(path)
    arr = load_raw(path, meta)
    conv = temperature_converter(meta)
    n = min(arr.shape[0], per_rec)
    full, sub, cts, amps = [], [], [], []
    for i in range(0, n, stride):
        cels = conv(np.asarray(arr[i])).astype(np.float64)
        ok, c, pat, how = hard_detect(cels, [FULL])
        if ok:
            full.append((i, c, pat, how, cels))
            cts.append((float(contrast_c(cels, c, pat) or 0.0), pat))
            continue
        ok, c, pat, how = hard_detect(cels, SUBS)
        if ok:
            sub.append((i, c, pat, how, cels))
            cts.append((float(contrast_c(cels, c, pat) or 0.0), pat))
            continue
        ct, pt = measure_contrast(cels)
        if ct is not None:
            cts.append((ct, pt))
        else:
            amps.append(checker_amplitude(cels))
    best_ct = max(cts, key=lambda t: t[0]) if cts else (None, None)
    amp = max(amps, key=lambda t: t[0]) if amps else (0.0, None, None)
    return full, sub, best_ct, amp, arr.shape[0]


def judge(full, sub, best_ct, amp):
    ct_meas, ct_pat = best_ct
    ctxt = (f"대비 {ct_meas:.2f}℃" if ct_meas is not None else "대비 못 잼")
    if full:
        i, c, pat, how, cels = max(
            full, key=lambda t: contrast_c(t[4], t[1], t[2]) or 0)
        sq = square_px(c, pat)
        ct = contrast_c(cels, c, pat) or 0.0
        if sq < MIN_SQ:
            return "못 살림", f"칸 {sq:.1f} px < {MIN_SQ} — 너무 멀다 · {ctxt}", None
        # ★ 기하 검사만으로는 부족하다. 깨진 프레임의 모아레 무늬에서 7×4 가
        #   «잡히는» 일이 있었다 (180721, 대비 0.09 ℃). 검정 칸과 금속 칸이
        #   구별되지 않는다면 그것은 보드가 아니다.
        if ct < MIN_CONTRAST:
            return "못 살림", (f"7x4 가 잡혔으나 대비 {ct:.2f}℃ — 검정칸과 "
                             f"금속칸이 구별되지 않는다. 오검출로 봄"), None
        return "살림(전체)", f"7x4 · 칸 {sq:.1f} px · 대비 {ct:.2f}℃ · {how}", \
               dict(dist=round(F_PX*CELL/sq), sq=round(sq, 1),
                    ct=round(ct, 2), pat="7x4", how=how, nframe=len(full))
    if len(sub) >= 2:
        i, c, pat, how, cels = max(
            sub, key=lambda t: contrast_c(t[4], t[1], t[2]) or 0)
        sq = square_px(c, pat)
        ct = contrast_c(cels, c, pat) or 0.0
        # 여러 프레임의 칸 크기가 서로 맞아야 진짜다. 오검출은 흩어진다.
        sqs = np.array([square_px(t[1], t[2]) for t in sub])
        spread = float(np.ptp(sqs)/max(np.median(sqs), 1e-6))
        if spread > 0.35:
            return "못 살림", (f"부분격자가 프레임마다 칸 {sqs.min():.1f}~"
                             f"{sqs.max():.1f} px 로 흩어짐 — 오검출로 봄"), None
        if ct < MIN_CONTRAST:
            return "못 살림", (f"부분격자는 잡혔으나 대비 {ct:.2f}℃ — "
                             f"무늬가 잡음에 묻힘"), None
        if sq < MIN_SQ:
            return "못 살림", f"부분격자도 칸 {sq:.1f} px — 너무 멀다", None
        return "살림(부분)", (f"{pat[0]}x{pat[1]} · {len(sub)}프레임 · "
                            f"칸 {sq:.1f} px · 대비 {ct:.2f}℃"), \
               dict(dist=round(F_PX*CELL/sq), sq=round(sq, 1),
                    ct=round(ct, 2), pat=f"{pat[0]}x{pat[1]}", how=how,
                    nframe=len(sub))
    # 여기부터는 못 살림 — 다만 «왜» 를 수치로 남긴다
    if ct_meas is not None:
        p = f"{ct_pat[0]}x{ct_pat[1]}"
        if ct_meas < MIN_CONTRAST:
            return "못 살림", (f"보드는 찾음({p}) · 대비 {ct_meas:.2f}℃ — "
                             f"판이 안 데워짐"), None
        return "못 살림", (f"보드는 찾음({p}) · 대비 {ct_meas:.2f}℃ 는 있으나 "
                         f"격자가 안 잡힘 — 잘림·가림·흔들림"), None
    a, s, ang = amp
    return "못 살림", (f"보드를 못 찾음 · 체커 커널 응답 {a:.2f}(칸 ~{s}px) "
                      f"— 대비가 아님, 참고값"), None


def main():
    ap = argparse.ArgumentParser(description="탈락 녹화 구제")
    ap.add_argument("root", help="원본 녹화 폴더 (하위 폴더까지 훑습니다)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--date", default=None,
                    help="옮겨 넣을 calib 하위 날짜. 없으면 파일명에서 읽음")
    ap.add_argument("--apply", action="store_true",
                    help="살릴 것을 calib/<날짜>/thermal 로 복사")
    ap.add_argument("--out", default="output/rescue")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raws = []
    for dp, _, fns in os.walk(args.root):
        for f in sorted(fns):
            if f.endswith(".y16raw"):
                raws.append(os.path.join(dp, f))
    raws.sort()
    if not raws:
        raise SystemExit(f"{args.root} 에 .y16raw 가 없습니다.")

    # 이미 남아 있는 것은 건너뛴다 (시각으로 대조)
    have = set()
    idx = os.path.join(HERE, "calib", "INDEX.csv")
    if os.path.exists(idx):
        for r in csv.DictReader(open(idx, encoding="utf-8-sig")):
            m = re.search(r"(\d{8}_\d{6})", r["파일"])
            if m:
                have.add(m.group(1))

    print("=" * 96)
    print(f"탈락 녹화 구제 · {len(raws)}건  (이미 보유 {len(have)}건은 건너뜀)")
    print("=" * 96)
    rows, cur = [], None
    for p in raws:
        stamp = re.search(r"(\d{8}_\d{6})", os.path.basename(p))
        stamp = stamp.group(1) if stamp else None
        grp = os.path.basename(os.path.dirname(p))
        if grp != cur:
            cur = grp
            print(f"\n── {grp} " + "─"*max(0, 90-len(grp)))
        if stamp and stamp in have:
            continue
        try:
            full, sub, best_ct, amp, nfr = scan(p, args.stride)
        except Exception as e:
            print(f"  {os.path.basename(p)[-17:-7]}  읽기 실패 {e}")
            continue
        verdict, why, meta = judge(full, sub, best_ct, amp)
        mark = {"살림(전체)": "○", "살림(부분)": "◐", "못 살림": "×"}[verdict]
        ctv = best_ct[0]
        print(f"  {mark} {os.path.basename(p)[-17:-7]}  {nfr:>4}f  "
              f"{(f'{ctv:4.2f}C' if ctv is not None else '  -  ')}  {why}")
        rows.append(dict(path=os.path.relpath(p, HERE).replace("\\", "/"),
                         stamp=stamp, group=grp, verdict=verdict, why=why,
                         contrast_C=(round(best_ct[0], 3)
                                     if best_ct[0] is not None else None),
                         contrast_pat=(f"{best_ct[1][0]}x{best_ct[1][1]}"
                                       if best_ct[1] else None),
                         kernel_resp=round(amp[0], 3), frames=nfr,
                         **(meta or {})))

    ok = [r for r in rows if r["verdict"].startswith("살림")]
    print("\n" + "=" * 96)
    print(f"  살림 {len(ok)}건 "
          f"(전체격자 {sum(1 for r in ok if r['verdict']=='살림(전체)')} · "
          f"부분격자 {sum(1 for r in ok if r['verdict']=='살림(부분)')})"
          f" / 새로 본 {len(rows)}건")
    if ok:
        d = [r["dist"] for r in ok]
        c = [r["ct"] for r in ok]
        print(f"  거리 {min(d)} ~ {max(d)} mm · 대비 {min(c):.2f} ~ {max(c):.2f} ℃")
    dead = [r for r in rows if r["verdict"] == "못 살림"]
    if dead:
        m = [r["contrast_C"] for r in dead if r["contrast_C"] is not None]
        print(f"  못 살린 {len(dead)}건 중 보드를 찾아 대비를 잰 것 {len(m)}건")
        if m:
            a = np.array(m)
            print(f"    대비 {a.min():.2f} ~ {a.max():.2f} ℃ · 중앙 {np.median(a):.2f} ℃")
            print(f"    {MIN_CONTRAST} ℃ 미만(판이 안 데워짐) {int((a < MIN_CONTRAST).sum())}건 · "
                  f"1 ℃ 미만 {int((a < 1).sum())}건")
        if len(dead) > len(m):
            print(f"    보드를 아예 못 찾은 것 {len(dead)-len(m)}건")
    json.dump(rows, open(os.path.join(args.out, "rescue.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  저장 {args.out}/rescue.json")

    if args.apply and ok:
        n = 0
        for r in ok:
            ymd = r["stamp"][:8]
            date = args.date or f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
            dst = os.path.join(HERE, "calib", date, "thermal")
            os.makedirs(dst, exist_ok=True)
            nm = f"d{r['dist']:04d}_{r['stamp']}"
            base = os.path.splitext(os.path.join(HERE, r["path"]))[0]
            for e in (".y16raw", ".y16meta", ".avi", ".csv"):
                if os.path.exists(base+e):
                    shutil.copy2(base+e, os.path.join(dst, nm+e))
            n += 1
        print(f"  복사 {n}건 → calib/<날짜>/thermal/")
        print("  ⚠ calib/INDEX.csv 를 다시 만들려면 "
              "scripts/inventory.py 를 돌리십시오.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
