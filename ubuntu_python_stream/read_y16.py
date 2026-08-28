#################################################################
# File: read_y16.py
#
# multi_cameras_multi_roi_y16.py 가 저장한 .y16raw 를 읽어
# 온도 배열로 복원하거나 라벨링용 이미지로 내보낸다.
# TmSDK 나 카메라 없이 동작한다 (.y16meta 의 LUT 사용).
#
# 실행:
#   # 메타데이터와 온도 통계 확인
#   python3 read_y16.py info raw_output/192_168_0_151_20260806_080000.y16raw
#
#   # 온도 배열(float32, °C)로 저장 → (frames, H, W)
#   python3 read_y16.py npy raw_output/*.y16raw --out temps/
#
#   # 라벨링용 8-bit PNG 내보내기 (고정 온도 범위 필수)
#   python3 read_y16.py png raw_output/*.y16raw --out frames/ \
#           --tmin 15 --tmax 45 --stride 300
#
#   # 폴더 전체를 훑어 온도 범위를 먼저 조사
#   python3 read_y16.py range raw_output/
#
# ★ PNG 내보내기 시 --tmin/--tmax 를 반드시 고정하십시오.
#   프레임별 min-max 정규화(auto-gain)를 쓰면 프레임마다 밝기 매핑이
#   달라져 학습·라벨링에 쓸 수 없는 데이터가 됩니다. 기존 output/ 의
#   AVI 가 정확히 그 문제를 안고 있습니다.
#
# 요구사항:
#   pip3 install numpy
#   (png 서브커맨드만 opencv-python-headless 필요)
#################################################################

import argparse
import glob
import json
import os
import sys

import numpy as np


# ─────────────────────────────────────────────────────────────
# 로딩
# ─────────────────────────────────────────────────────────────
def meta_path_for(raw_path: str) -> str:
    return raw_path[:-len(".y16raw")] + ".y16meta"


def load_meta(raw_path: str) -> dict:
    mp = meta_path_for(raw_path)
    if not os.path.isfile(mp):
        raise FileNotFoundError(
            f"메타데이터가 없습니다: {mp}\n"
            "(.y16raw 와 .y16meta 는 항상 같은 폴더에 함께 두어야 합니다)")
    with open(mp, encoding="utf-8") as f:
        return json.load(f)


def load_raw(raw_path: str, meta: dict) -> np.ndarray:
    """(frames, H, W) uint16 배열로 메모리 맵핑하여 반환."""
    w, h = meta["width"], meta["height"]
    itemsize = 2
    total = os.path.getsize(raw_path)
    n_full = total // (w * h * itemsize)
    if n_full != meta["frame_count"]:
        print(f"  ⚠ 프레임 수 불일치: 파일 {n_full} vs 메타 {meta['frame_count']}"
              f" — 녹화가 비정상 종료된 파일일 수 있습니다. {n_full} 프레임으로 읽습니다.",
              file=sys.stderr)
    if n_full == 0:
        raise ValueError(f"프레임이 없습니다: {raw_path}")
    return np.memmap(raw_path, dtype="<u2", mode="r", shape=(n_full, h, w))


def temperature_converter(meta: dict):
    """raw(uint16) → 섭씨 변환 함수를 반환한다."""
    lut = meta.get("raw_to_temperature_lut")
    if not lut:
        raise ValueError(
            "메타데이터에 raw→온도 LUT 가 없습니다. "
            "이 파일은 카메라 없이는 온도로 복원할 수 없습니다.")
    xs = np.arange(lut["count"], dtype=np.float64) * lut["stride"] + lut["raw_start"]
    ys = np.asarray(lut["values"], dtype=np.float64)

    def to_celsius(raw: np.ndarray) -> np.ndarray:
        return np.interp(raw.astype(np.float64), xs, ys).astype(np.float32)

    return to_celsius


def frame_mean_celsius(arr, conv) -> np.ndarray:
    """프레임별 평균 온도 시계열. 메모리를 아끼려 청크 단위로 훑는다."""
    n, h, w = arr.shape
    out = np.empty(n, dtype=np.float64)
    chunk = max(1, 2_000_000 // (h * w))
    for s in range(0, n, chunk):
        block = np.asarray(arr[s:s + chunk])
        out[s:s + chunk] = block.mean(axis=(1, 2))
    return conv(out).astype(np.float64)


def timestamp_gaps(meta: dict, factor: float = 2.5) -> np.ndarray:
    """타임스탬프가 크게 벌어진 지점 '이후' 프레임 인덱스를 반환한다."""
    from datetime import datetime
    ts = meta.get("timestamps") or []
    if len(ts) < 3:
        return np.array([], dtype=int)
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    t = np.array([datetime.strptime(s, fmt).timestamp() for s in ts])
    dt = np.diff(t)
    expected = 1.0 / (float(meta.get("fps", 8.7)) or 8.7)
    return np.where(dt > expected * factor)[0] + 1


def detect_ffc(tmean: np.ndarray, meta: dict = None,
               threshold: float = 0.15) -> np.ndarray:
    """셔터(FFC) 이벤트 직후 프레임의 인덱스를 반환한다.

    비냉각 볼로미터는 주기적으로 셔터를 닫아 flat-field 보정을 하는데,
    이때 약 1.8초 프레임 전송이 멈추고 재개 직후 전역 평균이 계단형으로 튄다.
    재개 직후 1~2 프레임은 상단 행이 손상되는 경우가 있어 제외 대상이다.

    두 신호를 합집합으로 쓴다.
      · 타임스탬프 공백 — 셔터 블랙아웃. 가장 확실하다.
      · 전역 평균 점프 — 계단이 작으면 놓칠 수 있어 단독으로는 불충분하다.
    실제로 온도 점프만 쓰면 계단이 임계 미만인 이벤트를 놓쳐 주기 추정이
    2배로 튀는 일이 생긴다.
    """
    idx = set()
    if meta:
        idx.update(int(i) for i in timestamp_gaps(meta))
    if tmean is not None and tmean.size >= 3:
        d = np.abs(np.diff(tmean))
        thr = max(float(np.median(d)) * 12.0, threshold)
        idx.update(int(i) + 1 for i in np.where(d > thr)[0])
    return np.array(sorted(idx), dtype=int)


def ffc_period(ffc_idx: np.ndarray, meta: dict):
    """FFC 이벤트 간격에서 주기를 추정한다. (프레임, 초, 일관성) 반환.

    이벤트를 하나 놓치면 그 구간만 2배가 되므로, 평균이 아니라 중앙값을 쓰고
    각 간격이 중앙값의 정수배인지 확인해 일관성을 판정한다.
    """
    if len(ffc_idx) < 2:
        return None, None, None
    diffs = np.diff(ffc_idx).astype(float)
    base = float(np.median(diffs))
    fps = float(meta.get("fps", 8.7)) or 8.7
    # 각 간격이 base 의 정수배에서 얼마나 벗어나는지
    ratio = diffs / base
    resid = np.abs(ratio - np.round(ratio))
    consistent = bool(np.all(resid < 0.1))
    return base, base / fps, consistent


def bad_frame_mask(n: int, ffc_idx: np.ndarray, skip: int) -> np.ndarray:
    """FFC 직후 skip 프레임을 제외하는 불리언 마스크(True = 사용 가능)."""
    keep = np.ones(n, dtype=bool)
    for i in ffc_idx:
        keep[i:i + max(0, skip)] = False
    return keep


def expand_inputs(paths) -> list:
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.y16raw"))))
        elif any(ch in p for ch in "*?["):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    if not out:
        raise SystemExit("입력 파일을 찾지 못했습니다.")
    return out


# ─────────────────────────────────────────────────────────────
# 서브커맨드
# ─────────────────────────────────────────────────────────────
def cmd_info(args):
    for path in expand_inputs(args.inputs):
        meta = load_meta(path)
        arr  = load_raw(path, meta)
        conv = temperature_converter(meta)
        n, h, w = arr.shape

        # 전 프레임을 다 읽으면 느리므로 최대 200 프레임만 표본 추출
        idx = np.linspace(0, n - 1, min(n, 200)).astype(int)
        t = conv(np.asarray(arr[idx]))

        print(f"\n{os.path.basename(path)}")
        print(f"  카메라       : {meta.get('camera_label','?')} "
              f"({meta.get('camera_ip','?')})  포맷 {meta.get('camera_format','?')}")
        print(f"  해상도       : {w} x {h}   프레임 {n}   fps {meta.get('fps','?')}")
        print(f"  녹화 시작    : {meta.get('recording_started','?')}")
        if meta.get("timestamps"):
            print(f"  타임스탬프   : {meta['timestamps'][0]}  ~  "
                  f"{meta['timestamps'][-1]}")
        print(f"  파일 크기    : {os.path.getsize(path)/1024**2:.1f} MB")
        print(f"  raw 값 범위  : {meta.get('raw_value_min')} ~ "
              f"{meta.get('raw_value_max')}")
        print(f"  온도(표본)   : {t.min():.2f} ~ {t.max():.2f} °C  "
              f"(평균 {t.mean():.2f})")
        rois = meta.get("rois", [])
        print(f"  ROI          : {len(rois)}개")
        for i, r in enumerate(rois):
            print(f"     ROI{i} {r['type']:<8} {r['geometry']:<20} "
                  f"≈{r['approx_pixels']}px")


def cmd_range(args):
    """여러 파일의 전체 온도 범위를 조사한다 (PNG 고정 범위 정할 때 사용)."""
    lo, hi = None, None
    per_file = []
    for path in expand_inputs(args.inputs):
        meta = load_meta(path)
        arr  = load_raw(path, meta)
        conv = temperature_converter(meta)
        n = arr.shape[0]
        idx = np.linspace(0, n - 1, min(n, 100)).astype(int)
        t = conv(np.asarray(arr[idx]))
        flo, fhi = float(t.min()), float(t.max())
        per_file.append((os.path.basename(path), flo, fhi))
        lo = flo if lo is None else min(lo, flo)
        hi = fhi if hi is None else max(hi, fhi)

    for name, a, b in per_file:
        print(f"  {name}: {a:6.2f} ~ {b:6.2f} °C")
    pad_lo = np.floor(lo - 1)
    pad_hi = np.ceil(hi + 1)
    print(f"\n  전체 범위: {lo:.2f} ~ {hi:.2f} °C")
    print(f"  권장 PNG 옵션: --tmin {pad_lo:.0f} --tmax {pad_hi:.0f}")
    print("  (이 값을 모든 파일에 동일하게 적용해야 프레임 간 밝기가 일관됩니다)")


def cmd_check(args):
    """데이터 품질 진단 — 프레임 간격, 이상 프레임, 셔터(FFC) 이벤트,
    고정 화소, 극단값의 발생 위치를 찾는다."""
    from datetime import datetime

    for path in expand_inputs(args.inputs):
        meta = load_meta(path)
        arr  = load_raw(path, meta)
        conv = temperature_converter(meta)
        n, h, w = arr.shape
        print(f"\n{'='*70}")
        print(f"  {os.path.basename(path)}   {w}x{h}  {n} frames")
        print(f"{'='*70}")

        # ── 1. 타임스탬프 간격 ──
        ts = meta.get("timestamps", [])
        if len(ts) >= 2:
            fmt = "%Y-%m-%d %H:%M:%S.%f"
            t = np.array([datetime.strptime(s, fmt).timestamp() for s in ts])
            dt = np.diff(t)
            expected = 1.0 / float(meta.get("fps", 8.7))
            gaps = np.where(dt > expected * 2.5)[0]
            print(f"\n  [프레임 간격]  중앙값 {np.median(dt)*1000:.1f} ms  "
                  f"(기대 {expected*1000:.1f} ms)   최대 {dt.max()*1000:.0f} ms")
            if len(gaps):
                print(f"    ⚠ 간격이 2.5배를 넘는 지점 {len(gaps)}회 "
                      f"— 총 {dt[gaps].sum():.1f}초 누락 추정")
                for i in gaps[:5]:
                    print(f"       frame {i}→{i+1}: {dt[i]*1000:.0f} ms  ({ts[i]})")
                if len(gaps) > 5:
                    print(f"       ... 외 {len(gaps)-5}회")
            else:
                print("    ✔ 끊긴 구간 없음")

        # ── 2. 프레임별 통계 (청크 단위로 훑기) ──
        fmin = np.empty(n, dtype=np.uint16)
        fmax = np.empty(n, dtype=np.uint16)
        fmean = np.empty(n, dtype=np.float64)
        chunk = max(1, 2_000_000 // (h * w))
        for s in range(0, n, chunk):
            block = np.asarray(arr[s:s + chunk])
            fmin[s:s + chunk]  = block.min(axis=(1, 2))
            fmax[s:s + chunk]  = block.max(axis=(1, 2))
            fmean[s:s + chunk] = block.mean(axis=(1, 2))

        gmin_f0 = int(fmin.argmin())
        gmax_f0 = int(fmax.argmax())

        print(f"\n  [프레임 평균 raw]  {fmean.min():.0f} ~ {fmean.max():.0f}  "
              f"(전체 평균 {fmean.mean():.0f})")

        # 급격한 전역 점프 = 셔터(FFC).
        # raw 단위로 임계를 잡으면 카메라별 raw 배율에 따라 민감도가 달라지므로
        # 온도(°C)로 환산해 판정한다.
        tmean = conv(fmean).astype(np.float64)
        ffc = detect_ffc(tmean, meta, args.ffc_threshold)
        if len(ffc):
            steps = np.array([tmean[i] - tmean[i - 1] for i in ffc])
            pf, ps, consistent = ffc_period(ffc, meta)
            print(f"\n  [셔터(FFC)]  {len(ffc)}회 검출  "
                  f"계단 {steps.min():+.2f} ~ {steps.max():+.2f}°C "
                  f"(평균 {steps.mean():+.2f})")
            if ps:
                mark = "일정함" if consistent else "⚠ 불규칙 — 일부 이벤트를 놓쳤을 수 있음"
                print(f"    주기 {ps:.1f}초 ({ps/60:.2f}분, {pf:.0f} 프레임)  — {mark}")
            for i in ffc[:6]:
                print(f"       frame {i}: {tmean[i]-tmean[i-1]:+.2f}°C"
                      + (f"   {ts[i]}" if i < len(ts) else ""))
            if len(ffc) > 6:
                print(f"       ... 외 {len(ffc)-6}회")

            same = set(int(x) for x in ffc)
            hit = [f for f in (gmin_f0, gmax_f0) if f in same or f - 1 in same]
            if hit:
                print("    ⚠ 극단값이 FFC 직후 프레임에서 발생했습니다 "
                      "— 실제 물체가 아니라 셔터 재개 직후의 프레임 손상입니다.")
            if ps and abs(steps.mean()) > 0.1:
                print(f"    → 주기 {ps/60:.2f}분, 진폭 약 ±{abs(steps.mean())/2:.2f}°C 의 "
                      "톱니파로 볼 수 있습니다.")
                print(f"      · 녹화 슬롯 길이를 {ps/60:.2f}분의 정수배"
                      f"(예 {ps/60:.0f}분, {ps/60*2:.0f}분)로 잡으면 슬롯 평균에서 상쇄됩니다.")
                print("      · 내보내기 시 --skip-after-ffc 로 손상 프레임을 제외하세요.")
                print("      · 근본 해결은 화면 내 참조표면(온도 기지)으로 매 프레임 보정하는 것입니다.")
        else:
            print(f"\n  [셔터(FFC)]  ✔ {args.ffc_threshold:.2f}°C 를 넘는 전역 점프 없음")

        # ── 3. 극단값 발생 위치 ──
        for label, fi, take in [("최저", gmin_f0, "min"), ("최고", gmax_f0, "max")]:
            fr = np.asarray(arr[fi])
            idx = np.unravel_index(fr.argmin() if take == "min" else fr.argmax(),
                                   fr.shape)
            raw_v = int(fr[idx])
            cel = float(conv(np.array([raw_v]))[0])
            print(f"\n  [{label} raw]  {raw_v}  =  {cel:.2f}°C   "
                  f"frame {fi}  화소 (x={idx[1]}, y={idx[0]})")
            if ts and fi < len(ts):
                print(f"       시각 {ts[fi]}")
            # 그 화소가 상시 극단인지, 그 순간만인지
            col = np.asarray(arr[:, idx[0], idx[1]])
            print(f"       이 화소의 전체 이력: raw {col.min()}~{col.max()} "
                  f"(중앙 {int(np.median(col))})")
            if abs(int(np.median(col)) - raw_v) > (col.max() - col.min()) * 0.4:
                print("       → 일시적 이벤트로 보입니다 (실제 고온/저온 물체 또는 노이즈)")
            else:
                print("       → 이 화소가 상시 극단값입니다 (불량 화소 의심)")

        # ── 4. 고정(stuck) 화소 ──
        step = max(1, n // 200)
        sample = np.asarray(arr[::step]).astype(np.int32)
        pix_range = sample.max(axis=0) - sample.min(axis=0)
        stuck = np.argwhere(pix_range == 0)
        typical = float(np.median(pix_range))
        print(f"\n  [화소 변동폭]  중앙값 {typical:.0f} raw "
              f"({float(conv(np.array([typical+fmean.mean()]))[0] - conv(np.array([fmean.mean()]))[0]):.3f}°C 상당)")
        if len(stuck):
            print(f"    ⚠ 값이 전혀 변하지 않는 화소 {len(stuck)}개 (불량 화소 의심)")
            for y, x in stuck[:8]:
                print(f"       (x={x}, y={y})  값 {int(sample[0, y, x])}")
        else:
            print("    ✔ 고정 화소 없음")

        # 이상하게 변동이 큰 화소 (노이즈 화소)
        hot = np.argwhere(pix_range > typical * 8)
        if len(hot):
            print(f"    변동폭이 중앙값의 8배를 넘는 화소 {len(hot)}개 "
                  f"— 실제 변화(잎/기기)일 수도, 노이즈 화소일 수도 있습니다")
            for y, x in hot[:5]:
                print(f"       (x={x}, y={y})  변동폭 {int(pix_range[y, x])} raw")

        # ── 4-b. 상시 따뜻한/차가운 영역 (설비·센서 등 고정 물체 탐지) ──
        # 화소별 '중앙값' 을 쓰므로 일시적 스파이크(FFC 손상 등)에는 반응하지 않고,
        # 항상 그 자리에 있는 물체만 잡힌다.
        med_img = conv(np.median(sample, axis=0).astype(np.float64))
        canopy = float(np.median(med_img))
        dev = med_img - canopy
        warm = np.argwhere(dev > args.hotspot_delta)
        cold = np.argwhere(dev < -args.hotspot_delta)
        print(f"\n  [고정 물체 탐지]  캐노피 중앙값 {canopy:.2f}°C 기준 "
              f"±{args.hotspot_delta:.1f}°C 이탈 화소")
        if len(warm):
            print(f"    따뜻한 영역 {len(warm)}화소 "
                  f"(최대 +{dev.max():.2f}°C @ x={int(np.argwhere(dev==dev.max())[0][1])},"
                  f" y={int(np.argwhere(dev==dev.max())[0][0])})")
            ys, xs = warm[:, 0], warm[:, 1]
            print(f"      범위 x {xs.min()}~{xs.max()}, y {ys.min()}~{ys.max()}")
            print("      → 상시 존재하는 물체입니다 (일사센서·구동기·조명 등).")
            print("        마스크에서 제외하거나, 참조표면이라면 좌표를 기록해 두세요.")
        else:
            print("    ✔ 상시 따뜻한 고정 물체 없음")
        if len(cold):
            ys, xs = cold[:, 0], cold[:, 1]
            print(f"    차가운 영역 {len(cold)}화소  "
                  f"범위 x {xs.min()}~{xs.max()}, y {ys.min()}~{ys.max()} "
                  f"(최저 {dev.min():.2f}°C)")

        # ── 5. 온도 분포 요약 ──
        idx_s = np.linspace(0, n - 1, min(n, 200)).astype(int)
        t_all = conv(np.asarray(arr[idx_s]))
        qs = np.percentile(t_all, [0.1, 1, 50, 99, 99.9])
        print(f"\n  [온도 분포]  p0.1 {qs[0]:.2f}  p1 {qs[1]:.2f}  "
              f"중앙 {qs[2]:.2f}  p99 {qs[3]:.2f}  p99.9 {qs[4]:.2f} °C")
        print(f"    라벨링용 PNG 권장 범위: --tmin {np.floor(qs[1]-0.5):.0f} "
              f"--tmax {np.ceil(qs[3]+0.5):.0f}   "
              f"(극단값은 잘라내고 캐노피 대비를 살리는 범위)")


def orient(img: np.ndarray, rotate180: bool) -> np.ndarray:
    """RGB 카메라와 방향을 맞춘다.

    열화상 카메라가 RGB 대비 거꾸로 장착되어 있어 180° 회전 관계다
    (2026-08-11 확인). 기본 파이프라인은 as-recorded 좌표계를 쓰므로
    RGB 와 겹쳐 보거나 정합할 때만 이 변환을 적용한다.
        x' = W-1-x,  y' = H-1-y
    """
    return img[..., ::-1, ::-1] if rotate180 else img


def usable_frames(arr, meta, conv, stride: int, skip_after_ffc: int):
    """stride 로 뽑되 FFC 직후 손상 프레임을 제외한 프레임 인덱스."""
    n = arr.shape[0]
    sel = np.arange(0, n, max(1, stride))
    if skip_after_ffc <= 0:
        return sel, 0
    ffc = detect_ffc(frame_mean_celsius(arr, conv), meta)
    if not len(ffc):
        return sel, 0
    keep = bad_frame_mask(n, ffc, skip_after_ffc)
    kept = sel[keep[sel]]
    return kept, len(sel) - len(kept)


def cmd_npy(args):
    os.makedirs(args.out, exist_ok=True)
    for path in expand_inputs(args.inputs):
        meta = load_meta(path)
        arr  = load_raw(path, meta)
        conv = temperature_converter(meta)
        sel, dropped = usable_frames(arr, meta, conv,
                                     args.stride, args.skip_after_ffc)
        if not len(sel):
            print(f"  ⚠ {os.path.basename(path)}: 사용 가능한 프레임이 없습니다")
            continue
        temps = orient(conv(np.asarray(arr[sel])), args.rotate180)
        dst = os.path.join(
            args.out,
            os.path.basename(path)[:-len(".y16raw")] + "_celsius.npy")
        np.save(dst, temps)
        # 어떤 원본 프레임을 담았는지 함께 저장해야 타임스탬프와 대조할 수 있다
        np.save(dst[:-4] + "_frameidx.npy", sel)
        print(f"  {dst}  shape={temps.shape} dtype={temps.dtype} "
              f"({temps.nbytes/1024**2:.1f} MB)"
              + (f"   FFC 손상 {dropped}장 제외" if dropped else ""))


def cmd_png(args):
    try:
        import cv2
    except ImportError:
        raise SystemExit("png 내보내기에는 opencv 가 필요합니다: "
                         "pip3 install opencv-python-headless")

    if args.tmin is None or args.tmax is None:
        raise SystemExit(
            "--tmin 과 --tmax 를 반드시 지정하세요.\n"
            "먼저 'read_y16.py range <폴더>' 로 적절한 범위를 확인하십시오.\n"
            "프레임별 자동 정규화는 학습·라벨링 데이터를 망칩니다.")
    if args.tmax <= args.tmin:
        raise SystemExit("--tmax 는 --tmin 보다 커야 합니다.")

    os.makedirs(args.out, exist_ok=True)
    span = args.tmax - args.tmin
    total = 0
    for path in expand_inputs(args.inputs):
        meta = load_meta(path)
        arr  = load_raw(path, meta)
        conv = temperature_converter(meta)
        stem = os.path.basename(path)[:-len(".y16raw")]
        sel, dropped = usable_frames(arr, meta, conv,
                                     args.stride, args.skip_after_ffc)
        clipped_any = False

        for i in sel:
            t = orient(conv(np.asarray(arr[i])), args.rotate180)
            if t.min() < args.tmin or t.max() > args.tmax:
                clipped_any = True
            g = np.clip((t - args.tmin) / span, 0.0, 1.0)
            img = (g * 255).astype(np.uint8)
            if args.colormap:
                img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
            cv2.imwrite(os.path.join(args.out, f"{stem}_f{i:06d}.png"), img)
            total += 1

        if clipped_any:
            print(f"  ⚠ {stem}: --tmin/--tmax 범위를 벗어난 화소가 잘렸습니다")
        print(f"  {stem}: {len(sel)}장 저장"
              + (f"   FFC 손상 {dropped}장 제외" if dropped else ""))

    print(f"\n  총 {total}장  →  {args.out}")
    print(f"  고정 범위 {args.tmin}~{args.tmax}°C "
          f"(1 계조 = {span/255:.4f}°C)")
    print("  ※ 이 범위를 기록해 두고, 이후 모든 내보내기에 동일하게 쓰십시오.")


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=".y16raw 열화상 원본을 온도로 복원하거나 이미지로 내보낸다")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="메타데이터와 온도 통계 출력")
    p.add_argument("inputs", nargs="+", help=".y16raw 파일 / 글롭 / 폴더")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("range", help="여러 파일의 온도 범위 조사 (PNG 범위 결정용)")
    p.add_argument("inputs", nargs="+")
    p.set_defaults(func=cmd_range)

    p = sub.add_parser(
        "check",
        help="데이터 품질 진단 (끊긴 구간·셔터 이벤트·불량 화소·극단값 위치)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--ffc-threshold", type=float, default=0.15,
                   help="셔터 의심으로 볼 전역 평균 점프 하한 (°C, 기본 0.15)")
    p.add_argument("--hotspot-delta", type=float, default=2.0,
                   help="고정 물체로 볼 캐노피 중앙값 대비 편차 (°C, 기본 2.0)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("npy", help="섭씨 float32 배열(.npy)로 내보내기")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="temps", help="출력 폴더")
    p.add_argument("--stride", type=int, default=1, help="N 프레임마다 1장")
    p.add_argument("--skip-after-ffc", type=int, default=2,
                   help="셔터(FFC) 직후 손상 프레임 N장 제외 (기본 2, 0=제외 안 함)")
    p.add_argument("--rotate180", action="store_true",
                   help="RGB 카메라와 방향 맞추기 (열화상이 180° 회전 장착됨). "
                        "정합·중첩용. 기본 파이프라인은 as-recorded 좌표계를 씀")
    p.set_defaults(func=cmd_npy)

    p = sub.add_parser("png", help="라벨링용 8-bit PNG 내보내기 (고정 온도 범위)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="frames", help="출력 폴더")
    p.add_argument("--stride", type=int, default=1, help="N 프레임마다 1장")
    p.add_argument("--skip-after-ffc", type=int, default=2,
                   help="셔터(FFC) 직후 손상 프레임 N장 제외 (기본 2, 0=제외 안 함)")
    p.add_argument("--rotate180", action="store_true",
                   help="RGB 카메라와 방향 맞추기 (열화상이 180° 회전 장착됨). "
                        "정합·중첩용. 기본 파이프라인은 as-recorded 좌표계를 씀")
    p.add_argument("--tmin", type=float, help="정규화 하한 (°C, 필수)")
    p.add_argument("--tmax", type=float, help="정규화 상한 (°C, 필수)")
    p.add_argument("--colormap", action="store_true",
                   help="INFERNO 의사색 적용 (사람이 볼 때만. 학습 입력은 회색조 권장)")
    p.set_defaults(func=cmd_png)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
