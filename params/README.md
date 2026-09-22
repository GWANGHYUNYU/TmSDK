# params/ — 확정된 카메라 파라미터

코드에서 읽어 쓰는 값입니다. **산출 근거와 한계는
[최종 보고](../docs/calibration_final.md) 에 정리돼 있습니다.**

| 파일 | 내용 |
|---|---|
| `rgb_intrinsics.json` | RGB 내부 파라미터 + 왜곡 |
| `thermal_rgb_stereo.json` | 두 카메라 사이의 R, T |
| `layer_homography.json` | 캐노피 층별 RGB→열화상 호모그래피 |
| `rgb_intrinsics.npz` · `thermal_rgb_stereo.npz` | 같은 값의 numpy 판. 스크립트가 읽습니다 |

## 쓰기 전에 반드시

**① 열화상을 먼저 180° 돌리십시오.** 카메라가 거꾸로 장착돼 있습니다.

```python
th = cv2.rotate(th, cv2.ROTATE_180)
```

**② RGB 좌표는 왜곡을 먼저 푸십시오.** k1 = −0.37 은 무시할 크기가 아닙니다.

```python
und = cv2.undistortPoints(pts, K_rgb, dist_rgb, P=K_rgb)
th_pts = cv2.perspectiveTransform(und, H)      # 그 층의 H
```

**③ 층마다 다른 H 를 쓰십시오.** 하나로 묶으면 8~12 화소 어긋납니다.

## 값

| | |
|---|---|
| RGB f | **1217 px** (1920×1080, 화각 76.5° × 47.9°) |
| RGB 왜곡 | k1 −0.3665 · k2 +0.1223 |
| 열화상 f | **147.4 px** (160×120, 화각 57.0° × 44.3°) |
| **베이스라인 ‖T‖** | **53.3 mm** ✅ 확정 |
| 두 카메라 회전 | pitch 6.26° · yaw 1.43° · roll −0.19° |
| **정합 오차** | **중앙 0.93 px** (419쌍, 250~724 mm) |

### 믿을 수 있는 정도

| 값 | 상태 |
|---|---|
| RGB f = 1217 px | **확정.** RMS 최저점이 뚜렷함. 1 % 대역 1160~1270 |
| RGB 왜곡 | 반경 **660 px 안쪽만** 자료로 뒷받침. 바깥은 외삽 |
| 열화상 f = 147.4 px | **줄자 + RGB 대조**가 근거. 영상만으로는 결정 안 됨 |
| **베이스라인 53.3 mm** | ✅ **확정.** 녹화를 4분할해 8회 다시 풀어 52.69~54.03 mm |

**베이스라인이 확정된 근거** (2026-09-18 자료, 419쌍, 판 거리 250~724 mm)

| | |
|---|---|
| 녹화 단위 8회 재분할 | **52.69 ~ 54.03 mm** (폭 1.34 · 표준편차 0.46) |
| 캘리퍼스 실측 51.9 mm 와 차 | **+1.4 mm** (2.6 %) |
| **횡방향 성분** −49.56 mm vs 캘리퍼스 b = 50.0 mm | **0.9 % 일치** |
| 남겨 둔 녹화에서의 정합 오차 | **0.95 px** |

3차까지는 같은 시험에서 53.5~69.2 mm (폭 15.7 mm) 로 흔들렸습니다. 판 거리
폭이 96 mm 뿐이었기 때문입니다. 4차에서 **474 mm 폭**을 확보해 풀렸습니다.

> **3층 1000 mm 는 직접 검증하지 못했습니다** — 상단 캐노피가 체커보드를
> 가려 물리적으로 촬영이 안 됩니다. 다만 위험하지 않습니다: 베이스라인
> 불확실도 기여가 1000 mm 에서 **0.099 px** 로 가장 작고, 실측 정합 오차도
> 깊이에 따라 **줄어듭니다**(r = −0.284). 근거는
> [최종 보고 4장](../docs/calibration_final.md).

## 다시 만들려면

```bash
python3 scripts/calib_rgb.py calib/2026-08-28/rgb --out output/rgb_intrinsics
python3 scripts/verify_rgb.py output/rgb_intrinsics/rgb_intrinsics.npz \
    calib/2026-08-28 --max-dt 0.02 --max-blur 1.0 --per-rec 14 --cache
python3 scripts/layer_homography.py output/rgb_verify/stereo.npz \
    --depths 420 700 1000 --out output/layer_H
```
