# params/ — 확정된 카메라 파라미터

코드에서 읽어 쓰는 값입니다. 산출 근거는
[status.md 0-A · 0-E](../docs/status.md) 에 있습니다.

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
| 베이스라인 | **58.1 mm** ⚠ 아래 참조 |
| 정합 오차 | 중앙 **1.25 px** (91쌍, 382~478 mm) |

### ⚠ 믿을 수 있는 정도

| 값 | 상태 |
|---|---|
| RGB f = 1217 px | **확정.** RMS 최저점이 뚜렷함. 1 % 대역 1160~1270 |
| RGB 왜곡 | 반경 **660 px 안쪽만** 자료로 뒷받침. 바깥은 외삽 |
| 열화상 f = 147.4 px | **줄자 + RGB 대조**가 근거. 영상만으로는 결정 안 됨 |
| **베이스라인 58.1 mm** | **미확정.** 녹화를 바꾸면 53.5~69.2 mm |

베이스라인이 흔들리는 이유는 2026-08-28 의 판 거리 폭이 **96 mm**(382~478)
뿐이기 때문입니다. 깊이 폭이 넓은 2026-09-14 자료는 소등 뒤에 찍혀 RGB 가
겹치지 않습니다.

**4차 촬영에서 소등(18:01) 전에 깊이를 넓게 찍으면 그 자리에서 풀립니다**
([촬영 지침 5-6](../docs/thermal_rgb_calibration.md)).

## 다시 만들려면

```bash
python3 scripts/calib_rgb.py calib/2026-08-28/rgb --out output/rgb_intrinsics
python3 scripts/verify_rgb.py output/rgb_intrinsics/rgb_intrinsics.npz \
    calib/2026-08-28 --max-dt 0.02 --max-blur 1.0 --per-rec 14 --cache
python3 scripts/layer_homography.py output/rgb_verify/stereo.npz \
    --depths 420 700 1000 --out output/layer_H
```
