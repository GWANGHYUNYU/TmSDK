# TmSDK

This project is a SDK for controlling TMC80/160/256/384 cameras.
After downloading the code, please refer to the **'Document/TmSDK Manual.pdf'** file.

---

## 이 저장소에서 추가된 것 — 수직농장 개체별 VPD 과제

> [ThermoEye/TmSDK](https://github.com/ThermoEye/TmSDK) 포크입니다. 아래는
> **수직농장 AI 기반 작물 개체별 VPD 모니터링 및 환경 제어 시스템** 과제를
> 위해 추가한 부분입니다. 원본 SDK 내용은 이 절 아래에 그대로 있습니다.

**처음 오셨다면 [`docs/README.md`](docs/README.md) 부터 읽으십시오.**
문서 폴더만으로 프로젝트 전체를 이해하고 작업을 이어갈 수 있게 구성했습니다.

### 폴더

| 경로 | 내용 |
|---|---|
| [`docs/`](docs/) | **프로젝트 문서 전부.** 현황·기획·데이터·캘리브레이션 |
| [`ubuntu_python_stream/`](ubuntu_python_stream/) | 녹화 프로그램과 캘리브레이션 도구 (Ubuntu 22.04) |
| [`scripts/`](scripts/) | 조사·정리·보정·검토 스크립트 |
| [**`params/`**](params/README.md) | **확정된 카메라 파라미터.** 코드는 여기서 읽습니다 |
| [`calib/`](calib/README.md) | 캘리브레이션 원본 (녹화는 저장소에 없음. 목록만) |
| `window_python_stream/` | Windows 배포용 |
| `output/`, `submit/` | 데이터·제출 서류 (저장소에 포함하지 않음) |

### 녹화

```bash
cd ubuntu_python_stream/
python3 multi_cameras_multi_roi_y16.py     # Y16 라디오메트릭 + 복수 ROI + 예약 녹화
python3 read_y16.py check raw_output/      # 품질 진단
```

> ⚠ **온도 분석이 목적이면 반드시 `_y16` 판을 쓰십시오.** AVI 는 SDK 가
> 프레임마다 auto-gain 을 적용해 절대온도를 복원할 수 없습니다.

### 열화상 ↔ RGB 정합 캘리브레이션

방사율 체커보드(무광 검정 시트지 + 바이브레이션 마감 금속) 기반입니다.
확정된 파라미터는 [`params/`](params/README.md), 절차는
[`docs/thermal_rgb_calibration.md`](docs/thermal_rgb_calibration.md),
**다음 촬영 작업지시서**는
[`docs/confluence/열화상_캘리브레이션_4차_촬영표.md`](docs/confluence/열화상_캘리브레이션_4차_촬영표.md).

```bash
# 현장 — 녹화가 생길 때마다 즉시 합격 판정 + 남은 구간 표시
python3 scripts/live_check.py <녹화폴더>
python3 scripts/live_check.py <녹화폴더> --cell 50 --pattern 5x4   # 먼 거리용 보드

# 사무실 — 전수 조사 후 쓸 것만 남기고 날짜별로 정리
python3 scripts/inventory.py calib/2026-09-14/thermal
python3 scripts/consolidate_calib.py --apply

# 사무실 — RGB 내부 파라미터 → 스테레오 → 층별 호모그래피
python3 scripts/calib_rgb.py calib/2026-08-28/rgb --out output/rgb_intrinsics
python3 scripts/verify_rgb.py output/rgb_intrinsics/rgb_intrinsics.npz \
    calib/2026-08-28 --max-dt 0.02 --max-blur 1.0 --cache
python3 scripts/layer_homography.py output/rgb_verify/stereo.npz --depths 420 700 1000
```

### ★ 사양서와 실측이 다릅니다

| 열화상 TMC160F | 사양서 | **실측 (2026-08-28)** |
|---|---|---|
| 초점거리 | 208.4 px | **147.4 px** |
| 화각 | 42° × 32° | **57.0° × 44.3°** |
| 카메라 상수 | 4.75 mm/px/m | **6.78 mm/px/m** |

물리 초점거리 **2.52 mm 는 원래 맞았습니다.** 화소 피치가 17.1 µm 인데 사양서의
42° 는 12.1 µm 를 전제로 한 값입니다. **거리·GSD 를 다루는 모든 계산에서
147.4 px 를 쓰십시오.**

| RGB cam3 (2026-09-15 확정) | |
|---|---|
| 초점거리 | **1217 px** (1920×1080, 화각 76.5° × 47.9°) |
| 왜곡 | k1 **−0.3665** · k2 **+0.1223** |
| 두 카메라 회전 | pitch 4.4° · yaw 4.7° · roll 6.9° |
| 베이스라인 ‖T‖ | **58.1 mm** ⚠ 미확정 (녹화를 바꾸면 53.5~69.2) |
| 정합 오차 | 중앙 **1.25 px** |

> **열화상은 RGB 대비 180° 회전 장착**입니다 — 정합 전에
> `cv2.rotate(img, cv2.ROTATE_180)`. RGB 좌표는
> `cv2.undistortPoints(..., P=K)` 로 왜곡을 먼저 푸십시오.
> **캐노피는 3층(420 / 700 / 1000 mm)이고 층마다 다른 `H(d)` 를 씁니다.**

---

## Directory
```
├─Document                   ; API Documentation and User Manual
│  └─API
│    ├─Android               ; Android API
│    ├─Cpp                   ; C++ API
│    ├─CSharp                ; C# API
│    └─Python                ; Python API
└─examples                   ; TmSDK sample code
     ├─Android               ; Java application for android 
     ├─Linux                 ; Qt5-based C++ application for Linux
     ├─Python                ; Python application
     └─Windows
        ├─TmWinDotnet        ; C# application for Windows
        └─TmWinQt            ; Qt5-based C++ application for Window
```
## Requirement

Windows C++
- Windows 10 or 11
- Visual Studio 2022
- Qt5.14.2
- qtcreator

Windows C#
- Windows 10 or 11
- Visual Studio 2022

Windows Python
- Windows 10 or 11
- Python 3.9 or higher
- PyQt5
- qtcreator

Linux C++
- Ubuntu 20.04 or higher
- Gcc-11
- Qt5.14.2
- qtcreator

Linux Python
- Ubuntu 20.04 or higher
- Python 3.9 or higher
- PyQt5
- qtcreator

Android
- android-24 or later

## Downloads

```
git clone https://github.com/ThermoEye/TmSDK
```
or

Download from [releases](https://github.com/ThermoEye/TmSDK/releases)

## Installing

Please refer to [TmSDK Manual.pdf](https://github.com/ThermoEye/TmSDK/blob/main/Document/TmSDK%20Manual.pdf)

## Support

Thermoeye Inc. operates service channels to keep your camera running at all times. 
If you discover a problem with your camera, please get in touch with us for technical support.

- Website: [www.thermoeye.co.kr](http://www.thermoeye.co.kr)
- E-mail: help@thermoeye.co.kr
- Tel: +82-70-4489-6196
- Head Office: 307, Research Building 3, 70, Yuseong-daero 1689 beon-gil, Yuseong-gu, Daejeon, Republic of Korea
- Seoul R&D: 4~5F, 169 Sadang-ro, Dongjak-gu, Seoul, Republic of Korea
