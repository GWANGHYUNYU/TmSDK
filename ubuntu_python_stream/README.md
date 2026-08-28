# TmSDK Python Stream

Thermoeye TMC 시리즈 열화상카메라 다중 스트리밍 및 녹화 도구

---

## 목차

1. [파일 구성](#1-파일-구성)
2. [시스템 구성 및 네트워크 환경](#2-시스템-구성-및-네트워크-환경)
3. [패키지 설치](#3-패키지-설치)
4. [알려진 문제 및 해결 방법](#4-알려진-문제-및-해결-방법)
5. [실행 방법](#5-실행-방법)
6. [기능 설명](#6-기능-설명)
7. [장기 운용 관련](#7-장기-운용-관련)
8. [배포 체크리스트](#8-배포-체크리스트)
9. [Raw Y16 녹화 (화소 단위 온도 분석용)](#9-raw-y16-녹화-화소-단위-온도-분석용)

---

## 1. 파일 구성

```
python_stream/
├── multi_cameras.py                  # N대 카메라 동시 스트리밍·녹화 (PyQt5 GUI, 단일 ROI)
├── multi_cameras_multi_roi.py        # 확장: 복수 ROI(Spot/Line/Rect/Ellipse) + 예약 녹화
├── multi_cameras_multi_roi_logo.py   # 위와 동일 + 로고 오버레이
├── multi_cameras_multi_roi_y16.py    # ★ 위 기능 전부 + Raw Y16 라디오메트릭 저장
├── read_y16.py                       # ★ .y16raw 를 온도 배열/라벨링용 PNG 로 변환
├── raw_y16_recorder.py               # Raw Y16 저장 (카메라 1대, 예약·ROI 없음)
├── record_camera.py                  # 보조: 카메라 1대 단독 녹화 (CLI 인자로 IP 지정)
├── scan_cameras.py                   # 진단: 카메라 탐색·연결 테스트
├── TmSDK-2.0.0-py3-none-win_amd64.whl   # Windows 전용 Python 바인딩
└── README.md                         # 이 문서
```

> **화소 단위 온도 분석·segmentation 이 목적이라면
> `multi_cameras_multi_roi_y16.py` 를 쓰십시오.**
> AVI 는 SDK 가 프레임마다 auto-gain(히스토그램 풀스케일 스트레치)과
> 의사색·손실압축을 적용하므로 절대온도를 복원할 수 없습니다.
> 자세한 내용은 [9. Raw Y16 녹화](#9-raw-y16-녹화-화소-단위-온도-분석용) 참고.

> Ubuntu용 `.whl` 및 `.deb` 파일은 TmSDK GitHub Releases에서 별도 다운로드 필요
> → https://github.com/ThermoEye/TmSDK/releases

---

## 2. 시스템 구성 및 네트워크 환경

### 하드웨어 구성

```
[열화상카메라 TMC160F × 2]
        │ LAN (PoE)
   [PoE 스위치]
        │ LAN
   [Ubuntu PC]
     ├─ enp7s0   ← 유선 (카메라와 연결, IPv4 별도 설정 불필요)
     └─ wlx...   ← WiFi  192.168.0.64/24
```

### 카메라 정보

| 항목 | Camera 1 | Camera 2 |
|------|----------|----------|
| IP | 192.168.0.151 | 192.168.0.152 |
| 모델명 | TMC160F | TMC160F |
| Serial | TAE57F0261200001 | TAE57F0261200002 |
| MAC | 8C:1F:64:66:40:D5 | 8C:1F:64:66:40:D6 |

### TMC160EH 사양 (참고)

| 항목 | 값 |
|------|----|
| 해상도 | 160 × 120 px |
| 픽셀 피치 | 12 μm |
| 프레임레이트 | 8.7 Hz |
| 포맷 | Y16 |

> `query_frame(w, h)` 호출 시 SDK가 자동 스케일업하므로 미리보기·녹화 해상도는 자유롭게 설정 가능
> 현재 설정: `PREVIEW_W = 720`, `PREVIEW_H = 480`, `RECORD_FPS = 8.7`

---

## 3. 패키지 설치

### 3-1. TmSDK 설치 (Ubuntu 22.04)

TmSDK는 **두 가지 파일**이 모두 필요합니다.

```bash
# ① 네이티브 라이브러리 (.deb)
sudo dpkg -i tmsdk-2.0.0-lib-Ubuntu-22.04-amd64.deb

# ② Python 바인딩 (.whl)
pip3 install TmSDK-2.0.0-py3-none-manylinux2014_x86_64.whl
```

> ⚠️ `.deb` 없이 `.whl`만 설치하면 `import TmCore` 시 네이티브 라이브러리를 찾지 못해 오류 발생

### 3-2. TmSDK 설치 (Windows)

```powershell
python -m pip install .\TmSDK-2.0.0-py3-none-win_amd64.whl
```

> ⚠️ Windows PowerShell에서 `pip` 명령어가 인식되지 않을 수 있음
> → `python -m pip install ...` 형식으로 사용

### 3-3. Python 의존 패키지 설치

```bash
# Ubuntu
pip3 install PyQt5 opencv-python-headless numpy

# Windows
python -m pip install PyQt5 opencv-python numpy
```

> ⚠️ Ubuntu에서 `opencv-python` (headless 아닌 버전) 설치 시 Qt platform plugin 충돌 발생
> → **반드시 `opencv-python-headless` 사용** (자세한 내용은 섹션 4-3 참고)

### 3-4. 현재 설치된 패키지 버전 확인

```bash
pip3 list | grep -E "PyQt5|opencv|numpy|TmSDK"
```

| 패키지 | 버전 |
|--------|------|
| TmSDK | 2.0.0 |
| PyQt5 | 5.15.11 |
| PyQt5-Qt5 | 5.15.18 |
| PyQt5_sip | 12.18.0 |
| opencv-python-headless | 4.13.0.92 |
| numpy | 2.2.6 |

배포 시 동일 버전 고정 설치:

```bash
pip3 install \
  PyQt5==5.15.11 \
  opencv-python-headless==4.13.0.92 \
  numpy==2.2.6
```

---

## 4. 알려진 문제 및 해결 방법

### 4-1. 카메라 점유 문제 (가장 중요)

**증상**
```
[연결 시도 1/5] 192.168.0.151 실패, 3초 후 재시도...
[연결 시도 2/5] 192.168.0.151 실패, 3초 후 재시도...
...
[실패] 192.168.0.151
```
한쪽 카메라(예: .151)만 연결되고 다른 쪽은 계속 실패하는 경우

**원인**
이전 프로세스가 카메라를 열어둔 채 비정상 종료되거나, SDK가 카메라 내부 네트워크 스택을 점유한 상태로 남아있음

**해결 방법**
1. **카메라 물리 재부팅** (PoE 스위치 포트 off/on 또는 전원 재공급)
2. Ubuntu PC 재부팅은 효과 없음 — 카메라 측 문제이므로 카메라를 재부팅해야 함

**예방 조치 (코드 반영)**
- `get_remote_camera_list()` 호출을 `ScanWorker` 1개로 분리하여 **딱 1번만** 호출
- 이전 코드: 카메라마다 `ConnectWorker`가 각자 스캔 호출 → 동시 소켓 충돌 → "Failed to bind socket"
- 현재 코드: 스캔 1회 완료 후 스캔 결과를 각 `ConnectWorker`에 전달

---

### 4-2. 소켓 충돌 문제 (Failed to bind socket)

**증상**
```
Failed to bind socket:
[경고] 192.168.0.152 스캔 목록에 없음 — 기본값으로 시도
```

**원인**
`get_remote_camera_list()`를 여러 스레드에서 동시에 호출하면 SDK 내부에서 소켓 바인딩 충돌 발생

**해결 방법**
```python
# ✗ 잘못된 방법: ConnectWorker마다 스캔 호출
class ConnectWorker(QThread):
    def run(self):
        cam_list = TmCamera.get_remote_camera_list()  # 동시 호출 → 충돌

# ✓ 올바른 방법: ScanWorker 1개로 1회만 스캔
class ScanWorker(QThread):
    def run(self):
        cam_list = TmCamera.get_remote_camera_list()  # 단 1회
        self.scan_done.emit({c.ip: c for c in cam_list})
```

---

### 4-3. Qt platform plugin 'xcb' 충돌

**증상**
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
This application failed to start because no Qt platform plugin could be initialized.
```

**원인**
`opencv-python` 패키지가 자체 Qt 라이브러리를 내장하고 있어 PyQt5의 Qt 라이브러리와 충돌

**해결 방법**
```bash
pip3 uninstall opencv-python
pip3 install opencv-python-headless
```

> `opencv-python-headless`는 GUI 컴포넌트 없이 영상 처리 기능만 포함하므로 Qt 충돌이 발생하지 않음
> `VideoWriter`, `cvtColor`, `rectangle` 등 이 프로젝트에서 사용하는 기능은 모두 headless에서 지원됨

---

### 4-4. TmCore 모듈을 찾을 수 없음 (No module named 'TmCore')

**증상**
```
ModuleNotFoundError: No module named 'TmCore'
```

**원인 및 해결**

| 원인 | 해결 |
|------|------|
| `.whl`만 설치, `.deb` 미설치 (Ubuntu) | `sudo dpkg -i tmsdk-*.deb` 먼저 실행 |
| Windows에서 `pip` 명령어 미인식 | `python -m pip install ...` 사용 |
| 가상환경 미활성화 상태로 설치 | 동일 Python 환경에 재설치 확인 |

---

### 4-5. ROI 온도 측정 오류 (TmRoiRect is not defined)

**증상**
```
[FrameWorker] name 'TmRoiRect' is not defined
```

**원인**
TmSDK 2.0.0에서 `TmRoiRect` 클래스가 제거되거나 이름이 변경됨

**해결 방법**
```python
# ✗ 구버전 방식 (동작 안 함)
rect = TmRoiRect()
rect.set_xywh(x, y, w, h)
frame.do_measure(rect)

# ✓ 현재 올바른 방식
from TmCore.TmRoi import *

roi_mgr = TmRoiManager()
roi_mgr.add_item_xywh(RoiType.Rect, x, y, w, h)
item = roi_mgr.get_roi_item(0)
frame.do_measure(item)
rect_item = roi_mgr.get_roi_rect_item(0)
min_v = camera.get_temperature(rect_item.get_roi_minloc().value)
avg_v = camera.get_temperature(rect_item.get_roi_avgloc().value)
max_v = camera.get_temperature(rect_item.get_roi_maxloc().value)
```

---

### 4-6. 카메라 연결 순서 문제

**증상**
두 카메라 중 하나만 연결되고 하나는 실패 → 연결 순서를 바꾸면 반대쪽이 실패

**원인**
`get_remote_camera_list()` 호출 직후 곧바로 `open_remote_camera()` 호출 시 SDK 내부 상태가 완전히 초기화되기 전에 두 번째 카메라 연결 시도가 겹침

**해결 방법**
- 스캔 1회 완료 후 결과를 캐싱
- 각 카메라 연결은 스캔 완료 이후 시작 (현재 코드에 반영됨)
- 여전히 실패 시 → 카메라 물리 재부팅 (4-1 참고)

---

### 4-7. 카메라가 탐색되지 않음 (PC 이더넷 IP 미설정)

**증상**
```bash
python3 scan_cameras.py
# [스캔 완료] 0대 발견: []
```
또는
```
ping 192.168.0.151
# Destination Host Unreachable
```

카메라가 물리적으로 연결되어 있고 PoE 전원도 정상인데, `scan_cameras.py`에서 카메라가 하나도 탐색되지 않음

**원인**

카메라 IP가 `192.168.0.151`, `192.168.0.152` 등 `192.168.0.x` 대역을 사용하는데, PC의 이더넷 인터페이스(카메라와 연결된 포트)에 같은 서브넷의 IP가 할당되어 있지 않으면 통신 자체가 불가능합니다.

DHCP 환경에서 PC의 이더넷에 자동으로 `192.168.0.x` 대역 IP가 할당되지 않는 경우, **수동으로 고정 IP를 설정**해야 합니다.

**해결 방법 (Ubuntu 22.04)**

1. 카메라와 연결된 이더넷 인터페이스 이름을 확인합니다:
   ```bash
   ip link show
   # 예: enp7s0, eth0, eno1 등
   ```

2. 네트워크 설정에서 해당 인터페이스에 수동 IPv4를 설정합니다:
   ```
   Settings → Network → 해당 이더넷 → IPv4
   Method: Manual
   Address:  192.168.0.100
   Netmask:  255.255.255.0
   Gateway:  (비워둠)
   ```
   또는 CLI로:
   ```bash
   sudo ip addr add 192.168.0.100/24 dev enp7s0
   ```
   > CLI 방식은 재부팅 시 초기화됨. 영구 설정은 Netplan 또는 NetworkManager GUI 사용

3. IP 설정 후 카메라와 통신 가능한지 확인합니다:
   ```bash
   ping -c 2 192.168.0.151
   ping -c 2 192.168.0.152
   ```

**해결 방법 (Windows)**

1. `제어판` → `네트워크 및 인터넷` → `네트워크 및 공유 센터` → `어댑터 설정 변경`
2. 카메라와 연결된 이더넷 어댑터 우클릭 → `속성`
3. `인터넷 프로토콜 버전 4 (TCP/IPv4)` 선택 → `속성`
4. `다음 IP 주소 사용` 선택:
   ```
   IP 주소:       192.168.0.100
   서브넷 마스크:  255.255.255.0
   기본 게이트웨이: (비워둠)
   ```
5. 확인 후 `ping 192.168.0.151`로 통신 확인

> ⚠️ IP 주소는 카메라와 겹치지 않는 값을 사용해야 합니다 (`.151`, `.152` 제외).
> ⚠️ 인터넷용 이더넷과 카메라용 이더넷이 같은 포트라면, 고정 IP 설정 시 인터넷 연결이 끊길 수 있습니다. 가능하면 카메라 전용 이더넷 포트를 분리하세요.

---

### 4-8. TmSDK GitHub Releases 파일 선택

https://github.com/ThermoEye/TmSDK/releases 에서 버전별로 여러 파일이 제공됨

| 파일 | 용도 |
|------|------|
| `tmsdk-X.X.X-lib-Ubuntu-22.04-amd64.deb` | Ubuntu 네이티브 라이브러리 |
| `TmSDK-X.X.X-py3-none-manylinux2014_x86_64.whl` | Ubuntu Python 바인딩 |
| `TmSDK-X.X.X-py3-none-win_amd64.whl` | Windows Python 바인딩 |
| `TmSDK-X.X.X-lib-*.zip` | C++ 네이티브 라이브러리 (Python 불필요) |

> Python에서 사용 시 Ubuntu는 `.deb` + `.whl` 두 파일 모두 필요
> Windows는 `.whl` 하나만 필요

---

## 5. 실행 방법

### multi_cameras.py (단일 ROI)

```bash
cd python_stream/
python3 multi_cameras.py
```

- GUI가 열리면 자동으로 `CAMERAS` 리스트에 등록된 카메라에 연결 시도
- 추가 카메라는 상단 IP 입력란에 입력 후 **연결** 버튼 클릭
- 카메라별 ROI 1개(사각형) 드래그 설정 가능

### multi_cameras_multi_roi.py (복수 ROI + 예약 녹화)

```bash
cd python_stream/
python3 multi_cameras_multi_roi.py
```

- `multi_cameras.py`의 모든 기능을 포함하며, 아래 기능이 추가됨:
  - **복수 ROI**: 카메라당 여러 개의 ROI를 추가 가능 (Spot, Line, Rect, Ellipse)
  - **ROI 타입 선택**: 콤보박스에서 타입 선택 후 미리보기 화면에서 드래그
  - **ROI별 온도 측정**: 각 ROI마다 Min / Avg / Max 개별 표시 및 CSV 저장
  - **ROI 관리**: "마지막 ROI 삭제" / "ROI 전체 초기화" 버튼
  - **예약 녹화**: 매일 기준 시작/종료 시각(0~24시) 설정, 매일 자동 반복
    - "예약 녹화 설정" 클릭 → 시간 설정 UI 활성화 → "예약 시작" 클릭으로 예약 시작
    - 반복 간격 미설정 시: 설정 시간대 전체를 연속 녹화 (예: 8시~19시 통째로)
    - 반복 간격 설정 시: 간격마다 녹화 시간만큼만 녹화하여 용량 절감 (예: 60분 간격 / 녹화 10분 → 매 정시 10분만 녹화, 50분 대기)
    - 남은 시간 카운트다운 실시간 표시
    - 예약을 설정하지 않고 "전체 녹화" 버튼을 누르면 즉시 연속 녹화

### raw_y16_recorder.py (Raw Y16 비압축 저장)

```bash
# 기본 (720x480 해상도)
python3 raw_y16_recorder.py --ip 192.168.0.151

# 저장 폴더 지정
python3 raw_y16_recorder.py --ip 192.168.0.151 --output ./raw_data

# 카메라 원본 해상도(160x120)로 저장
python3 raw_y16_recorder.py --ip 192.168.0.151 --native
```

- 지정한 IP의 카메라 1대에서 Raw Y16 (16-bit) 프레임을 비압축 바이너리로 저장
- 저장 파일: `.y16raw` (프레임 데이터) + `.y16meta` (JSON 메타데이터)
- 2 GB 초과 시 자동 파일 분할 (`_part1`, `_part2`, ...)
- 저장된 데이터에서 픽셀별 온도 복원 가능 (`camera.get_temperature(raw_value)`)

### record_camera.py (단일 카메라 녹화)

```bash
python3 record_camera.py --ip 192.168.0.151
python3 record_camera.py --ip 192.168.0.152
```

### scan_cameras.py (진단용)

```bash
python3 scan_cameras.py
```

네트워크 연결, 카메라 탐색, 직접 연결 테스트를 순서대로 수행

---

## 6. 기능 설명

### multi_cameras.py 주요 기능

| 기능 | 설명 |
|------|------|
| N대 카메라 동시 스트리밍 | `CAMERAS` 리스트에 등록 또는 런타임 IP 입력으로 추가 |
| ROI 설정 | 미리보기 화면에서 마우스 드래그 (카메라당 사각형 1개) |
| ROI 온도 측정 | Min / Avg / Max 실시간 표시 |
| 개별 녹화 | 카메라별 **녹화 시작** 버튼 (초록→빨강으로 상태 표시) |
| 전체 녹화 | 상단 **전체 녹화** 버튼으로 모든 카메라 동시 녹화 시작 |
| 저장 폴더 설정 | 전역 폴더 + 카메라별 개별 폴더 (개별 설정 우선) |
| AVI 자동 분할 | 1.4 GB 또는 1시간 초과 시 자동으로 새 파일 생성 |
| 카메라 자동 재연결 | 연결 끊김 감지 시 자동 재연결 + 녹화 자동 재개 |
| CSV 기록 | timestamp, roi_x, roi_y, roi_w, roi_h, min_temp, avg_temp, max_temp |
| 로그 파일 | `logs/tmsdk.log` 일별 로테이션, 60일 보관 |

### multi_cameras_multi_roi.py 추가 기능

`multi_cameras.py`의 모든 기능을 포함하며, 아래 기능이 추가됩니다.

| 기능 | 설명 |
|------|------|
| 복수 ROI | 카메라당 여러 개의 ROI 추가 가능 |
| ROI 타입 | Spot(점), Line(선), Rect(사각형), Ellipse(타원) 선택 |
| ROI별 온도 | 각 ROI마다 Min / Avg / Max 개별 측정·표시·CSV 저장 |
| ROI 색상 구분 | ROI별로 다른 색상 (노랑, 시안, 빨강 등 8색 순환) |
| ROI 관리 | "마지막 ROI 삭제" / "ROI 전체 초기화" 버튼 |
| 예약 녹화 | 매일 시작/종료 시각(0~24시) 설정, 매일 자동 반복 |
| 반복 간격 | 간격 설정 시 슬롯 단위 반복 녹화, 미설정 시 전체 구간 연속 녹화 |
| 녹화 시간 | 반복 간격 내 실제 녹화할 시간 설정 (예: 60분 간격 중 10분만 녹화 → 용량 1/6 절감) |

#### 예약 녹화 사용 방법

1. **"예약 녹화 설정" 버튼 클릭** → 시작 시각, 종료 시각, 반복 간격, 녹화 시간 입력 UI가 활성화됨
2. **시작/종료 시각 설정** (예: 시작 `8시`, 종료 `19시`)
3. **반복 간격 설정** (선택 사항)
   - **반복 간격 체크 안 함**: 8시~19시 전체를 하나의 연속 영상으로 녹화
   - **반복 간격 체크 시**: 반복 간격과 녹화 시간을 함께 설정
     - 반복 간격 `60분` + 녹화 시간 `10분`: 8:00~8:10, 9:00~9:10, ... 18:00~18:10 녹화 (매 정시 10분만 녹화, 50분 대기)
     - 반복 간격 `60분` + 녹화 시간 `60분`: 8:00~9:00, 9:00~10:00, ... 연속 녹화와 동일 (파일만 분할)
     - 반복 간격 `30분` + 녹화 시간 `5분`: 8:00~8:05, 8:30~8:35, 9:00~9:05, ... (30분마다 5분씩)
   - 녹화 시간은 반복 간격 이하여야 함 (초과 시 경고 메시지)
4. **"예약 시작" 버튼 클릭** → 예약 활성화, 설정 UI 잠금, 상태 표시줄에 남은 시간 표시
5. **"예약 취소" 버튼 클릭** → 녹화 중이면 정지, 예약 해제, 초기 상태로 복원

#### 예약 녹화 용량 절감 예시

| 설정 | 하루(24h) 실제 녹화량 | 절감률 |
|------|----------------------|--------|
| 0~24시, 반복 간격 없음 | 24시간 | - |
| 0~24시, 간격 60분 / 녹화 10분 | 4시간 | 약 83% |
| 0~24시, 간격 60분 / 녹화 5분 | 2시간 | 약 92% |
| 8~19시, 간격 30분 / 녹화 10분 | 7.3시간 | 약 67% |

> 예약 녹화를 설정하지 않고 **"전체 녹화"** 또는 카메라별 **"녹화 시작"** 버튼을 누르면 시간 제한 없이 즉시 연속 녹화됩니다.

### raw_y16_recorder.py 주요 기능

| 기능 | 설명 |
|------|------|
| Raw Y16 저장 | 16-bit 원본 프레임을 비압축 바이너리로 저장 |
| 메타데이터 | `.y16meta` (JSON)에 해상도, FPS, 프레임 수, 타임스탬프 기록 |
| 원본 해상도 | `--native` 옵션으로 카메라 원본 해상도(160x120) 저장 |
| 자동 분할 | 2 GB 초과 시 `_part1`, `_part2`... 자동 분할 |
| 온도 복원 | 저장된 raw 값에서 `get_temperature()`로 픽셀별 온도 계산 가능 |

### 초기 카메라 설정 변경

`multi_cameras.py`, `multi_cameras_multi_roi.py` 상단의 `CAMERAS` 리스트를 수정:

```python
CAMERAS = [
    {"ip": "192.168.0.151", "label": "Camera 1"},
    {"ip": "192.168.0.152", "label": "Camera 2"},
    # 추가 카메라는 여기에 등록
]
```

### 녹화 출력 파일

**multi_cameras.py / multi_cameras_multi_roi.py:**
```
output/
├── 192_168_0_151_20250327_143022.avi   # 영상 (ROI 오버레이 포함)
├── 192_168_0_151_20250327_143022.csv   # 온도 로그 (ROI별)
├── 192_168_0_151_20250327_153022.avi   # 1시간 후 자동 분할
└── ...
```

multi_cameras.py CSV 형식:
```
timestamp,roi_x,roi_y,roi_w,roi_h,min_temp,avg_temp,max_temp
2025-03-27 14:30:22.115,100,80,200,150,28.50,31.20,35.80
```

multi_cameras_multi_roi.py CSV 형식 (ROI별 컬럼 동적 생성):
```
timestamp,roi0_rect_params,roi0_rect_min,roi0_rect_avg,roi0_rect_max,roi1_spot_params,roi1_spot_min,roi1_spot_avg,roi1_spot_max
2025-03-27 14:30:22.115,(100,80 200x150),28.50,31.20,35.80,(300,200),29.10,29.10,29.10
```

**raw_y16_recorder.py:**
```
raw_output/
├── 192_168_0_151_20250327_143022.y16raw    # Raw 프레임 연속 바이너리 (uint16 LE)
├── 192_168_0_151_20250327_143022.y16meta   # JSON 메타데이터
├── 192_168_0_151_20250327_153022_part1.y16raw   # 2 GB 초과 시 분할
└── ...
```

---

## 7. 장기 운용 관련

한 달 이상 연속 운용을 위해 다음 기능이 구현되어 있습니다.

### AVI 자동 분할 (1.4 GB / 1시간)

- AVI 컨테이너는 2 GB 초과 시 파일 손상 위험
- 100 프레임마다 파일 크기 및 경과 시간 체크
- 분할 시 현재 프레임은 새 파일에 기록됨

```python
_SPLIT_MAX_BYTES = int(1.4 * 1024 ** 3)   # 1.4 GB
_SPLIT_MAX_SECS  = 3600                    # 1시간
```

### 자동 재연결

- `FrameWorker`에서 연속 예외 30회(약 3초) 발생 시 재연결 요청
- 자동으로 스캔 → 연결 최대 5회 재시도
- 재연결 성공 시 녹화도 자동 재개

```python
_RECONNECT_THRESHOLD = 30   # 연속 예외 임계값
```

### CSV 안전 쓰기

- 30 프레임마다 `flush()` + `fsync()` 호출
- 비정상 종료 시 최대 약 3.4초 분량 손실

```python
_CSV_FLUSH_EVERY = 30
```

### 로그 로테이션

- `logs/tmsdk.log` 매일 자정 자동 로테이션
- 60일치 보관 (`tmsdk.log.2025-03-27` 형식)
- INFO 이상 콘솔 출력, DEBUG 이상 파일 기록

---

## 8. 배포 체크리스트

새 Ubuntu 머신에 설치할 때 아래 순서를 따릅니다.

### Step 1. TmSDK 파일 다운로드

https://github.com/ThermoEye/TmSDK/releases 에서 다운로드:

```
tmsdk-2.0.0-lib-Ubuntu-22.04-amd64.deb
TmSDK-2.0.0-py3-none-manylinux2014_x86_64.whl
```

### Step 2. TmSDK 설치

```bash
sudo dpkg -i tmsdk-2.0.0-lib-Ubuntu-22.04-amd64.deb
pip3 install TmSDK-2.0.0-py3-none-manylinux2014_x86_64.whl
```

### Step 3. Python 패키지 설치

```bash
pip3 install PyQt5 opencv-python-headless numpy
```

> `opencv-python` (headless 아닌 버전) 이 이미 설치되어 있으면 반드시 교체:
> ```bash
> pip3 uninstall opencv-python
> pip3 install opencv-python-headless
> ```

### Step 4. python_stream 폴더 복사

```bash
scp -r python_stream/ user@<새서버IP>:~/
# 또는 USB/git 등으로 전달
```

### Step 5. 카메라 IP 설정

`multi_cameras.py` 상단 `CAMERAS` 리스트를 현재 환경에 맞게 수정:

```python
CAMERAS = [
    {"ip": "192.168.0.151", "label": "Camera 1"},
    {"ip": "192.168.0.152", "label": "Camera 2"},
]
```

### Step 6. 네트워크 확인

```bash
# 카메라 ping 확인
ping -c 2 192.168.0.151
ping -c 2 192.168.0.152

# 카메라 자동 탐색 테스트
cd python_stream/
python3 scan_cameras.py
```

`scan_cameras.py` 정상 출력 예시:
```
[Camera 1] name=TMC160F  ip=192.168.0.151  serial=TAE57F0261200001
[Camera 2] name=TMC160F  ip=192.168.0.152  serial=TAE57F0261200002
```

### Step 7. 실행

```bash
cd python_stream/
python3 multi_cameras.py
```

### 배포 전 확인 사항

- [ ] `sudo dpkg -i tmsdk-*.deb` 완료
- [ ] `pip3 install TmSDK-*.whl` 완료
- [ ] `pip3 install PyQt5 opencv-python-headless numpy` 완료
- [ ] `opencv-python` (headless 아닌 버전) 미설치 확인
- [ ] `ping 192.168.0.151` 응답 확인
- [ ] `python3 scan_cameras.py` 에서 카메라 탐색 성공
- [ ] `CAMERAS` 리스트 IP 주소 현재 환경에 맞게 수정
- [ ] 저장 폴더 (`output/`) 쓰기 권한 확인
- [ ] 디스크 여유 공간 충분한지 확인 (한 달 연속 녹화 시 수 TB 필요)

---

## 부록: 디스크 사용량 추정

| 해상도 | FPS | 예상 크기 (1시간) | 예상 크기 (1달, 2대) |
|--------|-----|-------------------|----------------------|
| 720×480 | 8.7 | 약 3~8 GB | 약 4~11 TB |

> XVID 코덱 압축률에 따라 크게 달라질 수 있음
> 실제 사용 시 첫 1시간 분할 파일 크기를 확인하고 디스크 용량을 계획하세요.

---

## 9. Raw Y16 녹화 (화소 단위 온도 분석용)

### 9-1. 왜 필요한가 — AVI 로는 온도를 복원할 수 없다

기존 `output/` 데이터를 검증한 결과, SDK 가 만드는 AVI 에는 네 겹의 손실이 있습니다.

| 손실 | 내용 |
|------|------|
| **프레임별 auto-gain** | 매 프레임 밝기를 0~241 로 풀스케일 스트레치. 온도가 올라도 밝기는 그대로거나 오히려 내려감 (실측: .152 카메라에서 온도-밝기 상관 r = **-0.507**) |
| 의사색 컬러맵 | INFERNO 계열 팔레트 적용 → 그레이스케일 변환이 비선형 손실 |
| XVID 손실압축 | 색상당 약 13 단위 오차. 순수 LUT 라면 256색이어야 할 프레임에 고유색 17,000개 이상 |
| 보간 업스케일 | 160x120 → 720x480 은 4.5x/4x 보간이라 정보량 증가 없이 종횡비만 4:3 → 3:2 로 왜곡 |

**결론: AVI 는 눈으로 확인하거나 마스크를 그릴 때만 쓰고,
온도 분석에는 `.y16raw` 를 쓰십시오.**

### 9-2. 실행

```bash
# 기본 — 카메라 원본 160x120 으로 Y16 + CSV + 참고용 AVI 저장
python3 multi_cameras_multi_roi_y16.py

# Y16 + CSV 만 (용량 절감)
python3 multi_cameras_multi_roi_y16.py --no-avi

# 저장 폴더 지정
python3 multi_cameras_multi_roi_y16.py --output /data/thermal

# 기존과 동일한 720x480 으로 저장 (권장하지 않음 — 용량 18배)
python3 multi_cameras_multi_roi_y16.py --query-size 720x480
```

GUI 사용법은 `multi_cameras_multi_roi.py` 와 동일합니다.
복수 ROI, 예약 녹화, 자동 재연결, 전체 녹화 버튼 모두 그대로 있습니다.

### 9-3. multi_cameras_multi_roi.py 와 달라진 점

| 항목 | multi_cameras_multi_roi.py | multi_cameras_multi_roi_y16.py |
|------|---------------------------|-------------------------------|
| 저장 형식 | AVI + CSV | **y16raw + y16meta + CSV** (+ 선택적 AVI) |
| 기본 해상도 | 720x480 (보간) | **160x120 (카메라 원본)** |
| ROI 좌표계 | 720x480 표시 좌표 | **원본 화소 좌표** — y16raw 에 그대로 대응 |
| ROI 크기 표시 | 없음 | **실제 검출기 화소 수 표시** (9화소 미만이면 경고) |
| 온도 복원 | 불가 | **y16meta 의 LUT 로 SDK 없이 복원 가능** |
| CSV ↔ 영상 정렬 | 별도 시그널이라 어긋날 수 있음 | **한 시그널로 묶어 1:1 보장** |
| 녹화 중 ROI 변경 | 허용 (CSV 열 수 깨짐) | **차단** |

> ROI 크기 경고에 대해: 열화상 측정에서는 대상이 최소 3x3 화소를 덮어야
> 방출 에너지의 90%를 포착합니다. 기존 `output/` 의 타원 ROI 7개는
> 720x480 좌표로 그려져 있어 **실제로는 검출기 화소 4~18개**(전체의 0.04~0.06%)
> 만 덮고 있었습니다. 원본 좌표계로 바꾸면 이 문제가 화면에서 바로 보입니다.

### 9-4. 저장 결과물

녹화 1회(또는 분할 1개)당 같은 이름으로 4개 파일이 생깁니다.

```
raw_output/
├── 192_168_0_151_20260806_080000.y16raw    # uint16 LE 연속 프레임
├── 192_168_0_151_20260806_080000.y16meta   # JSON: 해상도/타임스탬프/ROI/LUT
├── 192_168_0_151_20260806_080000.csv       # ROI별 Min/Avg/Max (행 N = 프레임 N)
└── 192_168_0_151_20260806_080000.avi       # 참고용 (--no-avi 시 생략)
```

`.y16meta` 에는 **raw → 섭씨 변환 LUT(1024 포인트)** 가 들어 있어,
나중에 카메라나 TmSDK 없이도 온도를 복원할 수 있습니다.

```python
import json, numpy as np
meta = json.load(open("....y16meta", encoding="utf-8"))
lut  = meta["raw_to_temperature_lut"]
xs   = np.arange(lut["count"]) * lut["stride"] + lut["raw_start"]
arr  = np.fromfile("....y16raw", dtype="<u2").reshape(
           -1, meta["height"], meta["width"])
celsius = np.interp(arr.astype(float), xs, lut["values"])   # (frames, H, W)
```

### 9-5. read_y16.py — 저장된 데이터 읽기

```bash
# 메타데이터와 온도 통계 확인
python3 read_y16.py info raw_output/

# 데이터 품질 진단 — 장기 운용 시 정기적으로 실행할 것
python3 read_y16.py check raw_output/

# 전체 온도 범위 조사 (PNG 내보내기 범위를 정하기 위해 먼저 실행)
python3 read_y16.py range raw_output/

# 섭씨 float32 배열로 내보내기
python3 read_y16.py npy raw_output/ --out temps/ --stride 10

# 라벨링용 8-bit PNG (고정 온도 범위 필수)
python3 read_y16.py png raw_output/ --out frames/ --tmin 15 --tmax 45 --stride 300
```

`check` 는 다음을 찾아냅니다. 장기 운용 중 데이터가 조용히 망가지는 것을 막습니다.

| 진단 항목 | 내용 |
|---|---|
| 프레임 간격 | 타임스탬프 기준 끊긴 구간과 누락 시간 |
| **셔터(FFC) 이벤트** | 전역 평균이 계단형으로 튀는 지점. 비냉각 볼로미터의 정기 보정 동작으로, 이 프레임 전후는 온도가 불연속이 됩니다 |
| 불량 화소 | 값이 전혀 변하지 않는 고정 화소, 변동폭이 비정상적으로 큰 화소 |
| 극단값 위치 | 최고·최저 raw 가 **몇 번 프레임 어느 좌표**에서 났는지, 그 화소가 상시 극단인지 일시적 이벤트인지 |
| 권장 PNG 범위 | 극단값을 잘라내고 캐노피 대비를 살리는 `--tmin/--tmax` |

셔터 오탐이 많으면 `--ffc-threshold 0.3` 처럼 임계값을 올리십시오 (기본 0.15°C).
고정 물체 탐지 민감도는 `--hotspot-delta` (기본 캐노피 중앙값 ±2.0°C).

### 9-6. 셔터(FFC) 대응 — TMC160F 실측 특성

2026-08-06 파일럿에서 확인된 실측값입니다.

| 항목 | 실측 |
|---|---|
| FFC 주기 | **182.8초 (3.05분)** — 두 카메라 동일, 편차 0.07초 |
| 셔터 블랙아웃 | 약 **1.82초** 프레임 전송 중단 (전체의 1.0%) |
| 재개 직후 계단 | **+0.29 ~ +0.74°C** (평균 +0.48°C), 전부 양의 방향 |
| 실효 파형 | 주기 183초, 진폭 약 **±0.24°C 의 톱니파** |
| 프레임 손상 | 재개 직후 **1~2 프레임의 상단 행(y=0,1)** 이 깨짐 (−17°C, +46°C 등) |

대응은 세 가지이며, 세 번째가 근본 해결책입니다.

**① 손상 프레임 제외** — `npy`/`png` 내보내기는 기본으로 FFC 직후 2장을 뺍니다.
```bash
python3 read_y16.py npy raw_output/ --out temps/ --skip-after-ffc 2
```
`_frameidx.npy` 가 함께 저장되어 원본 프레임 번호·타임스탬프와 대조할 수 있습니다.

**② 슬롯 길이를 FFC 주기의 정수배로** — 주기 신호를 길이 `L` 인 창으로 평균낼 때
`L` 이 주기 `T` 의 정수배가 아니면 슬롯마다 무작위 편향이 남습니다.

| 슬롯 | L/T | 최대 편향 | 판정 |
|---|---|---|---|
| 2분 | 0.656 | ±0.17°C | ✗ |
| **3분** | 0.985 | ±0.01°C | ◎ |
| 4분 | 1.313 | ±0.15°C | ✗ |
| **6분** | 1.969 | ±0.02°C | ◎ |
| **9분** | 2.954 | ±0.02°C | ◎ |

**3분·6분·9분 슬롯을 쓰십시오.** 2분·4분은 일주기 분석에 잡음으로 들어갑니다.

**③ 화면 내 참조표면** — 온도를 아는 무광 흑색판을 항상 시야에 두고 매 프레임
그 값을 빼면 톱니가 완전히 사라집니다. 슬롯 길이와 무관해지므로 장기 운용의
정답입니다.

> ⚠️ PNG 내보내기 시 `--tmin/--tmax` 를 **모든 파일에 동일하게** 지정해야 합니다.
> 프레임별 자동 정규화를 쓰면 AVI 와 똑같은 문제(밝기 매핑이 프레임마다 달라짐)가
> 생겨 학습·라벨링에 쓸 수 없게 됩니다. 지정하지 않으면 스크립트가 거부합니다.
> 학습 입력은 `--colormap` 없이 회색조로 쓰는 편이 좋습니다
> (컬러맵은 정보를 늘리지 않고 채널만 3배로 늘립니다).

### 9-6. 용량 계획

| 해상도 | 프레임당 | 연속 1시간 | 8~19시 연속 / 카메라 2대 | 60분 간격 10분 녹화 |
|--------|---------|-----------|------------------------|-------------------|
| 160x120 (기본) | 38.4 KB | 약 1.2 GB | 약 26 GB/일 | 약 4.3 GB/일 |
| 720x480 | 691 KB | 약 21 GB | 약 464 GB/일 | 약 77 GB/일 |

기본값인 160x120 은 정보 손실이 전혀 없으면서 720x480 대비 용량이 1/18 입니다.

### 9-7. 알려진 동작

- **녹화 중에는 ROI 를 바꿀 수 없습니다.** CSV 헤더는 녹화 시작 시점의 ROI
  개수로 고정되므로, 중간에 바꾸면 열 수가 어긋나 파일 전체를 못 쓰게 됩니다.
  ROI 를 바꾸려면 녹화를 정지한 뒤 다시 시작하십시오.
- **파일은 1시간 또는 1.4 GB 마다 자동 분할**되며, y16raw·y16meta·csv·avi 가
  항상 같은 이름으로 함께 분할됩니다.
- 디스크가 프레임 속도를 못 따라가면 `프레임 처리 지연 N초` 경고가
  `logs/tmsdk_y16.log` 에 기록됩니다. 이 경고가 계속 나오면 저장 해상도를
  줄이거나 더 빠른 디스크를 쓰십시오.

---

*최종 업데이트: 2026-08-06*
*TmSDK 버전: 2.0.0*
