# 카메라 탐색 및 직접 연결 진단 스크립트
# 실행: python3 scan_cameras.py
#
# ※ 이 스크립트는 Ubuntu(Linux) 전용입니다.
#   - dpkg 명령 (섹션 0): Linux 전용
#   - ping -c 플래그 (섹션 1): Linux 전용 (Windows는 ping -n)
#   섹션 2, 3은 Windows에서도 동작합니다.

import platform
import subprocess
from TmCore import TmCamera
from TmCore.TmTypes import *

CAMERA_IPS = ["192.168.0.151", "192.168.0.152"]

NAMES = [
    "TMC160EH",
    "TMC160E",  "TMC160B",  "TMC160F",
    "TMC160IE", "TMC160IB", "TMC160I",
    "TMC256E",  "TMC256B",  "TMC256I",
    "TMC256GE", "TMC256GB", "TMC256G",
    "TMC384GE", "TMC384GB", "TMC384G",
    "TMC80E",   "TMC80B",   "TMC80F",
]

IS_LINUX = platform.system() == "Linux"

# ─── 0. 네이티브 라이브러리 설치 확인 (Linux 전용) ───────────
print("[0] TmSDK 네이티브 라이브러리 설치 확인...")
if IS_LINUX:
    try:
        result = subprocess.run(["dpkg", "-l", "tmsdk*"], capture_output=True, text=True)
        if "tmsdk" in result.stdout:
            print(result.stdout)
        else:
            print("  ✘ tmsdk .deb 패키지가 설치되어 있지 않습니다!")
            print("  → 아래 명령으로 설치하세요:")
            print("    sudo dpkg -i tmsdk-2.0.0-lib-Ubuntu-22.04-amd64.deb")
    except Exception as e:
        print(f"  확인 오류: {e}")
else:
    print("  (Windows에서는 dpkg 확인 생략)")

# ─── 1. ping 확인 ─────────────────────────────────────────────
print("\n[1] ping 확인...")
for ip in CAMERA_IPS:
    if IS_LINUX:
        cmd = ["ping", "-c", "1", "-W", "1", ip]
    else:
        cmd = ["ping", "-n", "1", "-w", "1000", ip]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "✔" if result.returncode == 0 else "✘"
    print(f"  {status} {ip}")

# ─── 2. 자동 스캔 ─────────────────────────────────────────────
# get_remote_camera_list()는 1회만 호출 — 반복 호출 시 소켓 충돌 발생
print("\n[2] get_remote_camera_list() 스캔...")
try:
    cameras = TmCamera.get_remote_camera_list()
    if cameras:
        for i, cam in enumerate(cameras):
            print(f"  [Camera {i+1}] name={cam.name}  ip={cam.ip}  "
                  f"serial={cam.serial_number}  mac={cam.mac}")
            if cam.media_info_list:
                for m in cam.media_info_list:
                    print(f"    format: {m.format} {m.width}x{m.height}@{m.frame_rate}fps")
    else:
        print("  자동 스캔으로 발견된 카메라 없음")
        print("  → 카메라 전원 및 네트워크 연결 확인")
        print("  → 카메라가 점유 상태라면 전원 재공급 후 재시도")
except Exception as e:
    print(f"  오류: {e}")

# ─── 3. 직접 연결 시도 ────────────────────────────────────────
# 스캔에서 발견되지 않은 경우 모델명을 순서대로 시도
print(f"\n[3] 직접 연결 시도...")
for ip in CAMERA_IPS:
    print(f"\n  IP: {ip}")
    for name in NAMES:
        try:
            cam = TmCamera()
            ret = cam.open_remote_camera(name, "", "", ip)
            if ret:
                fmt = cam.get_camera_format()
                print(f"  ✔ 연결 성공!  name={name}  format={fmt}")
                cam.close()
                break
            del cam
        except Exception as e:
            print(f"  오류 ({name}): {e}")
            break
    else:
        print(f"  ✘ 모든 모델명 실패")
        print(f"  → 카메라 점유 문제일 수 있음 — 전원 재공급 후 재시도")
