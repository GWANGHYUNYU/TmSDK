#################################################################
# File: multi_cameras_multi_roi_y16.py
#
# 실행 (Ubuntu 22.04):
#   python3 multi_cameras_multi_roi_y16.py                  # 원본 160x120 저장 (권장)
#   python3 multi_cameras_multi_roi_y16.py --no-avi         # Y16 + CSV만 저장
#   python3 multi_cameras_multi_roi_y16.py --query-size 720x480
#
# 이 스크립트는 multi_cameras_multi_roi.py 의 모든 기능을 그대로 갖되,
# 녹화 결과물로 Raw Y16 (라디오메트릭 원본) 을 함께 저장한다.
#
# 기능:
#   - 카메라 N대 동시 스트리밍 (런타임에 추가 가능)
#   - 각 카메라별 복수 ROI 설정 (Spot, Line, Rect, Ellipse)
#   - ROI 타입 선택 후 드래그로 추가, 개별 삭제 또는 전체 초기화
#   - 각 ROI별 온도(Min/Avg/Max) 개별 측정 및 CSV 저장
#   - 각 카메라별 또는 전체 일괄 녹화
#   - 예약 녹화: 시작/종료 시간 지정, 반복 간격 설정 가능
#   - 카메라 연결 끊김 시 자동 재연결 + 녹화 자동 재개
#   - 로그 파일 자동 기록 (logs/tmsdk_y16.log, 일별 로테이션 60일 보관)
#
# multi_cameras_multi_roi.py 와 달라진 점:
#   1) ★ Raw Y16 저장: 프레임마다 16-bit 원본을 .y16raw 로 비압축 기록.
#      AVI 는 SDK 가 프레임별 auto-gain + 의사색 + XVID 손실압축을 적용하므로
#      절대온도를 복원할 수 없다. 화소 단위 온도 분석에는 .y16raw 를 써야 한다.
#   2) ★ 기본 저장 해상도가 카메라 원본(160x120). 기존 720x480 은 4.5x/4x
#      보간이라 정보량이 늘지 않으면서 용량만 18배가 된다.
#      ROI 좌표도 원본 화소 기준으로 기록되므로, 화면에 그린 ROI 가
#      실제 검출기 화소 몇 개를 덮는지 UI 에 그대로 표시된다.
#   3) ★ raw→온도 변환표(LUT)를 .y16meta 에 함께 저장. 나중에 카메라나
#      TmSDK 없이도 .y16raw 를 온도로 복원할 수 있다.
#   4) ★ .y16raw / .csv / .avi 가 항상 같은 파일명으로 동시에 분할되어
#      프레임 번호와 CSV 행 번호가 1:1 로 대응한다.
#
# 저장 결과물 (녹화 1회당):
#   192_168_0_151_20260806_080000.y16raw   # uint16 LE 연속 프레임
#   192_168_0_151_20260806_080000.y16meta  # JSON: 해상도/타임스탬프/ROI/LUT
#   192_168_0_151_20260806_080000.csv      # ROI별 Min/Avg/Max 온도
#   192_168_0_151_20260806_080000.avi      # 참고용 시각화 (--no-avi 로 생략)
#
# 용량 (카메라 1대, 연속 녹화 기준):
#   160x120  →  38.4 KB/frame  ×  8.7 fps  =  약 1.2 GB/시간
#   720x480  →  691 KB/frame   ×  8.7 fps  =  약 21 GB/시간  (권장하지 않음)
#
# 요구사항:
#   pip3 install PyQt5 opencv-python-headless numpy
#################################################################

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

import cv2
import numpy as np

from PyQt5.QtCore import QThread, Qt, pyqtSignal, QRect, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLineEdit, QFileDialog,
    QScrollArea, QMessageBox, QComboBox,
    QCheckBox, QSpinBox
)

from TmCore import TmCamera
from TmCore.TmTypes import *
from TmCore.TmRoi import *


# ─────────────────────────────────────────────────────────────
# 로거 설정
# ─────────────────────────────────────────────────────────────
LOG_DIR = "logs"


def _setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger("TmSDK.Y16")
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "tmsdk_y16.log"),
        when="midnight", interval=1, backupCount=60, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)
    return log


logger = _setup_logger()


# ─────────────────────────────────────────────────────────────
# ★ 시작 시 자동 연결할 카메라 목록
# ─────────────────────────────────────────────────────────────
CAMERAS = [
    {"ip": "192.168.0.151", "label": "Camera 1"},
    {"ip": "192.168.0.152", "label": "Camera 2"},
]

DEFAULT_OUTPUT_DIR = "raw_output"
RECORD_FPS = 8.7

# 화면 표시 크기. 실제 저장 해상도와 무관하며, 프레임 종횡비에 맞춰
# DISPLAY_H 를 기준으로 폭이 자동 계산된다 (160x120 → 640x480).
DISPLAY_H     = 480
DISPLAY_W_MAX = 760

# 원본 화소를 정직하게 보여주기 위해 최근접 보간을 쓴다.
# 부드러운 화면을 원하면 cv2.INTER_LINEAR 로 바꿀 것.
DISPLAY_INTERP = cv2.INTER_NEAREST

BASE_FONT_SIZE = 12
BTN_H_TOP      = 52
BTN_H_PANEL    = 58

# ── 장기 운용 관련 상수 ───────────────────────────────────────
_RECONNECT_THRESHOLD = 30                      # 연속 예외 N회 → 재연결 요청
_SPLIT_MAX_BYTES     = int(1.4 * 1024 ** 3)    # 1.4 GB 초과 시 파일 분할
_SPLIT_MAX_SECS      = 3600                    # 1시간 초과 시 파일 분할
_FLUSH_EVERY         = 30                      # N 프레임마다 flush (~3.4초 @ 8.7 fps)
_SPLIT_CHECK_EVERY   = 100                     # N 프레임마다 분할 조건 검사

# ── raw→온도 LUT 생성 파라미터 ───────────────────────────────
_LUT_STRIDE       = 64      # raw 값 샘플 간격 (65536 / 64 = 1024 포인트)
_LUT_TIME_BUDGET  = 5.0     # LUT 생성에 허용할 최대 초

# ── ROI 타입 매핑 ────────────────────────────────────────────
ROI_TYPE_NAMES = {
    RoiType.Spot:    "Spot",
    RoiType.Line:    "Line",
    RoiType.Rect:    "Rect",
    RoiType.Ellipse: "Ellipse",
}

ROI_COLORS = [
    QColor(255, 255,   0),   # 노랑
    QColor(  0, 255, 255),   # 시안
    QColor(255, 100, 100),   # 빨강
    QColor(100, 255, 100),   # 초록
    QColor(255, 100, 255),   # 마젠타
    QColor(255, 180,   0),   # 오렌지
    QColor(100, 180, 255),   # 하늘
    QColor(200, 200, 200),   # 회색
]

# ── 실행 옵션 (main 에서 설정) ────────────────────────────────
OPT_QUERY_SIZE = None    # None = 카메라 원본 해상도, 또는 (w, h)
OPT_WRITE_AVI  = True


# ─────────────────────────────────────────────────────────────
# ROI 데이터 구조
# ─────────────────────────────────────────────────────────────
class RoiItem:
    """한 개의 ROI 정보를 담는 데이터 클래스.

    좌표는 모두 '저장 해상도(=카메라 질의 해상도)' 기준이다.
    화면 표시 좌표가 아니므로 .y16raw 프레임에 그대로 대응한다.
    """

    def __init__(self, roi_type: RoiType, x1: int, y1: int, x2: int, y2: int):
        self.roi_type = roi_type
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def label(self, idx: int) -> str:
        return f"ROI{idx}({ROI_TYPE_NAMES[self.roi_type]})"

    def geometry_str(self) -> str:
        if self.roi_type == RoiType.Spot:
            return f"({self.x1},{self.y1})"
        if self.roi_type == RoiType.Line:
            return f"({self.x1},{self.y1})->({self.x2},{self.y2})"
        # Rect / Ellipse: x1,y1 = 위치, x2,y2 = 너비,높이
        return f"({self.x1},{self.y1} {self.x2}x{self.y2})"

    def pixel_count(self) -> float:
        """이 ROI 가 덮는 실제 검출기 화소 수(근사)."""
        if self.roi_type == RoiType.Spot:
            return 1.0
        if self.roi_type == RoiType.Line:
            return float(max(abs(self.x2 - self.x1), abs(self.y2 - self.y1)) + 1)
        if self.roi_type == RoiType.Rect:
            return float(self.x2 * self.y2)
        return 3.14159265 / 4.0 * self.x2 * self.y2   # Ellipse

    def to_dict(self) -> dict:
        return {
            "type": ROI_TYPE_NAMES[self.roi_type],
            "x": self.x1, "y": self.y1, "w": self.x2, "h": self.y2,
            "geometry": self.geometry_str(),
            "approx_pixels": round(self.pixel_count(), 2),
        }


class RoiSample:
    """한 프레임에서 측정한 ROI 온도의 값 복사본.

    FrameWorker 는 워커 스레드에서 측정한 뒤 메인 스레드로 시그널을 보낸다.
    이때 RoiItem 을 그대로 넘기면 시그널이 큐에 머무는 사이 워커가 다음
    프레임 값으로 같은 객체를 덮어써서, CSV 에 그 프레임과 다른 시각의
    온도가 기록된다. 그래서 값만 떼어 스냅샷으로 넘긴다.
    """

    __slots__ = ("roi_type", "geometry", "min_temp", "avg_temp", "max_temp")

    def __init__(self, roi_type, geometry, min_temp, avg_temp, max_temp):
        self.roi_type = roi_type
        self.geometry = geometry
        self.min_temp = min_temp
        self.avg_temp = avg_temp
        self.max_temp = max_temp


# ─────────────────────────────────────────────────────────────
# 스캔 전용 스레드 (소켓 충돌 방지 — 1회 호출)
# ─────────────────────────────────────────────────────────────
class ScanWorker(QThread):
    scan_done = pyqtSignal(dict)   # {ip: cam_info}

    def run(self):
        logger.info("[스캔] 카메라 목록 조회 중...")
        cam_list = TmCamera.get_remote_camera_list()
        cam_map  = {c.ip: c for c in cam_list}
        logger.info(f"[스캔 완료] {len(cam_map)}대 발견: {list(cam_map.keys())}")
        self.scan_done.emit(cam_map)


# ─────────────────────────────────────────────────────────────
# 카메라 연결 스레드
# ─────────────────────────────────────────────────────────────
class ConnectWorker(QThread):
    connected = pyqtSignal(object, str)   # (TmCamera, label)
    failed    = pyqtSignal(str)           # ip

    def __init__(self, ip, label, cam_info=None):
        super().__init__()
        self.ip       = ip
        self.label    = label
        self.cam_info = cam_info

    def run(self):
        if self.cam_info:
            name   = self.cam_info.name
            serial = self.cam_info.serial_number
            mac    = self.cam_info.mac
            fmt    = (self.cam_info.media_info_list[0].format
                      if self.cam_info.media_info_list else "Y16")
            logger.info(f"[연결] {self.ip}  name={name}  serial={serial}")
        else:
            name, serial, mac, fmt = "TMC160F", "", "", "Y16"
            logger.warning(f"[연결] {self.ip} 스캔 결과 없음 — 기본값 사용")

        for attempt in range(5):
            cam = TmCamera()
            ret = cam.open_remote_camera(name, serial, mac, self.ip, fmt)
            if ret:
                self.connected.emit(cam, self.label)
                return
            del cam
            logger.warning(f"[연결 시도 {attempt+1}/5] {self.ip} 실패, 3초 후 재시도...")
            time.sleep(3)

        self.failed.emit(self.ip)


# ─────────────────────────────────────────────────────────────
# 프레임 캡처 스레드 (복수 ROI + Raw Y16)
# ─────────────────────────────────────────────────────────────
class FrameWorker(QThread):
    """한 프레임에서 RGB 프리뷰 · Raw Y16 · ROI 온도를 한 번에 뽑아
    단일 시그널로 내보낸다.

    원본 multi_cameras_multi_roi.py 는 frame_ready(영상)와
    temp_updated(CSV)를 따로 발행해서 영상 프레임 수와 CSV 행 수가
    어긋날 수 있었다. 여기서는 한 시그널로 묶어 항상 1:1 을 보장한다.
    """

    frame_ready      = pyqtSignal(object, object, str, list)  # rgb, raw, ts, samples
    resolution_known = pyqtSignal(int, int)
    reconnect_needed = pyqtSignal()

    def __init__(self, camera: TmCamera, roi_list_ref: list,
                 ip: str = "", query_w: int = 0, query_h: int = 0):
        super().__init__()
        self.camera       = camera
        self.roi_list_ref = roi_list_ref     # 공유 list[RoiItem]
        self.ip           = ip
        self.query_w      = query_w          # 0 이면 카메라 원본 해상도
        self.query_h      = query_h
        self._running     = False
        self._announced   = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        fail_count    = 0

        while self._running:
            try:
                frame = self.camera.query_frame(self.query_w, self.query_h)
                if frame is None:
                    QThread.msleep(10)
                    continue

                fail_count = 0
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                fw = frame.width()
                fh = frame.height()

                if not self._announced:
                    self._announced = True
                    self.resolution_known.emit(fw, fh)

                # ── RGB 프리뷰 ──
                bitmap = frame.to_bitmap(ColorOrder.COLOR_RGB)
                img = np.frombuffer(bitmap, dtype=np.uint8).reshape(
                    fh, fw, 3).copy()

                # ── Raw Y16 추출 ──
                raw = None
                pixel_2d = frame.get_pixel(0, 0, fw, fh)
                if pixel_2d is not None:
                    raw = np.array(pixel_2d, dtype=np.uint16)
                    # get_pixel 은 [w][h] 로 반환 → [h][w] 로 전치
                    if raw.shape == (fw, fh):
                        raw = raw.T

                # ── ROI 온도 측정 ──
                samples = []
                if self.camera.get_camera_format() == "Y16":
                    samples = self._measure_rois(frame, list(self.roi_list_ref))

                self.frame_ready.emit(img, raw, ts, samples)
                del frame

            except Exception as e:
                fail_count += 1
                logger.error(
                    f"[{self.ip}] FrameWorker 오류 "
                    f"({fail_count}/{_RECONNECT_THRESHOLD}): {e}"
                )
                if fail_count >= _RECONNECT_THRESHOLD:
                    logger.warning(
                        f"[{self.ip}] 연속 오류 {_RECONNECT_THRESHOLD}회 — 재연결 요청"
                    )
                    self._running = False
                    self.reconnect_needed.emit()
                    return
                QThread.msleep(100)

    def _measure_rois(self, frame, roi_list: list) -> list:
        """이 프레임의 ROI 온도를 측정해 값 스냅샷 리스트로 반환한다."""
        if not roi_list:
            return []

        roi_mgr = TmRoiManager()
        for ri in roi_list:
            if ri.roi_type == RoiType.Spot:
                roi_mgr.add_item_xy(RoiType.Spot, ri.x1, ri.y1)
            else:
                roi_mgr.add_item_xywh(ri.roi_type, ri.x1, ri.y1, ri.x2, ri.y2)

        samples = []
        for idx, ri in enumerate(roi_list):
            item = roi_mgr.get_roi_item(idx)
            frame.do_measure(item)

            if ri.roi_type == RoiType.Spot:
                spot_item = roi_mgr.get_roi_spot_item(idx)
                t = self.camera.get_temperature(spot_item.get_roi_maxloc().value)
                mn = av = mx = t
            else:
                if ri.roi_type == RoiType.Line:
                    sub = roi_mgr.get_roi_line_item(idx)
                elif ri.roi_type == RoiType.Rect:
                    sub = roi_mgr.get_roi_rect_item(idx)
                else:
                    sub = roi_mgr.get_roi_ellipse_item(idx)
                mn = self.camera.get_temperature(sub.get_roi_minloc().value)
                av = self.camera.get_temperature(sub.get_roi_avgloc().value)
                mx = self.camera.get_temperature(sub.get_roi_maxloc().value)

            samples.append(
                RoiSample(ri.roi_type, ri.geometry_str(), mn, av, mx))
        return samples


# ─────────────────────────────────────────────────────────────
# Raw Y16 파일 기록기
# ─────────────────────────────────────────────────────────────
class RawY16Writer:
    """Raw Y16 프레임을 비압축 바이너리로 기록한다.

    파일 분할은 CameraPanel 이 CSV/AVI 와 함께 일괄 수행하므로
    이 클래스는 단일 파일만 담당한다.
    """

    def __init__(self, base_path: str, width: int, height: int, meta_extra: dict):
        self.base_path   = base_path
        self.width       = width
        self.height      = height
        self.meta_extra  = meta_extra
        self.frame_count = 0
        self.timestamps  = []
        self.raw_min     = None
        self.raw_max     = None
        self._path       = base_path + ".y16raw"
        self._file       = open(self._path, "wb")
        logger.info(f"Raw Y16 파일 생성: {self._path}")

    @property
    def path(self) -> str:
        return self._path

    def write_frame(self, raw: np.ndarray, timestamp: str):
        if self._file is None:
            return
        if raw.shape != (self.height, self.width):
            logger.error(
                f"프레임 크기 불일치: {raw.shape} != "
                f"{(self.height, self.width)} — 이 프레임을 건너뜁니다")
            return

        self._file.write(np.ascontiguousarray(raw, dtype="<u2").tobytes())
        self.timestamps.append(timestamp)
        self.frame_count += 1

        lo = int(raw.min())
        hi = int(raw.max())
        self.raw_min = lo if self.raw_min is None else min(self.raw_min, lo)
        self.raw_max = hi if self.raw_max is None else max(self.raw_max, hi)

        if self.frame_count % _FLUSH_EVERY == 0:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except OSError as e:
                logger.error(f"Y16 flush 오류: {e}")

    def size_bytes(self) -> int:
        try:
            return os.path.getsize(self._path)
        except OSError:
            return 0

    def close(self):
        if self._file:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except OSError as e:
                logger.error(f"Y16 fsync 오류: {e}")
            self._file.close()
            self._file = None

        meta = {
            "format": "Y16_RAW_UINT16_LE",
            "width": self.width,
            "height": self.height,
            "bytes_per_pixel": 2,
            "bytes_per_frame": self.width * self.height * 2,
            "frame_count": self.frame_count,
            "fps": RECORD_FPS,
            "raw_value_min": self.raw_min,
            "raw_value_max": self.raw_max,
            "coordinate_space": (
                "ROI 좌표와 프레임 화소는 모두 width x height 기준입니다."),
            "read_example": (
                "import numpy as np; "
                "a = np.fromfile(p, dtype='<u2').reshape(-1, height, width)"),
        }
        meta.update(self.meta_extra)
        meta["timestamps"] = self.timestamps

        meta_path = self.base_path + ".y16meta"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            logger.info(
                f"메타데이터 저장: {meta_path}  "
                f"({self.frame_count} frames, {self.width}x{self.height})")
        except OSError as e:
            logger.error(f"메타데이터 저장 실패: {e}")


# ─────────────────────────────────────────────────────────────
# 미리보기 라벨 (복수 ROI 드래그, 표시 좌표 ↔ 프레임 좌표 변환)
# ─────────────────────────────────────────────────────────────
class PreviewLabel(QLabel):
    """카메라 원본 해상도가 160x120 이어도 화면에는 확대해 보여주고,
    드래그한 ROI 좌표는 원본 화소 좌표로 되돌려 emit 한다."""

    roi_added = pyqtSignal(int, int, int, int)   # 프레임 좌표계 x1, y1, x2, y2

    def __init__(self):
        super().__init__()
        self._disp_w = 640
        self._disp_h = DISPLAY_H
        self.setFixedSize(self._disp_w, self._disp_h)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #1a1a1a; color: #888; font-size: 14px;")
        self.setText("연결 중...")

        self._drag_start   = None
        self._drag_end     = None
        self._roi_list     = []       # list[RoiItem] 참조
        self._current_type = RoiType.Rect
        self._frame_w      = 0        # 0 = 아직 첫 프레임 수신 전
        self._frame_h      = 0

    # ── 해상도 설정 ───────────────────────────────────────────
    def set_frame_size(self, fw: int, fh: int):
        """카메라 실제 해상도를 받아 표시 크기를 종횡비에 맞춘다."""
        if fw <= 0 or fh <= 0 or (fw, fh) == (self._frame_w, self._frame_h):
            return
        self._frame_w, self._frame_h = fw, fh
        disp_w = int(round(DISPLAY_H * fw / fh))
        if disp_w > DISPLAY_W_MAX:
            disp_w = DISPLAY_W_MAX
            self._disp_h = int(round(DISPLAY_W_MAX * fh / fw))
        else:
            self._disp_h = DISPLAY_H
        self._disp_w = disp_w
        self.setFixedSize(self._disp_w, self._disp_h)

    def display_size(self):
        return self._disp_w, self._disp_h

    def _to_frame(self, x: int, y: int):
        """표시 좌표 → 프레임(원본 화소) 좌표."""
        if self._frame_w <= 0:
            return x, y
        fx = int(round(x * self._frame_w / self._disp_w))
        fy = int(round(y * self._frame_h / self._disp_h))
        fx = max(0, min(self._frame_w - 1, fx))
        fy = max(0, min(self._frame_h - 1, fy))
        return fx, fy

    def _to_display(self, x: int, y: int):
        """프레임 좌표 → 표시 좌표."""
        if self._frame_w <= 0:
            return x, y
        return (int(round(x * self._disp_w / self._frame_w)),
                int(round(y * self._disp_h / self._frame_h)))

    def _scale_len(self, w: int, h: int):
        if self._frame_w <= 0:
            return w, h
        return (max(1, int(round(w * self._disp_w / self._frame_w))),
                max(1, int(round(h * self._disp_h / self._frame_h))))

    # ── 설정 ──────────────────────────────────────────────────
    def set_roi_list(self, roi_list):
        self._roi_list = roi_list

    def set_roi_type(self, roi_type: RoiType):
        self._current_type = roi_type

    # ── 마우스 ────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_start = e.pos()
            self._drag_end   = e.pos()

    def mouseMoveEvent(self, e):
        if self._drag_start:
            self._drag_end = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._drag_start:
            self._drag_end = e.pos()
            x1, y1 = self._to_frame(self._drag_start.x(), self._drag_start.y())
            x2, y2 = self._to_frame(self._drag_end.x(), self._drag_end.y())
            self._drag_start = None
            self._drag_end   = None
            self.roi_added.emit(x1, y1, x2, y2)
            self.update()

    # ── 그리기 ────────────────────────────────────────────────
    def set_frame(self, img: np.ndarray):
        h, w, _ = img.shape
        self.set_frame_size(w, h)

        if (w, h) != (self._disp_w, self._disp_h):
            img = cv2.resize(img, (self._disp_w, self._disp_h),
                             interpolation=DISPLAY_INTERP)
        img = np.ascontiguousarray(img)
        dh, dw, _ = img.shape

        qimg = QImage(img.data, dw, dh, 3 * dw, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg.copy())

        painter = QPainter(pix)
        for idx, ri in enumerate(self._roi_list):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            painter.setPen(QPen(color, 2))
            self._draw_roi(painter, ri, idx, color)

        if self._drag_start and self._drag_end:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine))
            self._draw_drag_preview(painter)

        painter.end()
        self.setPixmap(pix)

    def _draw_roi(self, painter: QPainter, ri: RoiItem, idx: int, color: QColor):
        dx, dy = self._to_display(ri.x1, ri.y1)

        if ri.roi_type == RoiType.Spot:
            painter.drawLine(dx - 6, dy, dx + 6, dy)
            painter.drawLine(dx, dy - 6, dx, dy + 6)
        elif ri.roi_type == RoiType.Line:
            ex, ey = self._to_display(ri.x2, ri.y2)
            painter.drawLine(dx, dy, ex, ey)
        else:
            dw, dh = self._scale_len(ri.x2, ri.y2)
            r = QRect(dx, dy, dw, dh)
            if ri.roi_type == RoiType.Rect:
                painter.drawRect(r)
            else:
                painter.drawEllipse(r)

        painter.setPen(QPen(color))
        painter.setFont(QFont("Consolas", 9))
        if ri.roi_type == RoiType.Line:
            ex, ey = self._to_display(ri.x2, ri.y2)
            lx, ly = min(dx, ex), min(dy, ey) - 4
        else:
            lx, ly = dx, dy - 4
        painter.drawText(lx, max(ly, 12), f"ROI{idx}")

    def _draw_drag_preview(self, painter: QPainter):
        x1, y1 = self._drag_start.x(), self._drag_start.y()
        x2, y2 = self._drag_end.x(), self._drag_end.y()
        if self._current_type == RoiType.Spot:
            painter.drawLine(x2 - 6, y2, x2 + 6, y2)
            painter.drawLine(x2, y2 - 6, x2, y2 + 6)
        elif self._current_type == RoiType.Line:
            painter.drawLine(x1, y1, x2, y2)
        else:
            r = QRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            if self._current_type == RoiType.Rect:
                painter.drawRect(r)
            else:
                painter.drawEllipse(r)


# ─────────────────────────────────────────────────────────────
# 카메라 1대 패널
# ─────────────────────────────────────────────────────────────
class CameraPanel(QGroupBox):
    def __init__(self, title: str, global_dir_ref: list):
        super().__init__(title)
        self.global_dir_ref = global_dir_ref
        self._local_dir     = ""
        self.camera         = None
        self.worker         = None
        self.roi_list       = []          # list[RoiItem]

        self._recording         = False
        self._y16_writer        = None
        self._video_writer      = None
        self._csv_file          = None
        self._csv_writer        = None
        self._current_base      = ""
        self._ip                = ""
        self._label             = ""
        self._split_start_time  = None
        self._split_frame_count = 0
        self._flush_count       = 0
        self._rec_frames        = 0
        self._rec_start         = None
        self._was_recording     = False
        self._csv_roi_count     = 0

        self._frame_w           = 0
        self._frame_h           = 0
        self._temp_lut          = None    # {"stride":.., "values":[..]}
        self._cam_meta          = {}

        self._reconnect_scan    = None
        self._reconnect_worker  = None

        self._current_roi_type  = RoiType.Rect

        self._build_ui()

    def _eff_dir(self):
        return self._local_dir if self._local_dir else self.global_dir_ref[0]

    # ── UI ────────────────────────────────────────────────────
    def _build_ui(self):
        font_main = QFont()
        font_main.setPointSize(BASE_FONT_SIZE)

        self.preview = PreviewLabel()
        self.preview.set_roi_list(self.roi_list)
        self.preview.roi_added.connect(self._on_roi_added)

        # 해상도 / 용량 정보
        self.lbl_res = QLabel("해상도: 대기 중")
        self.lbl_res.setStyleSheet(f"font-size: {BASE_FONT_SIZE}px; color: #7fb3d5;")

        # 저장 폴더 행
        lbl_dir = QLabel("저장 폴더:")
        lbl_dir.setFont(font_main)
        self.input_dir = QLineEdit()
        self.input_dir.setPlaceholderText("비워두면 전역 폴더 사용")
        self.input_dir.setFixedHeight(44)
        self.input_dir.setFont(font_main)
        self.input_dir.textChanged.connect(
            lambda t: setattr(self, "_local_dir", t.strip()))

        btn_browse_local = QPushButton("탐색...")
        btn_browse_local.setFixedSize(90, 44)
        btn_browse_local.setFont(font_main)
        btn_browse_local.clicked.connect(self._browse_local)

        dir_row = QHBoxLayout()
        dir_row.addWidget(lbl_dir)
        dir_row.addWidget(self.input_dir)
        dir_row.addWidget(btn_browse_local)

        # ROI 타입 선택 행
        lbl_roi_type = QLabel("ROI 타입:")
        lbl_roi_type.setFont(font_main)
        self.combo_roi_type = QComboBox()
        self.combo_roi_type.setFixedHeight(44)
        self.combo_roi_type.setFont(font_main)
        self.combo_roi_type.addItem("Spot (점)",      RoiType.Spot)
        self.combo_roi_type.addItem("Line (선)",      RoiType.Line)
        self.combo_roi_type.addItem("Rect (사각형)",  RoiType.Rect)
        self.combo_roi_type.addItem("Ellipse (타원)", RoiType.Ellipse)
        self.combo_roi_type.setCurrentIndex(2)  # 기본: Rect
        self.combo_roi_type.currentIndexChanged.connect(self._on_roi_type_changed)

        roi_type_row = QHBoxLayout()
        roi_type_row.addWidget(lbl_roi_type)
        roi_type_row.addWidget(self.combo_roi_type, stretch=1)

        self.lbl_roi = QLabel("ROI: 없음  (타입 선택 후 드래그로 추가)")
        self.lbl_roi.setStyleSheet(f"font-size: {BASE_FONT_SIZE}px; color: #aaa;")
        self.lbl_roi.setWordWrap(True)

        self.lbl_temp = QLabel("온도 데이터 없음")
        self.lbl_temp.setFont(QFont("Consolas", BASE_FONT_SIZE))
        self.lbl_temp.setAlignment(Qt.AlignLeft)
        self.lbl_temp.setWordWrap(True)
        self.lbl_temp.setMinimumHeight(40)

        self.lbl_rec = QLabel("")
        self.lbl_rec.setStyleSheet(f"font-size: {BASE_FONT_SIZE}px; color: #aaa;")

        # 버튼 행
        self.btn_record = QPushButton("● 녹화 시작")
        self.btn_record.setFixedHeight(BTN_H_PANEL)
        font_rec = QFont()
        font_rec.setPointSize(BASE_FONT_SIZE + 1)
        font_rec.setBold(True)
        self.btn_record.setFont(font_rec)
        self.btn_record.setEnabled(False)
        self._set_btn_idle()
        self.btn_record.clicked.connect(self.toggle_record)

        self.btn_undo_roi = QPushButton("마지막 ROI 삭제")
        self.btn_undo_roi.setFixedHeight(BTN_H_PANEL)
        self.btn_undo_roi.setFont(font_main)
        self.btn_undo_roi.clicked.connect(self._remove_last_roi)

        self.btn_clear = QPushButton("ROI 전체 초기화")
        self.btn_clear.setFixedHeight(BTN_H_PANEL)
        self.btn_clear.setFont(font_main)
        self.btn_clear.clicked.connect(self._clear_roi)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_record, stretch=2)
        btn_row.addWidget(self.btn_undo_roi, stretch=1)
        btn_row.addWidget(self.btn_clear, stretch=1)

        layout = QVBoxLayout()
        layout.addWidget(self.preview)
        layout.addWidget(self.lbl_res)
        layout.addLayout(dir_row)
        layout.addLayout(roi_type_row)
        layout.addWidget(self.lbl_roi)
        layout.addWidget(self.lbl_temp)
        layout.addWidget(self.lbl_rec)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def _set_btn_idle(self):
        self.btn_record.setText("● 녹화 시작")
        self.btn_record.setStyleSheet(
            "QPushButton { background-color: #2d6a2d; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a8a3a; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )

    def _set_btn_recording(self):
        self.btn_record.setText("■ 녹화 정지")
        self.btn_record.setStyleSheet(
            "QPushButton { background-color: #8b1a1a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #b52020; }"
        )

    def _browse_local(self):
        path = QFileDialog.getExistingDirectory(
            self, "이 카메라의 저장 폴더 선택", self._eff_dir())
        if path:
            self.input_dir.setText(path)

    def _on_roi_type_changed(self, index):
        roi_type = self.combo_roi_type.itemData(index)
        self._current_roi_type = roi_type
        self.preview.set_roi_type(roi_type)

    # ── 카메라 연결 완료 ──────────────────────────────────────
    def attach_camera(self, camera: TmCamera, ip: str, label: str = ""):
        self.camera = camera
        self._ip    = ip
        if label:
            self._label = label
        self.camera.set_temp_unit(TempUnit.CELSIUS)
        self.camera.set_color_map(ColormapTypes.Inferno + 1)

        cam_fmt = self.camera.get_camera_format()
        if cam_fmt != "Y16":
            logger.warning(
                f"[{ip}] 카메라 포맷이 '{cam_fmt}' 입니다. "
                "Raw Y16 저장과 ROI 온도 측정은 Y16 에서만 유효합니다.")

        self._cam_meta = {
            "camera_ip": ip,
            "camera_label": self._label,
            "camera_format": cam_fmt,
            "temp_unit": "Celsius",
            "sdk_api_version": self._safe(lambda: self.camera.get_api_version()),
        }

        if self._temp_lut is None:
            self._temp_lut = self._build_temp_lut()

        self.btn_record.setEnabled(True)
        self.preview.setText("")
        self._start_worker()

        if self._was_recording:
            self._was_recording = False
            logger.info(f"[{self._ip}] 재연결 완료 — 녹화 자동 재개")
            self._start_recording()

    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except Exception:
            return None

    def _build_temp_lut(self):
        """raw → 온도 변환표를 만든다.

        .y16meta 에 함께 저장해 두면 나중에 카메라나 TmSDK 없이도
        .y16raw 를 온도로 복원할 수 있다. 표 사이 값은 선형 보간한다.
        """
        if self.camera is None:
            return None
        t0 = time.time()
        values = []
        try:
            for raw in range(0, 65536, _LUT_STRIDE):
                values.append(round(float(self.camera.get_temperature(raw)), 4))
                if time.time() - t0 > _LUT_TIME_BUDGET:
                    logger.warning(
                        f"[{self._ip}] LUT 생성 시간 초과 — "
                        f"raw 0~{raw} 구간까지만 기록합니다")
                    break
        except Exception as e:
            logger.error(f"[{self._ip}] LUT 생성 실패: {e}")
            return None

        logger.info(
            f"[{self._ip}] raw→온도 LUT 생성 완료: {len(values)} 포인트, "
            f"{time.time() - t0:.2f}초")
        return {
            "stride": _LUT_STRIDE,
            "raw_start": 0,
            "count": len(values),
            "unit": "Celsius",
            "note": ("temperature = interp(raw, "
                     "raw_start + stride*arange(count), values)"),
            "values": values,
        }

    def set_status(self, text: str):
        self.preview.setText(text)
        self.lbl_temp.setText("온도 데이터 없음")

    # ── 프레임 워커 ───────────────────────────────────────────
    def _start_worker(self):
        qw, qh = (OPT_QUERY_SIZE if OPT_QUERY_SIZE else (0, 0))
        self.worker = FrameWorker(self.camera, self.roi_list, self._ip, qw, qh)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.resolution_known.connect(self._on_resolution)
        self.worker.reconnect_needed.connect(self._on_reconnect_needed)
        self.worker.start()

    def _on_resolution(self, fw: int, fh: int):
        self._frame_w, self._frame_h = fw, fh
        per_frame = fw * fh * 2
        per_hour  = per_frame * RECORD_FPS * 3600 / (1024 ** 3)
        self.lbl_res.setText(
            f"저장 해상도: {fw}x{fh}  |  프레임당 {per_frame/1024:.1f} KB  |  "
            f"연속 녹화 시 약 {per_hour:.2f} GB/시간")
        logger.info(f"[{self._ip}] 저장 해상도 확정: {fw}x{fh}")
        self._update_roi_label()

    # ── 프레임 수신: 프리뷰 + 녹화를 한 곳에서 처리 ───────────
    def _on_frame(self, img, raw, ts: str, samples: list):
        self.preview.set_frame(img)
        self._update_temp_label(samples)

        if not self._recording:
            return

        self._split_frame_count += 1
        if self._split_frame_count >= _SPLIT_CHECK_EVERY:
            self._split_frame_count = 0
            self._warn_if_lagging(ts)
            if self._should_rotate():
                self._rotate_recording()

        if raw is not None and self._y16_writer is not None:
            self._y16_writer.write_frame(raw, ts)
        if self._video_writer is not None:
            self._write_video_frame(img)
        self._write_csv_row(ts, samples)

        self._rec_frames += 1
        if self._rec_frames % 10 == 0:
            self._update_rec_label()

    def _warn_if_lagging(self, ts: str):
        """디스크가 못 따라가면 시그널 큐가 밀린다. 무인 운용 중 진단용."""
        try:
            captured = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return
        lag = (datetime.now() - captured).total_seconds()
        if lag > 2.0:
            logger.warning(
                f"[{self._ip}] 프레임 처리 지연 {lag:.1f}초 — "
                "디스크 쓰기 속도가 부족하거나 저장 해상도가 너무 큽니다")

    def _update_temp_label(self, samples: list):
        if not samples:
            return
        sym = self.camera.get_temp_unit_symbol() if self.camera else ""
        lines = []
        for idx, s in enumerate(samples):
            hex_color = ROI_COLORS[idx % len(ROI_COLORS)].name()
            lines.append(
                f"<span style='color:{hex_color}'>"
                f"ROI{idx}({ROI_TYPE_NAMES[s.roi_type]}): "
                f"Min={s.min_temp:.2f} Avg={s.avg_temp:.2f} "
                f"Max={s.max_temp:.2f} {sym}</span>"
            )
        self.lbl_temp.setText("<br>".join(lines))

    def _update_rec_label(self):
        elapsed = (datetime.now() - self._rec_start).total_seconds()
        mb = self._rec_frames * self._frame_w * self._frame_h * 2 / (1024 ** 2)
        self.lbl_rec.setText(
            f"녹화 중: {self._rec_frames} frames  |  {elapsed:.0f}초  |  "
            f"Y16 ~{mb:.0f} MB")

    # ── ROI 추가/삭제 ─────────────────────────────────────────
    def _roi_locked(self) -> bool:
        """녹화 중 ROI 를 바꾸면 CSV 헤더와 행의 열 수가 어긋난다."""
        if not self._recording:
            return False
        self._flash_roi_warning(
            "⚠ 녹화 중에는 ROI 를 바꿀 수 없습니다 — 먼저 녹화를 정지하세요")
        return True

    def _flash_roi_warning(self, text: str):
        self.lbl_roi.setText(text)
        self.lbl_roi.setStyleSheet(
            f"font-size: {BASE_FONT_SIZE}px; color: #f0c040;")
        QTimer.singleShot(2500, self._update_roi_label)

    def _on_roi_added(self, x1, y1, x2, y2):
        if self._roi_locked():
            return
        roi_type = self._current_roi_type
        if roi_type == RoiType.Spot:
            ri = RoiItem(RoiType.Spot, x2, y2, x2, y2)
        elif roi_type == RoiType.Line:
            if x1 == x2 and y1 == y2:
                return
            ri = RoiItem(RoiType.Line, x1, y1, x2, y2)
        else:
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            if rw < 2 or rh < 2:
                # 모달 대화상자를 띄우면 이벤트 루프가 멈춰 녹화 중 프레임이
                # 밀리므로, 라벨로만 알린다.
                self._flash_roi_warning(
                    f"⚠ ROI 가 너무 작습니다 — 원본 화소 기준 {rw}x{rh}. "
                    "최소 2x2 이상으로 드래그하세요 "
                    "(열화상은 대상이 3x3 화소는 덮어야 정확합니다)")
                return
            ri = RoiItem(roi_type, rx, ry, rw, rh)

        self.roi_list.append(ri)
        self._update_roi_label()

    def _remove_last_roi(self):
        if self._roi_locked():
            return
        if self.roi_list:
            self.roi_list.pop()
            self._update_roi_label()

    def _clear_roi(self):
        if self._roi_locked():
            return
        self.roi_list.clear()
        self._update_roi_label()
        self.lbl_temp.setText("온도 데이터 없음")

    def _update_roi_label(self):
        self.lbl_roi.setStyleSheet(
            f"font-size: {BASE_FONT_SIZE}px; color: #aaa;")
        if not self.roi_list:
            self.lbl_roi.setText("ROI: 없음  (타입 선택 후 드래그로 추가)")
            return
        parts = []
        for idx, ri in enumerate(self.roi_list):
            npx = ri.pixel_count()
            warn = "  ⚠ 화소 부족" if npx < 9 else ""
            parts.append(
                f"ROI{idx}:{ROI_TYPE_NAMES[ri.roi_type]}"
                f"{ri.geometry_str()} ≈{npx:.0f}px{warn}")
        self.lbl_roi.setText("  |  ".join(parts))

    # ── 녹화 ──────────────────────────────────────────────────
    def toggle_record(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _build_csv_header(self):
        self._csv_roi_count = len(self.roi_list)
        header = ["timestamp"]
        if not self.roi_list:
            header.extend(["min_temp", "avg_temp", "max_temp"])
        else:
            for idx, ri in enumerate(self.roi_list):
                prefix = f"roi{idx}_{ROI_TYPE_NAMES[ri.roi_type].lower()}"
                header.extend([
                    f"{prefix}_params",
                    f"{prefix}_min", f"{prefix}_avg", f"{prefix}_max"
                ])
        return header

    def _open_writers(self):
        """.y16raw / .csv / .avi 를 같은 base 이름으로 함께 연다."""
        if self._frame_w <= 0 or self._frame_h <= 0:
            logger.error(f"[{self._ip}] 해상도 미확정 — 녹화를 시작할 수 없습니다")
            return False

        out_dir = self._eff_dir()
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"[{self._ip}] 저장 폴더 생성 실패 ({out_dir}): {e}")
            return False

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"{self._ip.replace('.','_')}_{ts}")
        self._current_base = base

        meta_extra = dict(self._cam_meta)
        meta_extra["recording_started"] = datetime.now().isoformat(timespec="seconds")
        meta_extra["rois"] = [ri.to_dict() for ri in self.roi_list]
        if self._temp_lut:
            meta_extra["raw_to_temperature_lut"] = self._temp_lut

        try:
            self._y16_writer = RawY16Writer(
                base, self._frame_w, self._frame_h, meta_extra)
        except OSError as e:
            logger.error(f"[{self._ip}] Y16 파일 생성 실패: {e}")
            return False

        try:
            self._csv_file   = open(base + ".csv", "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self._build_csv_header())
        except OSError as e:
            logger.error(f"[{self._ip}] CSV 생성 실패: {e}")
            self._y16_writer.close()
            self._y16_writer = None
            return False

        if OPT_WRITE_AVI:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            vw = cv2.VideoWriter(base + ".avi", fourcc, RECORD_FPS,
                                 (self._frame_w, self._frame_h))
            if vw.isOpened():
                self._video_writer = vw
            else:
                logger.error(f"[{self._ip}] AVI 생성 실패 — Y16/CSV 만 기록합니다")
                self._video_writer = None

        self._split_start_time  = datetime.now()
        self._split_frame_count = 0
        self._flush_count       = 0
        return True

    def _start_recording(self):
        if not self._open_writers():
            return
        self._recording  = True
        self._rec_frames = 0
        self._rec_start  = datetime.now()
        self._set_btn_recording()
        self.lbl_rec.setStyleSheet(f"color: #ff4444; font-size: {BASE_FONT_SIZE}px;")
        self.lbl_rec.setText("녹화 중: 0 frames")
        logger.info(f"[{self._ip}] 녹화 시작: {self._current_base}.*")

    def _stop_recording(self):
        self._recording = False
        self._close_writers()
        self._set_btn_idle()
        self.lbl_rec.setStyleSheet(f"color: #aaa; font-size: {BASE_FONT_SIZE}px;")
        if self._rec_start:
            elapsed = (datetime.now() - self._rec_start).total_seconds()
            mb = self._rec_frames * self._frame_w * self._frame_h * 2 / (1024 ** 2)
            self.lbl_rec.setText(
                f"녹화 완료: {self._rec_frames} frames  |  "
                f"{elapsed:.0f}초  |  Y16 ~{mb:.0f} MB")
        logger.info(f"[{self._ip}] 녹화 정지 ({self._rec_frames} frames)")

    def _close_writers(self):
        if self._y16_writer:
            self._y16_writer.close()
            self._y16_writer = None
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._csv_file:
            try:
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
            except OSError as e:
                logger.error(f"[{self._ip}] CSV fsync 오류: {e}")
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

    # ── 파일 자동 분할 ────────────────────────────────────────
    def _should_rotate(self) -> bool:
        if self._split_start_time is None:
            return False
        elapsed = (datetime.now() - self._split_start_time).total_seconds()
        if elapsed >= _SPLIT_MAX_SECS:
            return True
        size = self._y16_writer.size_bytes() if self._y16_writer else 0
        if self._video_writer is not None:
            try:
                size += os.path.getsize(self._current_base + ".avi")
            except OSError:
                pass
        return size >= _SPLIT_MAX_BYTES

    def _rotate_recording(self):
        logger.info(f"[{self._ip}] 파일 분할 — 새 파일로 전환합니다")
        self._close_writers()
        if not self._open_writers():
            logger.error(f"[{self._ip}] 분할 후 파일 재생성 실패 — 녹화를 정지합니다")
            self._recording = False
            self._set_btn_idle()
            return
        logger.info(f"[{self._ip}] 새 녹화 파일: {self._current_base}.*")

    # ── 기록 ──────────────────────────────────────────────────
    def _write_video_frame(self, img: np.ndarray):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for idx, ri in enumerate(self.roi_list):
            cq = ROI_COLORS[idx % len(ROI_COLORS)]
            color = (cq.blue(), cq.green(), cq.red())
            if ri.roi_type == RoiType.Spot:
                cv2.drawMarker(bgr, (ri.x1, ri.y1), color, cv2.MARKER_CROSS, 6, 1)
            elif ri.roi_type == RoiType.Line:
                cv2.line(bgr, (ri.x1, ri.y1), (ri.x2, ri.y2), color, 1)
            elif ri.roi_type == RoiType.Rect:
                cv2.rectangle(bgr, (ri.x1, ri.y1),
                              (ri.x1 + ri.x2, ri.y1 + ri.y2), color, 1)
            elif ri.roi_type == RoiType.Ellipse:
                cv2.ellipse(bgr,
                            (ri.x1 + ri.x2 // 2, ri.y1 + ri.y2 // 2),
                            (max(1, ri.x2 // 2), max(1, ri.y2 // 2)),
                            0, 0, 360, color, 1)
        self._video_writer.write(bgr)

    def _write_csv_row(self, ts: str, samples: list):
        if self._csv_writer is None:
            return
        # 헤더는 녹화 시작 시점의 ROI 개수로 고정되어 있다. 열 수가 어긋나면
        # 파일 전체가 못 쓰게 되므로 맞춰서 채우거나 잘라낸다.
        n = self._csv_roi_count
        row = [ts]
        if n == 0:
            row.extend(["0.00", "0.00", "0.00"])
        else:
            for i in range(n):
                if i < len(samples):
                    s = samples[i]
                    row.append(s.geometry)
                    row.extend([f"{s.min_temp:.2f}",
                                f"{s.avg_temp:.2f}",
                                f"{s.max_temp:.2f}"])
                else:
                    row.extend(["", "", "", ""])
        self._csv_writer.writerow(row)

        self._flush_count += 1
        if self._flush_count >= _FLUSH_EVERY:
            self._flush_count = 0
            try:
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
            except OSError as e:
                logger.error(f"[{self._ip}] CSV flush 오류: {e}")

    # ── 카메라 재연결 ─────────────────────────────────────────
    def _on_reconnect_needed(self):
        logger.warning(f"[{self._ip}] 연결 끊김 감지 — 재연결 시작")
        self._was_recording = self._recording

        if self._recording:
            self._recording = False
            self._close_writers()

        self.worker = None

        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                logger.debug(f"[{self._ip}] camera.close() 오류: {e}")
            self.camera = None

        self.btn_record.setEnabled(False)
        self._set_btn_idle()
        self.set_status(f"재연결 중...\n{self._ip}")

        self._reconnect_scan = ScanWorker()
        self._reconnect_scan.scan_done.connect(self._on_reconnect_scan_done)
        self._reconnect_scan.start()

    def _on_reconnect_scan_done(self, cam_map: dict):
        cam_info = cam_map.get(self._ip)
        self._reconnect_worker = ConnectWorker(self._ip, self._label, cam_info)
        self._reconnect_worker.connected.connect(
            lambda cam, lbl: self.attach_camera(cam, self._ip, lbl)
        )
        self._reconnect_worker.failed.connect(self._on_reconnect_failed)
        self._reconnect_worker.start()

    def _on_reconnect_failed(self, ip: str):
        logger.error(f"[{ip}] 재연결 최종 실패 — 수동 재시도 필요")
        self._was_recording = False
        self.setTitle(f"재연결 실패  [{ip}]")
        self.set_status(f"재연결 실패\n{ip}\n(IP 입력창에서 다시 연결하세요)")

    # ── 정리 ──────────────────────────────────────────────────
    def cleanup(self):
        if self._recording:
            self._stop_recording()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        if self.camera:
            try:
                self.camera.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# 메인 윈도우
# ─────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TmSDK - Multi Camera Multi-ROI  ·  Raw Y16 Recorder")

        self.global_dir_ref = [DEFAULT_OUTPUT_DIR]
        self.panels: list = []
        self.connect_workers: list = []
        self._scan_worker  = None
        self._pending_cfgs = []

        # ── 예약 녹화 상태 ──
        self._schedule_active    = False
        self._schedule_running   = False
        self._sched_start_hour   = 8
        self._sched_stop_hour    = 19
        self._sched_use_repeat   = False
        self._sched_interval     = 60
        self._sched_rec_duration = 10

        self._build_ui()
        self._start_initial_scan()

        self._schedule_timer = QTimer(self)
        self._schedule_timer.setInterval(1000)
        self._schedule_timer.timeout.connect(self._on_schedule_tick)
        self._schedule_timer.start()

    def _build_ui(self):
        font = QFont()
        font.setPointSize(BASE_FONT_SIZE)

        # 행 1: 전역 저장 폴더 + 전체 녹화/정지
        lbl_dir = QLabel("전역 저장 폴더:")
        lbl_dir.setFont(font)

        self.input_dir = QLineEdit(DEFAULT_OUTPUT_DIR)
        self.input_dir.setFixedHeight(BTN_H_TOP)
        self.input_dir.setFixedWidth(360)
        self.input_dir.setFont(font)
        self.input_dir.textChanged.connect(
            lambda t: self.global_dir_ref.__setitem__(
                0, t.strip() or DEFAULT_OUTPUT_DIR)
        )

        btn_browse = QPushButton("탐색...")
        btn_browse.setFixedSize(100, BTN_H_TOP)
        btn_browse.setFont(font)
        btn_browse.clicked.connect(self._browse_global)

        self.btn_record_all = QPushButton("● 전체 녹화")
        self.btn_record_all.setFixedSize(200, BTN_H_TOP)
        self.btn_record_all.setFont(font)
        self.btn_record_all.clicked.connect(self._record_all)

        self.btn_stop_all = QPushButton("■ 전체 정지")
        self.btn_stop_all.setFixedSize(200, BTN_H_TOP)
        self.btn_stop_all.setFont(font)
        self.btn_stop_all.clicked.connect(self._stop_all)

        self.chk_avi = QCheckBox("참고용 AVI 함께 저장")
        self.chk_avi.setFont(font)
        self.chk_avi.setChecked(OPT_WRITE_AVI)
        self.chk_avi.setToolTip(
            "AVI 는 프레임별 auto-gain 과 손실압축이 적용되어\n"
            "절대온도 복원에 쓸 수 없습니다. 눈으로 확인하거나\n"
            "마스크를 그릴 때만 사용하세요.")
        self.chk_avi.toggled.connect(self._on_avi_toggled)

        row1 = QHBoxLayout()
        row1.addWidget(lbl_dir)
        row1.addWidget(self.input_dir)
        row1.addWidget(btn_browse)
        row1.addSpacing(30)
        row1.addWidget(self.btn_record_all)
        row1.addWidget(self.btn_stop_all)
        row1.addSpacing(20)
        row1.addWidget(self.chk_avi)
        row1.addStretch()

        # 행 2: 카메라 IP 입력 + 연결
        lbl_ip = QLabel("카메라 IP:")
        lbl_ip.setFont(font)

        self.input_ip = QLineEdit()
        self.input_ip.setPlaceholderText("예: 192.168.0.153")
        self.input_ip.setFixedHeight(BTN_H_TOP)
        self.input_ip.setFixedWidth(260)
        self.input_ip.setFont(font)
        self.input_ip.returnPressed.connect(self._connect_from_input)

        lbl_name = QLabel("이름:")
        lbl_name.setFont(font)

        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("예: Camera 3  (선택)")
        self.input_name.setFixedHeight(BTN_H_TOP)
        self.input_name.setFixedWidth(260)
        self.input_name.setFont(font)
        self.input_name.returnPressed.connect(self._connect_from_input)

        btn_connect = QPushButton("연결")
        btn_connect.setFixedSize(120, BTN_H_TOP)
        btn_connect.setFont(font)
        btn_connect.clicked.connect(self._connect_from_input)

        row2 = QHBoxLayout()
        row2.addWidget(lbl_ip)
        row2.addWidget(self.input_ip)
        row2.addWidget(lbl_name)
        row2.addWidget(self.input_name)
        row2.addWidget(btn_connect)
        row2.addStretch()

        # 행 3: 예약 녹화
        lbl_start = QLabel("시작 시각:")
        lbl_start.setFont(font)
        self.spin_start_hour = QSpinBox()
        self.spin_start_hour.setRange(0, 23)
        self.spin_start_hour.setValue(8)
        self.spin_start_hour.setSuffix(" 시")
        self.spin_start_hour.setFixedHeight(BTN_H_TOP)
        self.spin_start_hour.setFont(font)
        self.spin_start_hour.setEnabled(False)

        lbl_stop = QLabel("종료 시각:")
        lbl_stop.setFont(font)
        self.spin_stop_hour = QSpinBox()
        self.spin_stop_hour.setRange(1, 24)
        self.spin_stop_hour.setValue(19)
        self.spin_stop_hour.setSuffix(" 시")
        self.spin_stop_hour.setFixedHeight(BTN_H_TOP)
        self.spin_stop_hour.setFont(font)
        self.spin_stop_hour.setEnabled(False)

        self.chk_repeat = QCheckBox("반복 간격")
        self.chk_repeat.setFont(font)
        self.chk_repeat.setEnabled(False)
        self.chk_repeat.toggled.connect(self._on_repeat_toggled)

        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 1440)
        self.spin_interval.setValue(60)
        self.spin_interval.setSuffix(" 분")
        self.spin_interval.setFixedHeight(BTN_H_TOP)
        self.spin_interval.setFont(font)
        self.spin_interval.setEnabled(False)

        lbl_rec_dur = QLabel("녹화 시간:")
        lbl_rec_dur.setFont(font)
        self.spin_rec_duration = QSpinBox()
        self.spin_rec_duration.setRange(1, 1440)
        self.spin_rec_duration.setValue(10)
        self.spin_rec_duration.setSuffix(" 분")
        self.spin_rec_duration.setFixedHeight(BTN_H_TOP)
        self.spin_rec_duration.setFont(font)
        self.spin_rec_duration.setEnabled(False)
        self.lbl_rec_dur = lbl_rec_dur

        self.btn_schedule = QPushButton("예약 녹화 설정")
        self.btn_schedule.setFixedSize(200, BTN_H_TOP)
        self.btn_schedule.setFont(font)
        self.btn_schedule.setStyleSheet(
            "QPushButton { background-color: #1a5276; color: white;"
            " border-radius: 4px; }"
            "QPushButton:hover { background-color: #2471a3; }"
        )
        self.btn_schedule.clicked.connect(self._toggle_schedule)

        self.lbl_schedule_status = QLabel("예약: 비활성")
        self.lbl_schedule_status.setFont(font)
        self.lbl_schedule_status.setStyleSheet("color: #888;")

        row3 = QHBoxLayout()
        row3.addWidget(lbl_start)
        row3.addWidget(self.spin_start_hour)
        row3.addWidget(lbl_stop)
        row3.addWidget(self.spin_stop_hour)
        row3.addSpacing(15)
        row3.addWidget(self.chk_repeat)
        row3.addWidget(self.spin_interval)
        row3.addWidget(self.lbl_rec_dur)
        row3.addWidget(self.spin_rec_duration)
        row3.addSpacing(15)
        row3.addWidget(self.btn_schedule)
        row3.addWidget(self.lbl_schedule_status)
        row3.addStretch()

        # 카메라 패널 영역 (스크롤)
        self.panel_container = QWidget()
        self.panel_layout    = QHBoxLayout()
        self.panel_layout.setAlignment(Qt.AlignLeft)
        self.panel_container.setLayout(self.panel_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel_container)
        scroll.setMinimumHeight(DISPLAY_H + 360)

        main_layout = QVBoxLayout()
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addLayout(row3)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self.resize(1720, DISPLAY_H + 720)

    def _on_avi_toggled(self, checked: bool):
        global OPT_WRITE_AVI
        OPT_WRITE_AVI = checked
        logger.info(f"[설정] 참고용 AVI 저장: {'ON' if checked else 'OFF'} "
                    "(다음 녹화부터 적용)")

    # ── 초기 스캔 → 연결 ──────────────────────────────────────
    def _start_initial_scan(self):
        self._pending_cfgs = list(CAMERAS)
        self._scan_worker  = ScanWorker()
        self._scan_worker.scan_done.connect(self._on_initial_scan_done)
        self._scan_worker.start()

    def _on_initial_scan_done(self, cam_map: dict):
        if not self._pending_cfgs:
            for i, ip in enumerate(cam_map.keys()):
                self._launch_connect(ip, f"Camera {i+1}", cam_map)
            return
        for cfg in self._pending_cfgs:
            self._launch_connect(cfg["ip"], cfg.get("label", cfg["ip"]), cam_map)
        self._pending_cfgs = []

    def _connect_from_input(self):
        ip    = self.input_ip.text().strip()
        label = self.input_name.text().strip() or f"Camera ({ip})"
        if not ip:
            QMessageBox.warning(self, "입력 오류", "IP 주소를 입력해주세요.")
            return
        if ip in [p._ip for p in self.panels if p._ip]:
            QMessageBox.warning(self, "중복", f"{ip} 는 이미 추가되어 있습니다.")
            return
        self.input_ip.clear()
        self.input_name.clear()
        self._launch_connect_with_scan(ip, label)

    def _launch_connect(self, ip: str, label: str, cam_map: dict):
        panel = CameraPanel(f"{label}  [{ip}]  연결 중...", self.global_dir_ref)
        panel._label = label
        self.panels.append(panel)
        self.panel_layout.addWidget(panel)

        worker = ConnectWorker(ip, label, cam_map.get(ip))
        worker.connected.connect(lambda cam, lbl, p=panel, i=ip:
                                 self._on_connected(cam, lbl, p, i))
        worker.failed.connect(lambda i, p=panel: self._on_failed(i, p))
        worker.start()
        self.connect_workers.append(worker)

    def _launch_connect_with_scan(self, ip: str, label: str):
        panel = CameraPanel(f"{label}  [{ip}]  연결 중...", self.global_dir_ref)
        panel._label = label
        self.panels.append(panel)
        self.panel_layout.addWidget(panel)

        scan = ScanWorker()
        scan.scan_done.connect(lambda cam_map, i=ip, lbl=label, p=panel:
                               self._on_runtime_scan_done(i, lbl, p, cam_map))
        scan.start()
        self.connect_workers.append(scan)

    def _on_runtime_scan_done(self, ip: str, label: str, panel, cam_map: dict):
        worker = ConnectWorker(ip, label, cam_map.get(ip))
        worker.connected.connect(lambda cam, lbl, p=panel, i=ip:
                                 self._on_connected(cam, lbl, p, i))
        worker.failed.connect(lambda i, p=panel: self._on_failed(i, p))
        worker.start()
        self.connect_workers.append(worker)

    def _on_connected(self, cam: TmCamera, label: str, panel, ip: str):
        panel.setTitle(f"{label}  [{ip}]")
        panel.attach_camera(cam, ip, label)
        logger.info(f"[성공] {label} ({ip})")

    def _on_failed(self, ip: str, panel):
        panel.setTitle(f"연결 실패  [{ip}]")
        panel.set_status(f"연결 실패\n{ip}")
        logger.error(f"[실패] {ip} 연결 불가")

    def _browse_global(self):
        path = QFileDialog.getExistingDirectory(
            self, "전역 저장 폴더 선택", self.global_dir_ref[0])
        if path:
            self.input_dir.setText(path)

    def _record_all(self):
        for panel in self.panels:
            if panel.camera and not panel._recording:
                panel.toggle_record()

    def _stop_all(self):
        for panel in self.panels:
            if panel._recording:
                panel.toggle_record()

    # ── 예약 녹화 ─────────────────────────────────────────────
    def _on_repeat_toggled(self, checked):
        enabled = checked and self.chk_repeat.isEnabled()
        self.spin_interval.setEnabled(enabled)
        self.spin_rec_duration.setEnabled(enabled)
        self.lbl_rec_dur.setEnabled(enabled)

    def _toggle_schedule(self):
        if self._schedule_active:
            self._cancel_schedule()
        elif self.spin_start_hour.isEnabled():
            self._activate_schedule()
        else:
            self.spin_start_hour.setEnabled(True)
            self.spin_stop_hour.setEnabled(True)
            self.chk_repeat.setEnabled(True)
            self.spin_interval.setEnabled(self.chk_repeat.isChecked())
            self.spin_rec_duration.setEnabled(self.chk_repeat.isChecked())
            self.lbl_rec_dur.setEnabled(self.chk_repeat.isChecked())
            self.btn_schedule.setText("예약 시작")
            self.btn_schedule.setStyleSheet(
                "QPushButton { background-color: #1a7a1a; color: white;"
                " border-radius: 4px; }"
                "QPushButton:hover { background-color: #22a522; }"
            )
            self.lbl_schedule_status.setText(
                "시간대를 설정한 후 '예약 시작'을 누르세요")
            self.lbl_schedule_status.setStyleSheet(
                f"color: #f0c040; font-size: {BASE_FONT_SIZE}px;")

    def _activate_schedule(self):
        start_h = self.spin_start_hour.value()
        stop_h  = self.spin_stop_hour.value()

        if stop_h <= start_h:
            QMessageBox.warning(self, "시간 오류",
                                "종료 시각이 시작 시각보다 뒤여야 합니다.")
            return

        self._sched_start_hour   = start_h
        self._sched_stop_hour    = stop_h
        self._sched_use_repeat   = self.chk_repeat.isChecked()
        self._sched_interval     = (self.spin_interval.value()
                                    if self._sched_use_repeat else 0)
        self._sched_rec_duration = (self.spin_rec_duration.value()
                                    if self._sched_use_repeat else 0)

        if self._sched_use_repeat and self._sched_rec_duration > self._sched_interval:
            QMessageBox.warning(self, "설정 오류",
                                "녹화 시간은 반복 간격보다 작거나 같아야 합니다.")
            return

        self._schedule_active  = True
        self._schedule_running = False

        self._check_and_start_if_in_window()

        self.spin_start_hour.setEnabled(False)
        self.spin_stop_hour.setEnabled(False)
        self.chk_repeat.setEnabled(False)
        self.spin_interval.setEnabled(False)
        self.spin_rec_duration.setEnabled(False)
        self.btn_schedule.setText("예약 취소")
        self.btn_schedule.setStyleSheet(
            "QPushButton { background-color: #922b21; color: white;"
            " border-radius: 4px; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        self._update_schedule_label()

        repeat_str = ""
        if self._sched_use_repeat:
            repeat_str = (f"  반복: {self._sched_interval}분 간격"
                          f" / 녹화: {self._sched_rec_duration}분")
        logger.info(f"[예약] 활성화  매일 {start_h}시~{stop_h}시{repeat_str}")

    def _cancel_schedule(self):
        if self._schedule_running:
            self._schedule_stop_recording()

        self._schedule_active  = False
        self._schedule_running = False

        self.spin_start_hour.setEnabled(False)
        self.spin_stop_hour.setEnabled(False)
        self.chk_repeat.setEnabled(False)
        self.spin_interval.setEnabled(False)
        self.spin_rec_duration.setEnabled(False)
        self.btn_schedule.setText("예약 녹화 설정")
        self.btn_schedule.setStyleSheet(
            "QPushButton { background-color: #1a5276; color: white;"
            " border-radius: 4px; }"
            "QPushButton:hover { background-color: #2471a3; }"
        )
        self.lbl_schedule_status.setText("예약: 비활성")
        self.lbl_schedule_status.setStyleSheet("color: #888;")
        logger.info("[예약] 취소됨")

    def _get_today_window(self):
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        win_start = today + timedelta(hours=self._sched_start_hour)
        win_stop  = today + timedelta(hours=self._sched_stop_hour)
        return win_start, win_stop

    def _check_and_start_if_in_window(self):
        now = datetime.now()
        win_start, win_stop = self._get_today_window()
        if not (win_start <= now < win_stop):
            return
        if self._sched_use_repeat:
            slot_start, slot_stop = self._current_interval_slot(
                now, win_start, win_stop)
            if slot_start and slot_start <= now < slot_stop:
                self._schedule_start_recording()
        else:
            self._schedule_start_recording()

    def _current_interval_slot(self, now, win_start, win_stop):
        """반복 모드에서 현재 시각이 속한 녹화 슬롯(start, stop)을 반환.
        예) 간격 60분, 녹화 10분 → 00:00~00:10 녹화, 00:10~01:00 대기, ...
        """
        interval = timedelta(minutes=self._sched_interval)
        rec_dur  = timedelta(minutes=self._sched_rec_duration)
        elapsed  = now - win_start
        slot_idx = int(elapsed.total_seconds()) // int(interval.total_seconds())
        slot_start = win_start + interval * slot_idx
        slot_stop  = slot_start + rec_dur
        if slot_stop > win_stop:
            slot_stop = win_stop
        if slot_start >= win_stop:
            return None, None
        return slot_start, slot_stop

    def _on_schedule_tick(self):
        if not self._schedule_active:
            return

        now = datetime.now()
        win_start, win_stop = self._get_today_window()

        # 녹화 시간대 밖
        if now < win_start or now >= win_stop:
            if self._schedule_running:
                self._schedule_stop_recording()
                logger.info("[예약] 시간대 종료 — 녹화 정지")
            self._update_schedule_label()
            return

        # 녹화 시간대 내
        if self._sched_use_repeat:
            slot_start, slot_stop = self._current_interval_slot(
                now, win_start, win_stop)
            if slot_start is None:
                if self._schedule_running:
                    self._schedule_stop_recording()
            elif slot_start <= now < slot_stop:
                if not self._schedule_running:
                    self._schedule_start_recording()
                    logger.info(f"[예약] 반복 녹화 시작  "
                                f"{slot_start:%H:%M}~{slot_stop:%H:%M}")
            else:
                if self._schedule_running:
                    self._schedule_stop_recording()
        else:
            if not self._schedule_running:
                self._schedule_start_recording()

        self._update_schedule_label()

    def _schedule_start_recording(self):
        self._schedule_running = True
        for panel in self.panels:
            if panel.camera and not panel._recording:
                panel.toggle_record()
        logger.info("[예약] 녹화 시작")

    def _schedule_stop_recording(self):
        self._schedule_running = False
        for panel in self.panels:
            if panel._recording:
                panel.toggle_record()
        logger.info("[예약] 녹화 정지")

    def _update_schedule_label(self):
        now = datetime.now()
        win_start, win_stop = self._get_today_window()

        if self._schedule_running:
            if self._sched_use_repeat:
                _, slot_stop = self._current_interval_slot(
                    now, win_start, win_stop)
                end = slot_stop if slot_stop else win_stop
            else:
                end = win_stop
            secs_left = max(0, int((end - now).total_seconds()))
            mins, secs = divmod(secs_left, 60)
            hours, mins = divmod(mins, 60)
            self.lbl_schedule_status.setText(
                f"녹화 중  |  남은 시간: {hours:02d}:{mins:02d}:{secs:02d}"
                f"  ({self._sched_start_hour}시~{self._sched_stop_hour}시)")
            self.lbl_schedule_status.setStyleSheet(
                f"color: #ff4444; font-weight: bold;"
                f" font-size: {BASE_FONT_SIZE}px;")
        else:
            if now < win_start:
                next_start = win_start
            elif now >= win_stop:
                tomorrow = now.replace(hour=0, minute=0, second=0,
                                       microsecond=0) + timedelta(days=1)
                next_start = tomorrow + timedelta(hours=self._sched_start_hour)
            else:
                interval = timedelta(minutes=self._sched_interval)
                elapsed  = now - win_start
                slot_idx = (int(elapsed.total_seconds())
                            // int(interval.total_seconds()))
                next_start = win_start + interval * (slot_idx + 1)
                if next_start >= win_stop:
                    tomorrow = now.replace(hour=0, minute=0, second=0,
                                           microsecond=0) + timedelta(days=1)
                    next_start = tomorrow + timedelta(
                        hours=self._sched_start_hour)

            secs_left = max(0, int((next_start - now).total_seconds()))
            mins, secs = divmod(secs_left, 60)
            hours, mins = divmod(mins, 60)
            repeat_str = ""
            if self._sched_use_repeat:
                repeat_str = (f"  (반복: {self._sched_interval}분"
                              f" / 녹화: {self._sched_rec_duration}분)")
            self.lbl_schedule_status.setText(
                f"대기 중  |  시작까지: {hours:02d}:{mins:02d}:{secs:02d}"
                f"  ({self._sched_start_hour}시~{self._sched_stop_hour}시)"
                f"{repeat_str}")
            self.lbl_schedule_status.setStyleSheet(
                f"color: #2ecc71; font-size: {BASE_FONT_SIZE}px;")

    def closeEvent(self, event):
        if self._schedule_active:
            self._cancel_schedule()
        for panel in self.panels:
            panel.cleanup()
        logger.info("프로그램 종료")
        event.accept()


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
def _parse_query_size(text: str):
    if text.lower() in ("native", "원본", "auto"):
        return None
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--query-size 형식이 잘못되었습니다: {text!r} "
            "(예: native 또는 720x480)")


def main():
    global OPT_QUERY_SIZE, OPT_WRITE_AVI

    parser = argparse.ArgumentParser(
        description="TmSDK 다중 카메라 · 복수 ROI · 예약 녹화 + Raw Y16 저장")
    parser.add_argument(
        "--query-size", type=_parse_query_size, default="native",
        help="저장 해상도. 'native'(기본, 카메라 원본 160x120) 또는 720x480 형식")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help=f"전역 저장 폴더 (기본: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument(
        "--no-avi", action="store_true",
        help="참고용 AVI 를 저장하지 않는다 (Y16 + CSV 만)")
    args = parser.parse_args()

    OPT_QUERY_SIZE = args.query_size
    OPT_WRITE_AVI  = not args.no_avi

    size_str = ("카메라 원본" if OPT_QUERY_SIZE is None
                else f"{OPT_QUERY_SIZE[0]}x{OPT_QUERY_SIZE[1]}")
    logger.info("=" * 60)
    logger.info("Multi Camera Multi-ROI  ·  Raw Y16 Recorder 시작")
    logger.info(f"  저장 해상도 : {size_str}")
    logger.info(f"  저장 폴더   : {args.output}")
    logger.info(f"  참고용 AVI  : {'저장' if OPT_WRITE_AVI else '저장 안 함'}")
    logger.info("=" * 60)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.global_dir_ref[0] = args.output
    win.input_dir.setText(args.output)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
