#################################################################
# File: multi_cameras_multi_roi.py
#
# 실행:
#   python3 multi_cameras_multi_roi.py
#
# 기능:
#   - 카메라 N대 동시 스트리밍 (런타임에 추가 가능)
#   - 각 카메라별 복수 ROI 설정 (Spot, Line, Rect, Ellipse)
#   - ROI 타입 선택 후 드래그로 추가, 개별 삭제 또는 전체 초기화
#   - 각 ROI별 온도(Min/Avg/Max) 개별 측정 및 CSV 저장
#   - 각 카메라별 또는 전체 일괄 녹화
#   - AVI 자동 분할 (1.4 GB 또는 1시간 초과 시)
#   - 카메라 연결 끊김 시 자동 재연결 + 녹화 자동 재개
#   - CSV 주기적 flush (데이터 유실 방지)
#   - 로그 파일 자동 기록 (logs/tmsdk.log, 일별 로테이션 60일 보관)
#   - 예약 녹화: 시작/종료 시간 지정, 반복 간격 설정 가능
#
# 요구사항:
#   pip install PyQt5 opencv-python-headless
#################################################################

import csv
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

import cv2
import numpy as np

from PyQt5.QtCore import QThread, Qt, pyqtSignal, QRect, QPoint, QPointF, QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLineEdit, QFileDialog,
    QScrollArea, QMessageBox, QComboBox, QDateTimeEdit,
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
    log = logging.getLogger("TmSDK")
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
        os.path.join(LOG_DIR, "tmsdk.log"),
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

DEFAULT_OUTPUT_DIR = "output"
PREVIEW_W  = 720
PREVIEW_H  = 480
RECORD_FPS = 8.7

BASE_FONT_SIZE = 12
BTN_H_TOP      = 52
BTN_H_PANEL    = 58

# ── 장기 운용 관련 상수 ───────────────────────────────────────
_RECONNECT_THRESHOLD = 30             # 연속 예외 N회 → 재연결 요청
_SPLIT_MAX_BYTES     = int(1.4 * 1024 ** 3)   # 1.4 GB 초과 시 파일 분할
_SPLIT_MAX_SECS      = 3600           # 1시간 초과 시 파일 분할
_CSV_FLUSH_EVERY     = 30             # N 프레임마다 CSV flush (~3.4초 @ 8.7 fps)

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


# ─────────────────────────────────────────────────────────────
# ROI 데이터 구조
# ─────────────────────────────────────────────────────────────
class RoiItem:
    """한 개의 ROI 정보를 담는 데이터 클래스."""
    def __init__(self, roi_type: RoiType, x1: int, y1: int, x2: int, y2: int):
        self.roi_type = roi_type
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        # 최근 측정 온도
        self.min_temp = 0.0
        self.avg_temp = 0.0
        self.max_temp = 0.0

    def label(self, idx: int) -> str:
        return f"ROI{idx}({ROI_TYPE_NAMES[self.roi_type]})"

    def geometry_str(self) -> str:
        if self.roi_type == RoiType.Spot:
            return f"({self.x1},{self.y1})"
        if self.roi_type == RoiType.Line:
            return f"({self.x1},{self.y1})->({self.x2},{self.y2})"
        # Rect / Ellipse
        w = abs(self.x2 - self.x1)
        h = abs(self.y2 - self.y1)
        return f"({min(self.x1,self.x2)},{min(self.y1,self.y2)} {w}x{h})"


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
# 프레임 캡처 스레드 (복수 ROI 대응)
# ─────────────────────────────────────────────────────────────
class FrameWorker(QThread):
    frame_ready      = pyqtSignal(np.ndarray)
    temp_updated     = pyqtSignal(list)       # list[RoiItem] — 온도 갱신 후 전체 목록
    reconnect_needed = pyqtSignal()

    def __init__(self, camera: TmCamera, roi_list_ref: list, ip: str = ""):
        super().__init__()
        self.camera       = camera
        self.roi_list_ref = roi_list_ref  # 공유 list[RoiItem]
        self.ip           = ip
        self._running     = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        fail_count    = 0

        while self._running:
            try:
                frame = self.camera.query_frame(PREVIEW_W, PREVIEW_H)
                if frame is None:
                    QThread.msleep(10)
                    continue

                fail_count = 0

                bitmap = frame.to_bitmap(ColorOrder.COLOR_RGB)
                img = np.frombuffer(bitmap, dtype=np.uint8).reshape(
                    frame.height(), frame.width(), 3).copy()

                is_y16 = self.camera.get_camera_format() == "Y16"
                roi_snapshot = list(self.roi_list_ref)

                if is_y16 and roi_snapshot:
                    roi_mgr = TmRoiManager()
                    # 모든 ROI를 매니저에 등록
                    for ri in roi_snapshot:
                        if ri.roi_type == RoiType.Spot:
                            roi_mgr.add_item_xy(RoiType.Spot, ri.x1, ri.y1)
                        else:
                            roi_mgr.add_item_xywh(
                                ri.roi_type, ri.x1, ri.y1, ri.x2, ri.y2)

                    # 각 ROI별 온도 측정
                    for idx, ri in enumerate(roi_snapshot):
                        item = roi_mgr.get_roi_item(idx)
                        frame.do_measure(item)

                        if ri.roi_type == RoiType.Spot:
                            spot_item = roi_mgr.get_roi_spot_item(idx)
                            loc = spot_item.get_roi_maxloc()
                            temp = self.camera.get_temperature(loc.value)
                            ri.min_temp = ri.avg_temp = ri.max_temp = temp
                        elif ri.roi_type == RoiType.Line:
                            line_item = roi_mgr.get_roi_line_item(idx)
                            ri.min_temp = self.camera.get_temperature(
                                line_item.get_roi_minloc().value)
                            ri.avg_temp = self.camera.get_temperature(
                                line_item.get_roi_avgloc().value)
                            ri.max_temp = self.camera.get_temperature(
                                line_item.get_roi_maxloc().value)
                        elif ri.roi_type == RoiType.Rect:
                            rect_item = roi_mgr.get_roi_rect_item(idx)
                            ri.min_temp = self.camera.get_temperature(
                                rect_item.get_roi_minloc().value)
                            ri.avg_temp = self.camera.get_temperature(
                                rect_item.get_roi_avgloc().value)
                            ri.max_temp = self.camera.get_temperature(
                                rect_item.get_roi_maxloc().value)
                        elif ri.roi_type == RoiType.Ellipse:
                            ellipse_item = roi_mgr.get_roi_ellipse_item(idx)
                            ri.min_temp = self.camera.get_temperature(
                                ellipse_item.get_roi_minloc().value)
                            ri.avg_temp = self.camera.get_temperature(
                                ellipse_item.get_roi_avgloc().value)
                            ri.max_temp = self.camera.get_temperature(
                                ellipse_item.get_roi_maxloc().value)

                self.frame_ready.emit(img)
                self.temp_updated.emit(roi_snapshot)
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


# ─────────────────────────────────────────────────────────────
# 미리보기 라벨 (복수 ROI 드래그)
# ─────────────────────────────────────────────────────────────
class PreviewLabel(QLabel):
    roi_added = pyqtSignal(int, int, int, int)   # x1, y1, x2, y2

    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #1a1a1a; color: #888; font-size: 14px;")
        self.setText("연결 중...")
        self._drag_start  = None
        self._drag_end    = None
        self._roi_list    = []       # list[RoiItem] 참조
        self._current_type = RoiType.Rect

    def set_roi_list(self, roi_list):
        self._roi_list = roi_list

    def set_roi_type(self, roi_type: RoiType):
        self._current_type = roi_type

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
            x1, y1 = self._drag_start.x(), self._drag_start.y()
            x2, y2 = self._drag_end.x(), self._drag_end.y()
            self._drag_start = None
            self._drag_end   = None
            self.roi_added.emit(x1, y1, x2, y2)
            self.update()

    def set_frame(self, img: np.ndarray):
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        painter = QPainter(pix)
        # 기존 ROI 그리기
        for idx, ri in enumerate(self._roi_list):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            painter.setPen(QPen(color, 2))
            self._draw_roi(painter, ri, idx, color)

        # 드래그 진행 중 미리보기
        if self._drag_start and self._drag_end:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine))
            self._draw_drag_preview(painter)

        painter.end()
        self.setPixmap(pix)

    def _draw_roi(self, painter: QPainter, ri: RoiItem, idx: int, color: QColor):
        if ri.roi_type == RoiType.Spot:
            cx, cy = ri.x1, ri.y1
            painter.drawLine(cx - 6, cy, cx + 6, cy)
            painter.drawLine(cx, cy - 6, cx, cy + 6)
        elif ri.roi_type == RoiType.Line:
            painter.drawLine(ri.x1, ri.y1, ri.x2, ri.y2)
        elif ri.roi_type == RoiType.Rect:
            r = QRect(
                min(ri.x1, ri.x2), min(ri.y1, ri.y2),
                abs(ri.x2 - ri.x1), abs(ri.y2 - ri.y1))
            painter.drawRect(r)
        elif ri.roi_type == RoiType.Ellipse:
            r = QRect(
                min(ri.x1, ri.x2), min(ri.y1, ri.y2),
                abs(ri.x2 - ri.x1), abs(ri.y2 - ri.y1))
            painter.drawEllipse(r)

        # 라벨
        painter.setPen(QPen(color))
        font = QFont("Consolas", 9)
        painter.setFont(font)
        lx = ri.x1 if ri.roi_type == RoiType.Spot else min(ri.x1, ri.x2)
        ly = (ri.y1 - 4) if ri.roi_type == RoiType.Spot else (min(ri.y1, ri.y2) - 4)
        ly = max(ly, 12)
        painter.drawText(lx, ly, f"ROI{idx}")

    def _draw_drag_preview(self, painter: QPainter):
        x1, y1 = self._drag_start.x(), self._drag_start.y()
        x2, y2 = self._drag_end.x(), self._drag_end.y()
        if self._current_type == RoiType.Spot:
            painter.drawLine(x2 - 6, y2, x2 + 6, y2)
            painter.drawLine(x2, y2 - 6, x2, y2 + 6)
        elif self._current_type == RoiType.Line:
            painter.drawLine(x1, y1, x2, y2)
        elif self._current_type == RoiType.Rect:
            painter.drawRect(QRect(
                min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        elif self._current_type == RoiType.Ellipse:
            painter.drawEllipse(QRect(
                min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))


# ─────────────────────────────────────────────────────────────
# 카메라 1대 패널 (복수 ROI 대응)
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
        self._video_writer      = None
        self._csv_file          = None
        self._csv_writer        = None
        self._ip                = ""
        self._label             = ""
        self._current_avi_path  = ""
        self._split_start_time  = None
        self._split_frame_count = 0
        self._csv_frame_count   = 0
        self._was_recording     = False

        self._reconnect_scan    = None
        self._reconnect_worker  = None

        self._current_roi_type  = RoiType.Rect

        self._build_ui()

    def _eff_dir(self):
        return self._local_dir if self._local_dir else self.global_dir_ref[0]

    def _build_ui(self):
        font_main = QFont()
        font_main.setPointSize(BASE_FONT_SIZE)

        self.preview = PreviewLabel()
        self.preview.set_roi_list(self.roi_list)
        self.preview.roi_added.connect(self._on_roi_added)

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
        self.combo_roi_type.addItem("Spot (점)",    RoiType.Spot)
        self.combo_roi_type.addItem("Line (선)",    RoiType.Line)
        self.combo_roi_type.addItem("Rect (사각형)", RoiType.Rect)
        self.combo_roi_type.addItem("Ellipse (타원)", RoiType.Ellipse)
        self.combo_roi_type.setCurrentIndex(2)  # 기본: Rect
        self.combo_roi_type.currentIndexChanged.connect(self._on_roi_type_changed)

        roi_type_row = QHBoxLayout()
        roi_type_row.addWidget(lbl_roi_type)
        roi_type_row.addWidget(self.combo_roi_type, stretch=1)

        # ROI 목록 라벨
        self.lbl_roi = QLabel("ROI: 없음  (타입 선택 후 드래그로 추가)")
        self.lbl_roi.setStyleSheet(f"font-size: {BASE_FONT_SIZE}px; color: #aaa;")
        self.lbl_roi.setWordWrap(True)

        # 온도 라벨
        self.lbl_temp = QLabel("온도 데이터 없음")
        font_temp = QFont("Consolas", BASE_FONT_SIZE)
        self.lbl_temp.setFont(font_temp)
        self.lbl_temp.setAlignment(Qt.AlignLeft)
        self.lbl_temp.setWordWrap(True)
        self.lbl_temp.setMinimumHeight(40)

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
        layout.addLayout(dir_row)
        layout.addLayout(roi_type_row)
        layout.addWidget(self.lbl_roi)
        layout.addWidget(self.lbl_temp)
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

    # ── ROI 타입 변경 ─────────────────────────────────────────
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
        self.btn_record.setEnabled(True)
        self.preview.setText("")
        self._start_worker()

        if self._was_recording:
            self._was_recording = False
            logger.info(f"[{self._ip}] 재연결 완료 — 녹화 자동 재개")
            self._start_recording()

    def set_status(self, text: str):
        self.preview.setText(text)
        self.lbl_temp.setText("온도 데이터 없음")

    # ── 프레임 워커 ───────────────────────────────────────────
    def _start_worker(self):
        self.worker = FrameWorker(self.camera, self.roi_list, self._ip)
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.temp_updated.connect(self._on_temp)
        self.worker.reconnect_needed.connect(self._on_reconnect_needed)
        self.worker.start()

    def _on_temp(self, roi_snapshot: list):
        if not roi_snapshot:
            return
        sym = self.camera.get_temp_unit_symbol() if self.camera else ""
        lines = []
        for idx, ri in enumerate(roi_snapshot):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            hex_color = color.name()
            lines.append(
                f"<span style='color:{hex_color}'>"
                f"ROI{idx}({ROI_TYPE_NAMES[ri.roi_type]}): "
                f"Min={ri.min_temp:.2f} Avg={ri.avg_temp:.2f} "
                f"Max={ri.max_temp:.2f} {sym}</span>"
            )
        self.lbl_temp.setText("<br>".join(lines))

        if self._recording:
            self._write_frame_data(roi_snapshot)

    # ── ROI 추가/삭제 ─────────────────────────────────────────
    def _on_roi_added(self, x1, y1, x2, y2):
        roi_type = self._current_roi_type
        if roi_type == RoiType.Spot:
            # Spot은 마지막 위치(릴리스 위치) 사용
            ri = RoiItem(RoiType.Spot, x2, y2, x2, y2)
        elif roi_type == RoiType.Line:
            ri = RoiItem(RoiType.Line, x1, y1, x2, y2)
        elif roi_type == RoiType.Rect:
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            if rw < 2 or rh < 2:
                return
            ri = RoiItem(RoiType.Rect, rx, ry, rw, rh)
        elif roi_type == RoiType.Ellipse:
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            if rw < 2 or rh < 2:
                return
            ri = RoiItem(RoiType.Ellipse, rx, ry, rw, rh)
        else:
            return

        self.roi_list.append(ri)
        self._update_roi_label()

    def _remove_last_roi(self):
        if self.roi_list:
            self.roi_list.pop()
            self._update_roi_label()

    def _clear_roi(self):
        self.roi_list.clear()
        self._update_roi_label()
        self.lbl_temp.setText("온도 데이터 없음")

    def _update_roi_label(self):
        if not self.roi_list:
            self.lbl_roi.setText("ROI: 없음  (타입 선택 후 드래그로 추가)")
            return
        parts = []
        for idx, ri in enumerate(self.roi_list):
            parts.append(f"ROI{idx}:{ROI_TYPE_NAMES[ri.roi_type]}{ri.geometry_str()}")
        self.lbl_roi.setText("  |  ".join(parts))

    # ── 녹화 ──────────────────────────────────────────────────
    def toggle_record(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _build_csv_header(self):
        """현재 ROI 목록에 맞는 CSV 헤더 생성."""
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

    def _start_recording(self):
        out_dir = self._eff_dir()
        os.makedirs(out_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"{self._ip.replace('.','_')}_{ts}")
        self._current_avi_path = base + ".avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._video_writer = cv2.VideoWriter(
            self._current_avi_path, fourcc, RECORD_FPS, (PREVIEW_W, PREVIEW_H))

        self._csv_file   = open(base + ".csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self._build_csv_header())

        self._split_start_time  = datetime.now()
        self._split_frame_count = 0
        self._csv_frame_count   = 0

        self.worker.frame_ready.connect(self._write_video_frame)

        self._recording = True
        self._set_btn_recording()
        logger.info(f"[{self._ip}] 녹화 시작: {self._current_avi_path}")

    def _stop_recording(self):
        self._recording = False
        if self.worker:
            try:
                self.worker.frame_ready.disconnect(self._write_video_frame)
            except Exception:
                pass
        self._close_recording_files()
        self._set_btn_idle()
        logger.info(f"[{self._ip}] 녹화 정지")

    def _close_recording_files(self):
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

    # ── AVI 자동 분할 ─────────────────────────────────────────
    def _rotate_recording(self):
        logger.info(f"[{self._ip}] AVI 파일 분할 시작")
        self._close_recording_files()

        out_dir = self._eff_dir()
        os.makedirs(out_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(out_dir, f"{self._ip.replace('.','_')}_{ts}")
        self._current_avi_path = base + ".avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._video_writer = cv2.VideoWriter(
            self._current_avi_path, fourcc, RECORD_FPS, (PREVIEW_W, PREVIEW_H))

        self._csv_file   = open(base + ".csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self._build_csv_header())

        self._split_start_time  = datetime.now()
        self._csv_frame_count   = 0
        logger.info(f"[{self._ip}] 새 녹화 파일: {self._current_avi_path}")

    def _write_video_frame(self, img: np.ndarray):
        if self._video_writer is None:
            return

        self._split_frame_count += 1
        if self._split_frame_count >= 100:
            self._split_frame_count = 0
            elapsed = (datetime.now() - self._split_start_time).total_seconds()
            try:
                fsize = os.path.getsize(self._current_avi_path)
            except OSError:
                fsize = 0
            if fsize >= _SPLIT_MAX_BYTES or elapsed >= _SPLIT_MAX_SECS:
                self._rotate_recording()

        if self._video_writer is None:
            return

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 영상에 모든 ROI 오버레이
        for idx, ri in enumerate(self.roi_list):
            color_q = ROI_COLORS[idx % len(ROI_COLORS)]
            color_bgr = (color_q.blue(), color_q.green(), color_q.red())
            if ri.roi_type == RoiType.Spot:
                cv2.drawMarker(bgr, (ri.x1, ri.y1),
                               color_bgr, cv2.MARKER_CROSS, 12, 2)
            elif ri.roi_type == RoiType.Line:
                cv2.line(bgr, (ri.x1, ri.y1), (ri.x2, ri.y2), color_bgr, 2)
            elif ri.roi_type == RoiType.Rect:
                cv2.rectangle(bgr,
                    (ri.x1, ri.y1),
                    (ri.x1 + ri.x2, ri.y1 + ri.y2),
                    color_bgr, 2)
            elif ri.roi_type == RoiType.Ellipse:
                cx = ri.x1 + ri.x2 // 2
                cy = ri.y1 + ri.y2 // 2
                cv2.ellipse(bgr, (cx, cy),
                            (ri.x2 // 2, ri.y2 // 2),
                            0, 0, 360, color_bgr, 2)
        self._video_writer.write(bgr)

    def _write_frame_data(self, roi_snapshot: list):
        if self._csv_writer is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = [ts]
        if not roi_snapshot:
            row.extend(["0.00", "0.00", "0.00"])
        else:
            for ri in roi_snapshot:
                row.append(ri.geometry_str())
                row.extend([
                    f"{ri.min_temp:.2f}",
                    f"{ri.avg_temp:.2f}",
                    f"{ri.max_temp:.2f}"
                ])
        self._csv_writer.writerow(row)
        self._csv_frame_count += 1
        if self._csv_frame_count >= _CSV_FLUSH_EVERY:
            self._csv_frame_count = 0
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
            if self.worker:
                try:
                    self.worker.frame_ready.disconnect(self._write_video_frame)
                except Exception:
                    pass
            self._close_recording_files()

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
        self.setWindowTitle("TmSDK - Multi Camera Multi-ROI Stream")

        self.global_dir_ref  = [DEFAULT_OUTPUT_DIR]
        self.panels: list[CameraPanel] = []
        self.connect_workers: list[ConnectWorker] = []
        self._scan_worker  = None
        self._pending_cfgs = []

        # ── 예약 녹화 상태 ──
        self._schedule_active   = False
        self._schedule_running  = False   # 현재 예약에 의해 녹화 중인지
        self._schedule_next_start = None  # 다음 녹화 시작 시각
        self._schedule_next_stop  = None  # 다음 녹화 종료 시각

        self._build_ui()
        self._start_initial_scan()

        # 예약 녹화 타이머 (1초 간격)
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
            lambda t: self.global_dir_ref.__setitem__(0, t.strip() or DEFAULT_OUTPUT_DIR)
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

        row1 = QHBoxLayout()
        row1.addWidget(lbl_dir)
        row1.addWidget(self.input_dir)
        row1.addWidget(btn_browse)
        row1.addSpacing(30)
        row1.addWidget(self.btn_record_all)
        row1.addWidget(self.btn_stop_all)
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

        # ── 행 3: 예약 녹화 ──────────────────────────────────
        now = QDateTime.currentDateTime()
        start_default = now.addSecs(60)  # 1분 뒤

        lbl_start = QLabel("시작:")
        lbl_start.setFont(font)
        self.dt_start = QDateTimeEdit(start_default)
        self.dt_start.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dt_start.setCalendarPopup(True)
        self.dt_start.setFixedHeight(BTN_H_TOP)
        self.dt_start.setFont(font)

        lbl_stop = QLabel("종료:")
        lbl_stop.setFont(font)
        self.dt_stop = QDateTimeEdit(start_default.addSecs(3600))
        self.dt_stop.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dt_stop.setCalendarPopup(True)
        self.dt_stop.setFixedHeight(BTN_H_TOP)
        self.dt_stop.setFont(font)

        self.chk_repeat = QCheckBox("반복")
        self.chk_repeat.setFont(font)
        self.chk_repeat.toggled.connect(
            lambda checked: self.spin_interval.setEnabled(checked))

        lbl_interval = QLabel("간격:")
        lbl_interval.setFont(font)
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 1440)
        self.spin_interval.setValue(60)
        self.spin_interval.setSuffix(" 분")
        self.spin_interval.setFixedHeight(BTN_H_TOP)
        self.spin_interval.setFont(font)
        self.spin_interval.setEnabled(False)

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
        row3.addWidget(self.dt_start)
        row3.addWidget(lbl_stop)
        row3.addWidget(self.dt_stop)
        row3.addSpacing(15)
        row3.addWidget(self.chk_repeat)
        row3.addWidget(lbl_interval)
        row3.addWidget(self.spin_interval)
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
        scroll.setMinimumHeight(PREVIEW_H + 320)

        main_layout = QVBoxLayout()
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addLayout(row3)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self.resize(PREVIEW_W * 3 + 360, PREVIEW_H + 680)

    # ── 초기 스캔 → 연결 ──────────────────────────────────────
    def _start_initial_scan(self):
        self._pending_cfgs = list(CAMERAS)
        self._scan_worker  = ScanWorker()
        self._scan_worker.scan_done.connect(self._on_initial_scan_done)
        self._scan_worker.start()

    def _on_initial_scan_done(self, cam_map: dict):
        if not self._pending_cfgs:
            for i, (ip, _) in enumerate(cam_map.items()):
                self._launch_connect(ip, f"Camera {i+1}", cam_map)
            return
        for cfg in self._pending_cfgs:
            self._launch_connect(cfg["ip"], cfg.get("label", cfg["ip"]), cam_map)
        self._pending_cfgs = []

    # ── IP 입력 필드에서 직접 연결 ────────────────────────────
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

    def _on_runtime_scan_done(self, ip: str, label: str, panel: CameraPanel, cam_map: dict):
        worker = ConnectWorker(ip, label, cam_map.get(ip))
        worker.connected.connect(lambda cam, lbl, p=panel, i=ip:
                                 self._on_connected(cam, lbl, p, i))
        worker.failed.connect(lambda i, p=panel: self._on_failed(i, p))
        worker.start()
        self.connect_workers.append(worker)

    def _on_connected(self, cam: TmCamera, label: str, panel: CameraPanel, ip: str):
        panel.setTitle(f"{label}  [{ip}]")
        panel.attach_camera(cam, ip, label)
        logger.info(f"[성공] {label} ({ip})")

    def _on_failed(self, ip: str, panel: CameraPanel):
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

    # ── 예약 녹화 ──────────────────────────────────────────────
    def _toggle_schedule(self):
        if self._schedule_active:
            self._cancel_schedule()
        else:
            self._activate_schedule()

    def _activate_schedule(self):
        start_dt = self.dt_start.dateTime().toPyDateTime()
        stop_dt  = self.dt_stop.dateTime().toPyDateTime()
        now      = datetime.now()

        if stop_dt <= start_dt:
            QMessageBox.warning(self, "시간 오류",
                                "종료 시간이 시작 시간보다 뒤여야 합니다.")
            return

        # 반복 모드가 아닌데 종료 시간이 이미 지났으면 경고
        if stop_dt <= now and not self.chk_repeat.isChecked():
            QMessageBox.warning(self, "시간 오류",
                                "종료 시간이 이미 지났습니다.")
            return

        self._schedule_next_start = start_dt
        self._schedule_next_stop  = stop_dt
        self._schedule_active     = True
        self._schedule_running    = False

        # 이미 시작 시간이 지났고 종료 전이면 즉시 녹화 시작
        if start_dt <= now < stop_dt:
            self._schedule_start_recording()

        # UI 잠금
        self.dt_start.setEnabled(False)
        self.dt_stop.setEnabled(False)
        self.chk_repeat.setEnabled(False)
        self.spin_interval.setEnabled(False)
        self.btn_schedule.setText("예약 취소")
        self.btn_schedule.setStyleSheet(
            "QPushButton { background-color: #922b21; color: white;"
            " border-radius: 4px; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        self._update_schedule_label()

        repeat_str = ""
        if self.chk_repeat.isChecked():
            repeat_str = f"  반복: {self.spin_interval.value()}분 간격"
        logger.info(
            f"[예약] 활성화  "
            f"시작={start_dt:%Y-%m-%d %H:%M:%S}  "
            f"종료={stop_dt:%Y-%m-%d %H:%M:%S}{repeat_str}")

    def _cancel_schedule(self):
        # 예약에 의해 녹화 중이었으면 정지
        if self._schedule_running:
            self._schedule_stop_recording()

        self._schedule_active  = False
        self._schedule_running = False
        self._schedule_next_start = None
        self._schedule_next_stop  = None

        # UI 잠금 해제
        self.dt_start.setEnabled(True)
        self.dt_stop.setEnabled(True)
        self.chk_repeat.setEnabled(True)
        self.spin_interval.setEnabled(self.chk_repeat.isChecked())
        self.btn_schedule.setText("예약 녹화 설정")
        self.btn_schedule.setStyleSheet(
            "QPushButton { background-color: #1a5276; color: white;"
            " border-radius: 4px; }"
            "QPushButton:hover { background-color: #2471a3; }"
        )
        self.lbl_schedule_status.setText("예약: 비활성")
        self.lbl_schedule_status.setStyleSheet("color: #888;")
        logger.info("[예약] 취소됨")

    def _on_schedule_tick(self):
        if not self._schedule_active:
            return

        now = datetime.now()
        start = self._schedule_next_start
        stop  = self._schedule_next_stop

        # 녹화 시작 시점 도달
        if not self._schedule_running and start <= now < stop:
            self._schedule_start_recording()

        # 녹화 종료 시점 도달
        if self._schedule_running and now >= stop:
            self._schedule_stop_recording()

            if self.chk_repeat.isChecked():
                interval_min = self.spin_interval.value()
                interval = timedelta(minutes=interval_min)
                duration = stop - start

                # 다음 시작 = 이전 시작 + 반복 간격
                next_start = start + interval
                # 이미 지난 시간이면 현재 기준으로 다음 주기 계산
                while next_start + duration <= now:
                    next_start += interval

                self._schedule_next_start = next_start
                self._schedule_next_stop  = next_start + duration
                logger.info(
                    f"[예약] 다음 반복  "
                    f"시작={self._schedule_next_start:%Y-%m-%d %H:%M:%S}  "
                    f"종료={self._schedule_next_stop:%Y-%m-%d %H:%M:%S}")
            else:
                # 반복 아니면 예약 종료
                self._cancel_schedule()
                return

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
        start = self._schedule_next_start
        stop  = self._schedule_next_stop

        if self._schedule_running:
            remaining = stop - now
            mins, secs = divmod(int(remaining.total_seconds()), 60)
            hours, mins = divmod(mins, 60)
            self.lbl_schedule_status.setText(
                f"녹화 중  |  남은 시간: {hours:02d}:{mins:02d}:{secs:02d}")
            self.lbl_schedule_status.setStyleSheet(
                f"color: #ff4444; font-weight: bold;"
                f" font-size: {BASE_FONT_SIZE}px;")
        else:
            wait = start - now
            mins, secs = divmod(int(wait.total_seconds()), 60)
            hours, mins = divmod(mins, 60)
            repeat_str = ""
            if self.chk_repeat.isChecked():
                repeat_str = f"  (반복: {self.spin_interval.value()}분)"
            self.lbl_schedule_status.setText(
                f"대기 중  |  시작까지: {hours:02d}:{mins:02d}:{secs:02d}"
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
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TmSDK Multi Camera Multi-ROI Stream 시작")
    logger.info("=" * 60)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
