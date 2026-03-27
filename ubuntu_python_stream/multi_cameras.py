#################################################################
# File: multi_cameras.py
#
# 실행:
#   python3 multi_cameras.py
#
# 기능:
#   - 카메라 N대 동시 스트리밍 (런타임에 추가 가능)
#   - 각 카메라별 ROI 드래그 설정, 저장 폴더 개별 설정
#   - 각 카메라별 또는 전체 일괄 녹화
#   - AVI 자동 분할 (1.4 GB 또는 1시간 초과 시)
#   - 카메라 연결 끊김 시 자동 재연결 + 녹화 자동 재개
#   - CSV 주기적 flush (데이터 유실 방지)
#   - 로그 파일 자동 기록 (logs/tmsdk.log, 일별 로테이션 60일 보관)
#
# 요구사항:
#   pip install PyQt5 opencv-python-headless
#################################################################

import csv
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import cv2
import numpy as np

from PyQt5.QtCore import QThread, Qt, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLineEdit, QFileDialog,
    QScrollArea, QMessageBox
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
# 프레임 캡처 스레드
# ─────────────────────────────────────────────────────────────
class FrameWorker(QThread):
    frame_ready      = pyqtSignal(np.ndarray)
    temp_updated     = pyqtSignal(float, float, float)
    reconnect_needed = pyqtSignal()   # 연속 오류 임계 도달 시 재연결 요청

    def __init__(self, camera: TmCamera, roi_ref: list, ip: str = ""):
        super().__init__()
        self.camera   = camera
        self.roi_ref  = roi_ref
        self.ip       = ip
        self._running = False

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

                fail_count = 0  # 프레임 수신 성공 시 리셋

                bitmap = frame.to_bitmap(ColorOrder.COLOR_RGB)
                img = np.frombuffer(bitmap, dtype=np.uint8).reshape(
                    frame.height(), frame.width(), 3).copy()

                min_v = avg_v = max_v = 0.0
                roi = self.roi_ref[0]
                if self.camera.get_camera_format() == "Y16" and not roi.isNull():
                    roi_mgr = TmRoiManager()
                    roi_mgr.add_item_xywh(RoiType.Rect,
                                          roi.x(), roi.y(),
                                          roi.width(), roi.height())
                    item = roi_mgr.get_roi_item(0)
                    frame.do_measure(item)
                    rect_item = roi_mgr.get_roi_rect_item(0)
                    min_v = self.camera.get_temperature(rect_item.get_roi_minloc().value)
                    avg_v = self.camera.get_temperature(rect_item.get_roi_avgloc().value)
                    max_v = self.camera.get_temperature(rect_item.get_roi_maxloc().value)

                self.frame_ready.emit(img)
                self.temp_updated.emit(min_v, avg_v, max_v)
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
# 미리보기 라벨 (ROI 드래그)
# ─────────────────────────────────────────────────────────────
class PreviewLabel(QLabel):
    roi_changed = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #1a1a1a; color: #888; font-size: 14px;")
        self.setText("연결 중...")
        self._roi        = QRect()
        self._drag_start = None

    def roi(self):
        return self._roi

    def clear_roi(self):
        self._roi = QRect()
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_start = e.pos()

    def mouseMoveEvent(self, e):
        if self._drag_start:
            self._roi = QRect(self._drag_start, e.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._drag_start:
            self._roi = QRect(self._drag_start, e.pos()).normalized()
            self._drag_start = None
            self.roi_changed.emit(self._roi)
            self.update()

    def set_frame(self, img: np.ndarray):
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        if not self._roi.isNull():
            painter = QPainter(pix)
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawRect(self._roi)
            painter.end()
        self.setPixmap(pix)


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
        self.roi_ref        = [QRect()]

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
        self._was_recording     = False   # 재연결 후 녹화 재개용

        self._reconnect_scan    = None
        self._reconnect_worker  = None

        self._build_ui()

    def _eff_dir(self):
        return self._local_dir if self._local_dir else self.global_dir_ref[0]

    def _build_ui(self):
        font_main = QFont()
        font_main.setPointSize(BASE_FONT_SIZE)

        self.preview = PreviewLabel()
        self.preview.roi_changed.connect(self._on_roi_changed)

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

        self.lbl_roi = QLabel("ROI: 없음  (드래그로 설정)")
        self.lbl_roi.setStyleSheet(f"font-size: {BASE_FONT_SIZE}px; color: #aaa;")

        self.lbl_temp = QLabel("Min: —  |  Avg: —  |  Max: —")
        font_temp = QFont("Consolas", BASE_FONT_SIZE + 1)
        self.lbl_temp.setFont(font_temp)
        self.lbl_temp.setAlignment(Qt.AlignCenter)

        # 녹화 버튼
        self.btn_record = QPushButton("● 녹화 시작")
        self.btn_record.setFixedHeight(BTN_H_PANEL)
        font_rec = QFont()
        font_rec.setPointSize(BASE_FONT_SIZE + 1)
        font_rec.setBold(True)
        self.btn_record.setFont(font_rec)
        self.btn_record.setEnabled(False)
        self._set_btn_idle()
        self.btn_record.clicked.connect(self.toggle_record)

        self.btn_clear = QPushButton("ROI 초기화")
        self.btn_clear.setFixedHeight(BTN_H_PANEL)
        self.btn_clear.setFont(font_main)
        self.btn_clear.clicked.connect(self._clear_roi)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_record, stretch=2)
        btn_row.addWidget(self.btn_clear, stretch=1)

        layout = QVBoxLayout()
        layout.addWidget(self.preview)
        layout.addLayout(dir_row)
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

        # 재연결 후 녹화 자동 재개
        if self._was_recording:
            self._was_recording = False
            logger.info(f"[{self._ip}] 재연결 완료 — 녹화 자동 재개")
            self._start_recording()

    def set_status(self, text: str):
        self.preview.setText(text)
        self.lbl_temp.setText("Min: —  |  Avg: —  |  Max: —")

    # ── 프레임 워커 ───────────────────────────────────────────
    def _start_worker(self):
        self.worker = FrameWorker(self.camera, self.roi_ref, self._ip)
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.temp_updated.connect(self._on_temp)
        self.worker.reconnect_needed.connect(self._on_reconnect_needed)
        self.worker.start()

    def _on_temp(self, min_v, avg_v, max_v):
        if not self.roi_ref[0].isNull():
            sym = self.camera.get_temp_unit_symbol()
            self.lbl_temp.setText(
                f"Min: {min_v:.2f} {sym}  |  "
                f"Avg: {avg_v:.2f} {sym}  |  "
                f"Max: {max_v:.2f} {sym}"
            )
        if self._recording:
            self._write_frame_data(min_v, avg_v, max_v)

    # ── ROI ───────────────────────────────────────────────────
    def _on_roi_changed(self, rect: QRect):
        self.roi_ref[0] = rect
        self.lbl_roi.setText(
            f"ROI: x={rect.x()} y={rect.y()} w={rect.width()} h={rect.height()}"
        )

    def _clear_roi(self):
        self.roi_ref[0] = QRect()
        self.preview.clear_roi()
        self.lbl_roi.setText("ROI: 없음  (드래그로 설정)")
        self.lbl_temp.setText("Min: —  |  Avg: —  |  Max: —")

    # ── 녹화 ──────────────────────────────────────────────────
    def toggle_record(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

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
        self._csv_writer.writerow(
            ["timestamp", "roi_x", "roi_y", "roi_w", "roi_h",
             "min_temp", "avg_temp", "max_temp"])

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
        """VideoWriter와 CSV를 안전하게 닫는다."""
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
        """현재 AVI/CSV를 닫고 새 파일로 전환 (분할)."""
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
        self._csv_writer.writerow(
            ["timestamp", "roi_x", "roi_y", "roi_w", "roi_h",
             "min_temp", "avg_temp", "max_temp"])

        self._split_start_time  = datetime.now()
        self._csv_frame_count   = 0
        logger.info(f"[{self._ip}] 새 녹화 파일: {self._current_avi_path}")

    def _write_video_frame(self, img: np.ndarray):
        if self._video_writer is None:
            return

        # 100 프레임마다 분할 조건 체크
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
                # rotate 후 현재 프레임은 새 파일에 기록

        if self._video_writer is None:
            return
        roi = self.roi_ref[0]
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if not roi.isNull():
            cv2.rectangle(bgr,
                (roi.x(), roi.y()),
                (roi.x() + roi.width(), roi.y() + roi.height()),
                (0, 255, 255), 2)
        self._video_writer.write(bgr)

    def _write_frame_data(self, min_v, avg_v, max_v):
        if self._csv_writer is None:
            return
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        roi = self.roi_ref[0]
        if roi.isNull():
            rx, ry, rw, rh = "", "", "", ""
        else:
            rx, ry, rw, rh = roi.x(), roi.y(), roi.width(), roi.height()
        self._csv_writer.writerow(
            [ts, rx, ry, rw, rh, f"{min_v:.2f}", f"{avg_v:.2f}", f"{max_v:.2f}"])
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
            # 파일은 닫되 btn 상태 변경 없이 내부만 정리 (재연결 후 자동 재개)
            self._recording = False
            if self.worker:
                try:
                    self.worker.frame_ready.disconnect(self._write_video_frame)
                except Exception:
                    pass
            self._close_recording_files()

        self.worker = None  # FrameWorker 가 스스로 종료했으므로 참조만 제거

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
        self.setWindowTitle("TmSDK - Multi Camera Stream")

        self.global_dir_ref  = [DEFAULT_OUTPUT_DIR]
        self.panels: list[CameraPanel] = []
        self.connect_workers: list[ConnectWorker] = []
        self._scan_worker  = None
        self._pending_cfgs = []

        self._build_ui()
        self._start_initial_scan()

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

        # 카메라 패널 영역 (스크롤)
        self.panel_container = QWidget()
        self.panel_layout    = QHBoxLayout()
        self.panel_layout.setAlignment(Qt.AlignLeft)
        self.panel_container.setLayout(self.panel_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel_container)
        scroll.setMinimumHeight(PREVIEW_H + 220)

        main_layout = QVBoxLayout()
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
        self.resize(PREVIEW_W * 3 + 360, PREVIEW_H + 520)

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

    def closeEvent(self, event):
        for panel in self.panels:
            panel.cleanup()
        logger.info("프로그램 종료")
        event.accept()


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TmSDK Multi Camera Stream 시작")
    logger.info("=" * 60)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
