#################################################################
# File: record_camera.py
#
# 실행 예시:
#   python3 record_camera.py --ip 192.168.0.151
#   python3 record_camera.py --ip 192.168.0.152
#
# 조작:
#   - 미리보기 화면에서 마우스 드래그 → ROI 설정
#   - [녹화 시작] 버튼 → 녹화 시작/정지
#   - 녹화 결과: output/<IP>_<날짜시각>.avi  +  output/<IP>_<날짜시각>.csv
#   - AVI 자동 분할: 1.4 GB 또는 1시간 초과 시
#
# 요구사항:
#   pip install PyQt5 opencv-python-headless numpy
#   ※ opencv-python (headless 아닌 버전)은 Qt xcb 충돌 발생 — 사용 금지
#################################################################

import argparse
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
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
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
    log = logging.getLogger("TmSDK.record")
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


# ── 기본값 ────────────────────────────────────────────────────
CAM_NAME   = "TMC160F"
CAM_SERIAL = ""
CAM_MAC    = ""
OUTPUT_DIR = "output"

PREVIEW_W  = 720
PREVIEW_H  = 480
RECORD_FPS = 8.7   # TMC160EH 실제 프레임레이트

# 장기 운용 상수
_SPLIT_MAX_BYTES = int(1.4 * 1024 ** 3)   # 1.4 GB
_SPLIT_MAX_SECS  = 3600                    # 1시간
_CSV_FLUSH_EVERY = 30                      # 30 프레임마다 flush


# ─────────────────────────────────────────────────────────────
# 카메라 연결 스레드
# ─────────────────────────────────────────────────────────────
class ConnectWorker(QThread):
    connected = pyqtSignal(object)
    failed    = pyqtSignal(str)

    def __init__(self, ip, name, serial, mac):
        super().__init__()
        self.ip     = ip
        self.name   = name
        self.serial = serial
        self.mac    = mac

    def run(self):
        # 스캔은 1회만 — 반복 호출 시 카메라 네트워크 스택 교란
        logger.info("[스캔] 카메라 목록 조회 중...")
        cam_list = TmCamera.get_remote_camera_list()
        cam_info = next((c for c in cam_list if c.ip == self.ip), None)

        if cam_info:
            name   = cam_info.name
            serial = cam_info.serial_number
            mac    = cam_info.mac
            fmt    = cam_info.media_info_list[0].format if cam_info.media_info_list else "Y16"
            logger.info(f"[스캔 결과] {name}  serial={serial}  mac={mac}  fmt={fmt}")
        else:
            name, serial, mac, fmt = self.name, self.serial, self.mac, "Y16"
            logger.warning(f"[스캔 결과] {self.ip} 목록에 없음 — 수동 정보로 시도")

        for attempt in range(5):
            cam = TmCamera()
            ret = cam.open_remote_camera(name, serial, mac, self.ip, fmt)
            if ret:
                self.connected.emit(cam)
                return
            del cam
            logger.warning(f"[연결 시도 {attempt+1}/5] {self.ip} 실패, 3초 후 재시도...")
            time.sleep(3)

        self.failed.emit(self.ip)


# ─────────────────────────────────────────────────────────────
# 프레임 캡처 스레드
# ─────────────────────────────────────────────────────────────
class FrameWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, float, float, float)
    # (rgb_frame, min_temp, avg_temp, max_temp)

    def __init__(self, camera: TmCamera, roi: QRect):
        super().__init__()
        self.camera   = camera
        self.roi      = roi
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                frame = self.camera.query_frame(PREVIEW_W, PREVIEW_H)
                if frame is None:
                    QThread.msleep(10)
                    continue

                bitmap = frame.to_bitmap(ColorOrder.COLOR_RGB)
                img = np.frombuffer(bitmap, dtype=np.uint8).reshape(
                    frame.height(), frame.width(), 3).copy()

                min_v = avg_v = max_v = 0.0
                if self.camera.get_camera_format() == "Y16" and not self.roi.isNull():
                    roi_mgr = TmRoiManager()
                    roi_mgr.add_item_xywh(RoiType.Rect,
                                          self.roi.x(), self.roi.y(),
                                          self.roi.width(), self.roi.height())
                    item = roi_mgr.get_roi_item(0)
                    frame.do_measure(item)
                    rect_item = roi_mgr.get_roi_rect_item(0)
                    min_v = self.camera.get_temperature(rect_item.get_roi_minloc().value)
                    avg_v = self.camera.get_temperature(rect_item.get_roi_avgloc().value)
                    max_v = self.camera.get_temperature(rect_item.get_roi_maxloc().value)

                self.frame_ready.emit(img, min_v, avg_v, max_v)
                del frame

            except Exception as e:
                logger.error(f"[FrameWorker] {e}")
                QThread.msleep(100)


# ─────────────────────────────────────────────────────────────
# 미리보기 라벨 (마우스 드래그로 ROI 설정)
# ─────────────────────────────────────────────────────────────
class PreviewLabel(QLabel):
    roi_changed = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #1a1a1a; color: #888;")
        self.setText("카메라 연결 중...")
        self._roi        = QRect()
        self._drag_start = None

    def roi(self) -> QRect:
        return self._roi

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
# 메인 윈도우
# ─────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self, ip, name, serial, mac):
        super().__init__()
        self.ip      = ip
        self.camera  = None
        self.worker  = None
        self.roi     = QRect()

        self._recording         = False
        self._video_writer      = None
        self._csv_file          = None
        self._csv_writer        = None
        self._current_avi_path  = ""
        self._split_start_time  = None
        self._split_frame_count = 0
        self._csv_frame_count   = 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._build_ui(ip)
        self._start_connect(ip, name, serial, mac)

    # ── UI 구성 ───────────────────────────────────────────────
    def _build_ui(self, ip):
        self.setWindowTitle(f"TmSDK Record  —  {ip}")

        self.preview = PreviewLabel()
        self.preview.roi_changed.connect(self._on_roi_changed)

        self.lbl_roi  = QLabel("ROI: 없음  (드래그로 설정)")
        self.lbl_temp = QLabel("Min: —  |  Avg: —  |  Max: —")
        self.lbl_temp.setFont(QFont("Consolas", 10))

        self.btn_record = QPushButton("● 녹화 시작")
        self.btn_record.setFixedHeight(40)
        self.btn_record.setEnabled(False)
        self._set_btn_idle()
        self.btn_record.clicked.connect(self._toggle_record)

        self.btn_clear_roi = QPushButton("ROI 초기화")
        self.btn_clear_roi.setFixedHeight(40)
        self.btn_clear_roi.clicked.connect(self._clear_roi)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_record)
        btn_row.addWidget(self.btn_clear_roi)

        layout = QVBoxLayout()
        layout.addWidget(self.preview)
        layout.addWidget(self.lbl_roi)
        layout.addWidget(self.lbl_temp)
        layout.addLayout(btn_row)
        self.setLayout(layout)
        self.adjustSize()

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

    # ── 카메라 연결 ───────────────────────────────────────────
    def _start_connect(self, ip, name, serial, mac):
        self.connect_worker = ConnectWorker(ip, name, serial, mac)
        self.connect_worker.connected.connect(self._on_connected)
        self.connect_worker.failed.connect(self._on_connect_failed)
        self.connect_worker.start()

    def _on_connected(self, cam: TmCamera):
        self.camera = cam
        self.camera.set_temp_unit(TempUnit.CELSIUS)
        self.camera.set_color_map(ColormapTypes.Inferno + 1)
        logger.info(f"[연결 성공] {self.ip}")
        self.btn_record.setEnabled(True)
        self.preview.setText("")
        self._start_frame_worker()

    def _on_connect_failed(self, ip):
        logger.error(f"[연결 실패] {ip}")
        self.preview.setText(f"연결 실패\n{ip}")

    # ── 프레임 수신 ───────────────────────────────────────────
    def _start_frame_worker(self):
        self.worker = FrameWorker(self.camera, self.roi)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()

    def _on_frame(self, img: np.ndarray, min_v, avg_v, max_v):
        self.preview.set_frame(img)
        sym = self.camera.get_temp_unit_symbol()
        if not self.roi.isNull():
            self.lbl_temp.setText(
                f"Min: {min_v:.2f} {sym}  |  "
                f"Avg: {avg_v:.2f} {sym}  |  "
                f"Max: {max_v:.2f} {sym}"
            )
        if self._recording:
            self._write_frame(img, min_v, avg_v, max_v)

    # ── ROI 관리 ──────────────────────────────────────────────
    def _on_roi_changed(self, rect: QRect):
        self.roi = rect
        if self.worker:
            self.worker.roi = rect
        self.lbl_roi.setText(
            f"ROI: x={rect.x()} y={rect.y()} w={rect.width()} h={rect.height()}"
        )

    def _clear_roi(self):
        self.roi = QRect()
        if self.worker:
            self.worker.roi = QRect()
        self.preview._roi = QRect()
        self.preview.update()
        self.lbl_roi.setText("ROI: 없음  (드래그로 설정)")
        self.lbl_temp.setText("Min: —  |  Avg: —  |  Max: —")

    # ── 녹화 시작/정지 ────────────────────────────────────────
    def _toggle_record(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(OUTPUT_DIR, f"{self.ip.replace('.','_')}_{ts}")
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

        self._recording = True
        self._set_btn_recording()
        logger.info(f"[녹화 시작] {self._current_avi_path}")

    def _stop_recording(self):
        self._recording = False
        self._close_files()
        self._set_btn_idle()
        logger.info(f"[녹화 정지] {self.ip}")

    def _close_files(self):
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._csv_file:
            try:
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
            except OSError as e:
                logger.error(f"CSV fsync 오류: {e}")
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

    # ── AVI 자동 분할 ─────────────────────────────────────────
    def _rotate_recording(self):
        logger.info(f"[{self.ip}] AVI 파일 분할")
        self._close_files()

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(OUTPUT_DIR, f"{self.ip.replace('.','_')}_{ts}")
        self._current_avi_path = base + ".avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._video_writer = cv2.VideoWriter(
            self._current_avi_path, fourcc, RECORD_FPS, (PREVIEW_W, PREVIEW_H))

        self._csv_file   = open(base + ".csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["timestamp", "roi_x", "roi_y", "roi_w", "roi_h",
             "min_temp", "avg_temp", "max_temp"])

        self._split_start_time = datetime.now()
        self._csv_frame_count  = 0
        logger.info(f"[{self.ip}] 새 녹화 파일: {self._current_avi_path}")

    def _write_frame(self, img: np.ndarray, min_v, avg_v, max_v):
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

        if self._video_writer is None:
            return

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if not self.roi.isNull():
            cv2.rectangle(bgr,
                (self.roi.x(), self.roi.y()),
                (self.roi.x() + self.roi.width(), self.roi.y() + self.roi.height()),
                (0, 255, 255), 2)
        self._video_writer.write(bgr)

        # CSV 기록 + 주기적 flush
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        roi = self.roi
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
                logger.error(f"CSV flush 오류: {e}")

    # ── 종료 처리 ─────────────────────────────────────────────
    def closeEvent(self, event):
        if self._recording:
            self._stop_recording()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        if self.camera:
            self.camera.close()
        logger.info(f"[종료] {self.ip}")
        event.accept()


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="열화상카메라 단일 녹화")
    parser.add_argument("--ip",     required=True,       help="카메라 IP")
    parser.add_argument("--name",   default=CAM_NAME,    help="카메라 모델명")
    parser.add_argument("--serial", default=CAM_SERIAL,  help="시리얼 번호")
    parser.add_argument("--mac",    default=CAM_MAC,     help="MAC 주소")
    args = parser.parse_args()

    logger.info(f"record_camera.py 시작  ip={args.ip}")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(args.ip, args.name, args.serial, args.mac)
    win.show()
    sys.exit(app.exec_())
