#################################################################
# File: raw_y16_recorder.py
#
# 실행:
#   python3 raw_y16_recorder.py --ip 192.168.0.151
#   python3 raw_y16_recorder.py --ip 192.168.0.151 --output ./raw_data
#   python3 raw_y16_recorder.py --ip 192.168.0.151 --native
#
# 기능:
#   - 지정 IP 카메라 1대에서 Raw Y16 (16-bit) 프레임을 비압축 저장
#   - 저장 포맷: .y16raw (바이너리) + .y16meta (JSON 메타데이터)
#   - 실시간 프리뷰 + 녹화 시작/정지
#   - 파일 자동 분할 (2 GB 초과 시)
#   - 녹화 후 → 별도 스크립트로 픽셀별 온도 맵 변환 가능
#
# 파일 구조 (.y16raw):
#   [frame0: width*height*2 bytes (uint16 LE)]
#   [frame1: width*height*2 bytes (uint16 LE)]
#   ...
#
# 메타데이터 (.y16meta, JSON):
#   { width, height, fps, frame_count, timestamps[], ... }
#
# 요구사항:
#   pip install PyQt5 opencv-python-headless numpy
#################################################################

import argparse
import json
import logging
import os
import struct
import sys
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

import cv2
import numpy as np

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFileDialog, QMessageBox
)

from TmCore import TmCamera
from TmCore.TmTypes import *


# ─────────────────────────────────────────────────────────────
# 로거 설정
# ─────────────────────────────────────────────────────────────
LOG_DIR = "logs"

def _setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger("TmSDK.RawY16")
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
        os.path.join(LOG_DIR, "raw_y16.log"),
        when="midnight", interval=1, backupCount=60, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(ch)
    log.addHandler(fh)
    return log

logger = _setup_logger()


# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
PREVIEW_W  = 720
PREVIEW_H  = 480
RECORD_FPS = 8.7
BASE_FONT_SIZE = 12

_SPLIT_MAX_BYTES = int(2 * 1024 ** 3)   # 2 GB 초과 시 파일 분할


# ─────────────────────────────────────────────────────────────
# 카메라 스캔 스레드
# ─────────────────────────────────────────────────────────────
class ScanWorker(QThread):
    scan_done = pyqtSignal(dict)

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
    connected = pyqtSignal(object)    # TmCamera
    failed    = pyqtSignal(str)

    def __init__(self, ip, cam_info=None):
        super().__init__()
        self.ip       = ip
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
    frame_ready = pyqtSignal(np.ndarray)           # RGB 프리뷰용
    raw_ready   = pyqtSignal(np.ndarray, str)      # (raw_uint16, timestamp)
    stats       = pyqtSignal(float, float, float)  # min, avg, max 전체 프레임

    def __init__(self, camera: TmCamera, query_w: int, query_h: int):
        super().__init__()
        self.camera   = camera
        self.query_w  = query_w
        self.query_h  = query_h
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True

        while self._running:
            try:
                frame = self.camera.query_frame(self.query_w, self.query_h)
                if frame is None:
                    QThread.msleep(10)
                    continue

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                fw = frame.width()
                fh = frame.height()

                # ── RGB 프리뷰 ──
                bitmap = frame.to_bitmap(ColorOrder.COLOR_RGB)
                img = np.frombuffer(bitmap, dtype=np.uint8).reshape(
                    fh, fw, 3).copy()

                # ── Raw Y16 추출 ──
                pixel_2d = frame.get_pixel(0, 0, fw, fh)

                if pixel_2d is not None:
                    raw = np.array(pixel_2d, dtype=np.uint16)
                    # get_pixel은 [w][h] 형태로 반환 → [h][w]로 전치
                    if raw.shape == (fw, fh):
                        raw = raw.T
                    self.raw_ready.emit(raw, ts)

                # ── 전체 프레임 통계 ──
                result = frame.min_max_loc()
                if result is not None:
                    min_val, avg_val, max_val = result[0], result[1], result[2]
                    min_t = self.camera.get_temperature(min_val)
                    avg_t = self.camera.get_temperature(avg_val)
                    max_t = self.camera.get_temperature(max_val)
                    self.stats.emit(min_t, avg_t, max_t)

                self.frame_ready.emit(img)
                del frame

            except Exception as e:
                logger.error(f"FrameWorker 오류: {e}")
                QThread.msleep(100)


# ─────────────────────────────────────────────────────────────
# Raw Y16 파일 기록기
# ─────────────────────────────────────────────────────────────
class RawY16Writer:
    """Raw Y16 프레임을 비압축 바이너리로 기록한다."""

    def __init__(self, base_path: str, width: int, height: int):
        self.base_path   = base_path
        self.width       = width
        self.height      = height
        self.frame_count = 0
        self.timestamps  = []
        self._file       = None
        self._part       = 0
        self._open_new_part()

    def _current_raw_path(self) -> str:
        if self._part == 0:
            return self.base_path + ".y16raw"
        return f"{self.base_path}_part{self._part}.y16raw"

    def _current_meta_path(self) -> str:
        if self._part == 0:
            return self.base_path + ".y16meta"
        return f"{self.base_path}_part{self._part}.y16meta"

    def _open_new_part(self):
        self._file = open(self._current_raw_path(), "wb")
        self.frame_count = 0
        self.timestamps  = []
        logger.info(f"Raw Y16 파일 생성: {self._current_raw_path()}")

    def write_frame(self, raw: np.ndarray, timestamp: str):
        raw_bytes = raw.astype(np.uint16).tobytes()
        self._file.write(raw_bytes)
        self.timestamps.append(timestamp)
        self.frame_count += 1

        # 분할 체크 (100 프레임마다)
        if self.frame_count % 100 == 0:
            try:
                fsize = os.path.getsize(self._current_raw_path())
            except OSError:
                fsize = 0
            if fsize >= _SPLIT_MAX_BYTES:
                self._flush_and_close()
                self._part += 1
                self._open_new_part()

    def _flush_and_close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        # 메타데이터 저장
        meta = {
            "format": "Y16_RAW_UINT16_LE",
            "width": self.width,
            "height": self.height,
            "bytes_per_pixel": 2,
            "bytes_per_frame": self.width * self.height * 2,
            "frame_count": self.frame_count,
            "fps": RECORD_FPS,
            "timestamps": self.timestamps,
        }
        meta_path = self._current_meta_path()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(
            f"메타데이터 저장: {meta_path}  "
            f"({self.frame_count} frames, "
            f"{self.width}x{self.height})")

    def close(self):
        self._flush_and_close()


# ─────────────────────────────────────────────────────────────
# 메인 윈도우
# ─────────────────────────────────────────────────────────────
class RawRecorderWindow(QWidget):
    def __init__(self, ip: str, output_dir: str, use_native: bool):
        super().__init__()
        self.setWindowTitle(f"Raw Y16 Recorder — {ip}")
        self._ip         = ip
        self._output_dir = output_dir
        self._use_native = use_native

        self.camera       = None
        self.worker       = None
        self._scan_worker = None
        self._conn_worker = None
        self._writer      = None
        self._recording   = False
        self._frame_w     = 0
        self._frame_h     = 0
        self._rec_count   = 0

        self._build_ui()
        self._start_connect()

    def _build_ui(self):
        font = QFont()
        font.setPointSize(BASE_FONT_SIZE)

        self.preview = QLabel("카메라 연결 중...")
        self.preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            "background: #1a1a1a; color: #888; font-size: 14px;")

        # 상태 정보
        self.lbl_status = QLabel(f"IP: {self._ip}  |  상태: 연결 중...")
        self.lbl_status.setFont(font)

        self.lbl_temp = QLabel("전체 프레임 — Min: —  |  Avg: —  |  Max: —")
        temp_font = QFont("Consolas", BASE_FONT_SIZE + 1)
        self.lbl_temp.setFont(temp_font)

        self.lbl_rec = QLabel("녹화 대기")
        self.lbl_rec.setFont(font)
        self.lbl_rec.setStyleSheet("color: #aaa;")

        # 저장 폴더
        lbl_dir = QLabel("저장 폴더:")
        lbl_dir.setFont(font)
        self.input_dir = QLineEdit(self._output_dir)
        self.input_dir.setFixedHeight(44)
        self.input_dir.setFont(font)
        self.input_dir.textChanged.connect(
            lambda t: setattr(self, "_output_dir", t.strip() or "raw_output"))

        btn_browse = QPushButton("탐색...")
        btn_browse.setFixedSize(90, 44)
        btn_browse.setFont(font)
        btn_browse.clicked.connect(self._browse)

        dir_row = QHBoxLayout()
        dir_row.addWidget(lbl_dir)
        dir_row.addWidget(self.input_dir, stretch=1)
        dir_row.addWidget(btn_browse)

        # 녹화 버튼
        self.btn_record = QPushButton("● Raw Y16 녹화 시작")
        self.btn_record.setFixedHeight(60)
        rec_font = QFont()
        rec_font.setPointSize(BASE_FONT_SIZE + 2)
        rec_font.setBold(True)
        self.btn_record.setFont(rec_font)
        self.btn_record.setEnabled(False)
        self._set_btn_idle()
        self.btn_record.clicked.connect(self._toggle_record)

        layout = QVBoxLayout()
        layout.addWidget(self.preview)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_temp)
        layout.addWidget(self.lbl_rec)
        layout.addLayout(dir_row)
        layout.addWidget(self.btn_record)
        self.setLayout(layout)
        self.resize(PREVIEW_W + 40, PREVIEW_H + 300)

    def _set_btn_idle(self):
        self.btn_record.setText("● Raw Y16 녹화 시작")
        self.btn_record.setStyleSheet(
            "QPushButton { background-color: #2d6a2d; color: white;"
            " border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a8a3a; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )

    def _set_btn_recording(self):
        self.btn_record.setText("■ 녹화 정지")
        self.btn_record.setStyleSheet(
            "QPushButton { background-color: #8b1a1a; color: white;"
            " border-radius: 4px; }"
            "QPushButton:hover { background-color: #b52020; }"
        )

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "저장 폴더 선택", self._output_dir)
        if path:
            self.input_dir.setText(path)

    # ── 카메라 연결 ───────────────────────────────────────────
    def _start_connect(self):
        self._scan_worker = ScanWorker()
        self._scan_worker.scan_done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, cam_map: dict):
        cam_info = cam_map.get(self._ip)
        self._conn_worker = ConnectWorker(self._ip, cam_info)
        self._conn_worker.connected.connect(self._on_connected)
        self._conn_worker.failed.connect(self._on_failed)
        self._conn_worker.start()

    def _on_connected(self, cam: TmCamera):
        self.camera = cam
        self.camera.set_temp_unit(TempUnit.CELSIUS)
        self.camera.set_color_map(ColormapTypes.Inferno + 1)

        cam_fmt = self.camera.get_camera_format()
        if cam_fmt != "Y16":
            QMessageBox.warning(
                self, "포맷 경고",
                f"카메라 포맷이 '{cam_fmt}'입니다.\n"
                "Raw Y16 녹화는 Y16 포맷에서만 의미 있습니다.")

        # 프리뷰 크기 결정: --native 옵션이면 카메라 원본 해상도 사용
        if self._use_native:
            # 첫 프레임으로 원본 해상도 확인
            test_frame = self.camera.query_frame(PREVIEW_W, PREVIEW_H)
            if test_frame:
                self._frame_w = test_frame.width()
                self._frame_h = test_frame.height()
                del test_frame
            else:
                self._frame_w = PREVIEW_W
                self._frame_h = PREVIEW_H
        else:
            self._frame_w = PREVIEW_W
            self._frame_h = PREVIEW_H

        frame_bytes = self._frame_w * self._frame_h * 2
        self.lbl_status.setText(
            f"IP: {self._ip}  |  포맷: {cam_fmt}  |  "
            f"해상도: {self._frame_w}x{self._frame_h}  |  "
            f"프레임당: {frame_bytes:,} bytes ({frame_bytes/1024:.1f} KB)")

        self.btn_record.setEnabled(True)
        self.preview.setText("")

        self.worker = FrameWorker(
            self.camera, self._frame_w, self._frame_h)
        self.worker.frame_ready.connect(self._on_preview)
        self.worker.raw_ready.connect(self._on_raw_frame)
        self.worker.stats.connect(self._on_stats)
        self.worker.start()
        logger.info(f"[연결 성공] {self._ip}  {self._frame_w}x{self._frame_h}")

    def _on_failed(self, ip: str):
        self.lbl_status.setText(f"IP: {ip}  |  상태: 연결 실패")
        self.preview.setText(f"연결 실패\n{ip}")
        logger.error(f"[연결 실패] {ip}")

    # ── 프리뷰 ────────────────────────────────────────────────
    def _on_preview(self, img: np.ndarray):
        h, w, _ = img.shape
        # 프리뷰 크기에 맞춰 리사이즈
        if w != PREVIEW_W or h != PREVIEW_H:
            img = cv2.resize(img, (PREVIEW_W, PREVIEW_H))
            h, w = PREVIEW_H, PREVIEW_W
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(qimg))

    def _on_stats(self, min_t, avg_t, max_t):
        sym = self.camera.get_temp_unit_symbol() if self.camera else ""
        self.lbl_temp.setText(
            f"전체 프레임 — Min: {min_t:.2f} {sym}  |  "
            f"Avg: {avg_t:.2f} {sym}  |  Max: {max_t:.2f} {sym}")

    # ── Raw 프레임 수신 ───────────────────────────────────────
    def _on_raw_frame(self, raw: np.ndarray, timestamp: str):
        if self._recording and self._writer:
            self._writer.write_frame(raw, timestamp)
            self._rec_count += 1
            if self._rec_count % 10 == 0:
                elapsed = (datetime.now() -
                           self._rec_start).total_seconds()
                size_mb = (self._rec_count *
                           self._frame_w * self._frame_h * 2) / (1024**2)
                self.lbl_rec.setText(
                    f"녹화 중: {self._rec_count} frames  |  "
                    f"{elapsed:.0f}초  |  ~{size_mb:.0f} MB")

    # ── 녹화 제어 ─────────────────────────────────────────────
    def _toggle_record(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        os.makedirs(self._output_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(
            self._output_dir,
            f"{self._ip.replace('.','_')}_{ts}")

        self._writer = RawY16Writer(base, self._frame_w, self._frame_h)
        self._recording  = True
        self._rec_count  = 0
        self._rec_start  = datetime.now()
        self._set_btn_recording()
        self.lbl_rec.setStyleSheet(f"color: #ff4444; font-size: {BASE_FONT_SIZE}px;")
        self.lbl_rec.setText("녹화 중: 0 frames")
        logger.info(f"Raw Y16 녹화 시작: {base}")

    def _stop_recording(self):
        self._recording = False
        if self._writer:
            self._writer.close()
            self._writer = None
        self._set_btn_idle()
        self.lbl_rec.setStyleSheet(f"color: #aaa; font-size: {BASE_FONT_SIZE}px;")
        elapsed = (datetime.now() - self._rec_start).total_seconds()
        size_mb = (self._rec_count *
                   self._frame_w * self._frame_h * 2) / (1024**2)
        self.lbl_rec.setText(
            f"녹화 완료: {self._rec_count} frames  |  "
            f"{elapsed:.0f}초  |  ~{size_mb:.0f} MB")
        logger.info(
            f"Raw Y16 녹화 정지: {self._rec_count} frames, "
            f"{size_mb:.1f} MB, {elapsed:.1f}초")

    # ── 종료 ──────────────────────────────────────────────────
    def closeEvent(self, event):
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
        logger.info("Raw Y16 Recorder 종료")
        event.accept()


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="TmSDK Raw Y16 Recorder — 열화상 카메라 원본 데이터 비압축 저장")
    parser.add_argument(
        "--ip", required=True,
        help="카메라 IP 주소 (예: 192.168.0.151)")
    parser.add_argument(
        "--output", default="raw_output",
        help="저장 폴더 (기본: raw_output)")
    parser.add_argument(
        "--native", action="store_true",
        help="카메라 원본 해상도로 저장 (기본: 720x480)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Raw Y16 Recorder 시작  IP={args.ip}  output={args.output}")
    logger.info("=" * 60)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RawRecorderWindow(args.ip, args.output, args.native)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
