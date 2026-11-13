import os

import torch
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGroupBox,
    QSizePolicy, QStatusBar, QComboBox, QCheckBox, QSpinBox,
    QMessageBox, QFrame, QLineEdit, QMdiArea, QMdiSubWindow
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QBrush, QColor


class _SourceSubWindow(QMdiSubWindow):
    """닫기(X) 시 소스 정리를 트리거하는 MDI 서브창."""

    closed = pyqtSignal(int)   # source_id

    def __init__(self, sid: int):
        super().__init__()
        self.sid = sid

    def closeEvent(self, event):
        # 실제 정리는 MainWindow._remove_source 가 수행
        self.closed.emit(self.sid)
        event.ignore()


class ClickSeekSlider(QSlider):
    """클릭한 위치로 즉시 이동하는 슬라이더 (기본 QSlider 는 페이지 단위로만 점프)."""

    clicked_value = pyqtSignal(int)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.maximum() > self.minimum():
            ratio = ev.x() / max(1, self.width())
            val = self.minimum() + round(ratio * (self.maximum() - self.minimum()))
            self.setValue(int(val))
            self.clicked_value.emit(int(val))
            ev.accept()
        super().mousePressEvent(ev)

from core.video_thread import VideoThread

# CPU 전용 환경(Raspberry Pi 5 등)은 부하 때문에 동시 소스 수를 제한한다
_HAS_CUDA = torch.cuda.is_available()
MAX_SOURCES = 9 if _HAS_CUDA else 4


def _default_kpt_behavior_model_path() -> str:
    """학습 후 생성되는 키포인트 행동 분류 모델 기본 경로."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "train", "1_behavior", "output", "kpt_behavior.pth")
    return path if os.path.isfile(path) else ""


def _default_stand_model_path() -> str:
    """가판대·결제기 감지 모델 기본 경로 (objects.pt 우선, 구버전 stand.pt 폴백)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("objects.pt", "stand.pt"):
        path = os.path.join(root, "model", name)
        if os.path.isfile(path):
            return path
    return ""


class VideoDisplay(QLabel):
    """영상 출력 라벨 — 종횡비 유지하며 스케일."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a2e;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 180)
        self._pixmap = None

    def set_frame(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()

    def _rescale(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)


class VideoTile(QFrame):
    """다중 소스 그리드의 단일 타일 — 제목줄 + 영상 + 상태줄."""

    close_requested = pyqtSignal(int)   # source_id
    seek_requested = pyqtSignal(int, float)   # (source_id, 0.0~1.0)
    pause_toggled = pyqtSignal(int, bool)     # (source_id, paused)

    def __init__(self, source_id: int, title: str):
        super().__init__()
        self.source_id = source_id
        self._user_seeking = False
        self._paused = False
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "VideoTile { border: 1px solid #2e2e4e; border-radius: 6px; "
            "background-color: #16162a; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #aaaacc; font-size: 11px; font-weight: bold;")
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(20, 20)
        btn_close.setToolTip("이 소스 닫기")
        btn_close.clicked.connect(lambda: self.close_requested.emit(self.source_id))
        header.addWidget(self.lbl_title)
        header.addStretch()
        header.addWidget(btn_close)
        layout.addLayout(header)

        self.display = VideoDisplay()
        self.display.setText("연결 중...")
        self.display.setStyleSheet(
            "background-color: #1a1a2e; color: #8888aa; border-radius: 4px;"
        )
        layout.addWidget(self.display, stretch=1)

        # 컨트롤 행: [일시정지] [재생바(파일 소스)] [시간]
        seek_row = QHBoxLayout()
        seek_row.setSpacing(4)

        # 영상별 일시정지/재생 버튼 (전체 제어 대신 소스 단위)
        self.btn_pause = QPushButton("⏸")
        self.btn_pause.setFixedSize(28, 22)
        self.btn_pause.setToolTip("이 영상 일시정지/재생")
        self.btn_pause.clicked.connect(self._on_pause_clicked)

        self.slider_seek = ClickSeekSlider(Qt.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.setValue(0)
        self.slider_seek.setVisible(False)
        self.slider_seek.sliderPressed.connect(self._on_seek_pressed)
        self.slider_seek.sliderReleased.connect(self._on_seek_released)
        # 트랙의 임의 위치 클릭 → 즉시 그 지점에서 재생
        self.slider_seek.clicked_value.connect(self._on_seek_clicked)

        self.lbl_time = QLabel("")
        self.lbl_time.setStyleSheet("color: #8888aa; font-size: 10px;")
        self.lbl_time.setVisible(False)

        seek_row.addWidget(self.btn_pause)
        seek_row.addWidget(self.slider_seek, stretch=1)
        seek_row.addWidget(self.lbl_time)
        layout.addLayout(seek_row)

        self.lbl_status = QLabel("인원 — | FPS —")
        self.lbl_status.setStyleSheet("color: #8888aa; font-size: 10px;")
        layout.addWidget(self.lbl_status)

    def set_stats(self, count: int, fps: float):
        self.lbl_status.setText(f"인원 {count} | FPS {fps:.1f}")

    def set_finished(self):
        self.lbl_status.setText("재생 완료")

    # ----------------------------------------------------------- 재생바(탐색)
    def enable_seek(self):
        """동영상 파일 소스에 대해 재생바를 표시한다."""
        self.slider_seek.setVisible(True)
        self.lbl_time.setVisible(True)

    def set_progress(self, cur: int, total: int):
        """재생 진행률 갱신 — 사용자가 드래그 중일 때는 건드리지 않는다."""
        if total <= 0 or self._user_seeking:
            return
        self.slider_seek.setValue(int(cur / total * 1000))
        self.lbl_time.setText(f"{cur}/{total}")

    def _on_seek_pressed(self):
        self._user_seeking = True

    def _on_seek_released(self):
        self._user_seeking = False
        frac = self.slider_seek.value() / 1000.0
        self.seek_requested.emit(self.source_id, frac)

    def _on_seek_clicked(self, value: int):
        # 트랙 클릭 즉시 해당 위치로 이동
        self._user_seeking = False
        self.seek_requested.emit(self.source_id, value / 1000.0)

    def _on_pause_clicked(self):
        self._paused = not self._paused
        self.btn_pause.setText("▶" if self._paused else "⏸")
        self.pause_toggled.emit(self.source_id, self._paused)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV 이상 현상 감지 시스템")
        self.resize(1280, 780)
        # source_id → {"thread", "tile", "title", "count", "fps", "behavior"}
        self._sources: dict[int, dict] = {}
        self._next_id = 0
        self._build_ui()
        self._apply_stylesheet()

    # ------------------------------------------------------------------ UI 구성
    def _build_ui(self):
        self._build_menubar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_video_panel(), stretch=4)
        root.addWidget(self._build_control_panel(), stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("대기 중 — 파일, 웹캠 또는 IP 카메라를 추가하세요.")

    def _build_menubar(self):
        menubar = self.menuBar()
        rec_menu = menubar.addMenu("녹화 기록")
        act_open = rec_menu.addAction("녹화 기록 보기")
        act_open.triggered.connect(self._open_recordings)

    def _open_recordings(self):
        from gui.recordings_dialog import RecordingsDialog
        dlg = RecordingsDialog(self)
        dlg.exec_()

    def _build_video_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 다중 소스 — 자유 이동·크기조절 가능한 MDI 서브창
        self.mdi = QMdiArea()
        self.mdi.setBackground(QBrush(QColor("#12121e")))
        self.mdi.setViewMode(QMdiArea.SubWindowView)
        self.mdi.setOption(QMdiArea.DontMaximizeSubWindowOnActivation, True)
        layout.addWidget(self.mdi, stretch=1)

        # 창 정렬 보조 버튼 (위치는 자유롭게 조정 가능, 한 번에 정돈용)
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        self.btn_tile = QPushButton("바둑판 정렬")
        self.btn_tile.setFixedHeight(28)
        self.btn_tile.clicked.connect(lambda: self.mdi.tileSubWindows())
        self.btn_cascade = QPushButton("계단식 정렬")
        self.btn_cascade.setFixedHeight(28)
        self.btn_cascade.clicked.connect(lambda: self.mdi.cascadeSubWindows())
        ctrl.addWidget(self.btn_tile)
        ctrl.addWidget(self.btn_cascade)
        ctrl.addStretch()
        layout.addLayout(ctrl)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_stats_group())
        layout.addStretch()
        return panel

    def _build_source_group(self) -> QGroupBox:
        grp = QGroupBox("영상 소스 (다중 연결)")
        layout = QVBoxLayout(grp)

        self.btn_open_file = QPushButton("파일 추가...")
        self.btn_open_file.setFixedHeight(34)
        self.btn_open_file.clicked.connect(self._on_open_file)

        # 웹캠 — 인덱스 선택 후 추가 (여러 대 동시 연결 가능)
        lbl_cam = QLabel("웹캠 (인덱스 선택 후 추가)")
        lbl_cam.setStyleSheet("color: #aaaacc; font-size: 11px; margin-top: 4px;")
        cam_row = QHBoxLayout()
        self.spin_cam = QSpinBox()
        self.spin_cam.setRange(0, 7)
        self.spin_cam.setValue(0)
        self.btn_webcam = QPushButton("웹캠 추가")
        self.btn_webcam.setFixedHeight(28)
        self.btn_webcam.clicked.connect(self._on_add_webcam)
        cam_row.addWidget(self.spin_cam)
        cam_row.addWidget(self.btn_webcam, stretch=1)

        # IP 카메라 RTSP
        lbl_rtsp = QLabel("IP 카메라 (RTSP)")
        lbl_rtsp.setStyleSheet("color: #aaaacc; font-size: 11px; margin-top: 4px;")
        rtsp_row = QHBoxLayout()
        self.edit_rtsp = QLineEdit()
        self.edit_rtsp.setPlaceholderText("rtsp://user:pass@ip:port/path")
        self.btn_rtsp = QPushButton("추가")
        self.btn_rtsp.setFixedWidth(46)
        self.btn_rtsp.setFixedHeight(28)
        self.btn_rtsp.clicked.connect(self._on_rtsp)
        rtsp_row.addWidget(self.edit_rtsp)
        rtsp_row.addWidget(self.btn_rtsp)

        self.lbl_source = QLabel(f"연결된 소스: 0 / {MAX_SOURCES}")
        self.lbl_source.setStyleSheet("color: #aaaaaa; font-size: 11px;")

        layout.addWidget(self.btn_open_file)
        layout.addWidget(lbl_cam)
        layout.addLayout(cam_row)
        layout.addWidget(lbl_rtsp)
        layout.addLayout(rtsp_row)
        layout.addWidget(self.lbl_source)
        return grp

    def _build_model_group(self) -> QGroupBox:
        grp = QGroupBox("모델 설정 (이상행동 감지)")
        layout = QVBoxLayout(grp)

        # 감지 모델 — YOLO pose (인물 감지 + 17관절). 작은 모델일수록 경량 PC 친화적.
        lbl_det = QLabel("감지 모델 (포즈)")
        lbl_det.setStyleSheet("color: #aaaacc; font-size: 11px;")
        layout.addWidget(lbl_det)
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "yolo11n-pose.pt  (경량·권장)",
            "yolo11s-pose.pt  (균형)",
            "yolo11m-pose.pt  (정확)",
        ])
        self.combo_model.setCurrentIndex(0)
        layout.addWidget(self.combo_model)

        # 이상행동 분류 모델 (KeypointLSTM, kpt_behavior.pth)
        lbl_kpt_beh = QLabel("이상행동 분류 모델 (kpt_behavior.pth)")
        lbl_kpt_beh.setStyleSheet("color: #aaaacc; font-size: 11px; margin-top: 2px;")
        layout.addWidget(lbl_kpt_beh)
        kpt_beh_row = QHBoxLayout()
        self.edit_kpt_behavior_model = QLineEdit()
        self.edit_kpt_behavior_model.setPlaceholderText("kpt_behavior.pth 경로...")
        self.edit_kpt_behavior_model.setText(_default_kpt_behavior_model_path())
        self.btn_browse_kpt_behavior = QPushButton("찾기")
        self.btn_browse_kpt_behavior.setFixedWidth(46)
        self.btn_browse_kpt_behavior.clicked.connect(self._on_browse_kpt_behavior)
        kpt_beh_row.addWidget(self.edit_kpt_behavior_model)
        kpt_beh_row.addWidget(self.btn_browse_kpt_behavior)
        layout.addLayout(kpt_beh_row)

        # 처리 해상도 — 작을수록 빠름 (경량 PC 최적화)
        lbl_imgsz = QLabel("처리 해상도 (경량화)")
        lbl_imgsz.setStyleSheet("color: #aaaacc; font-size: 11px; margin-top: 2px;")
        layout.addWidget(lbl_imgsz)
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems([
            "384  (가장 빠름)",
            "480  (빠름·권장)",
            "640  (정확)",
        ])
        # CPU 전용(Raspberry Pi)은 384 기본, CUDA PC는 480 기본
        self.combo_imgsz.setCurrentIndex(1 if _HAS_CUDA else 0)
        layout.addWidget(self.combo_imgsz)

        # 검출 간격 — N프레임마다 YOLO 실행 (사이 프레임은 직전 결과 재사용)
        lbl_interval = QLabel("검출 간격 (다중 카메라 경량화)")
        lbl_interval.setStyleSheet("color: #aaaacc; font-size: 11px; margin-top: 2px;")
        layout.addWidget(lbl_interval)
        self.combo_interval = QComboBox()
        self.combo_interval.addItems([
            "1  (매 프레임)",
            "2  (1프레임 건너뜀)",
            "3  (2프레임 건너뜀)",
        ])
        self.combo_interval.setCurrentIndex(0 if _HAS_CUDA else 1)
        layout.addWidget(self.combo_interval)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #2e2e4e;")
        layout.addWidget(line)

        lbl_conf = QLabel("신뢰도 임계값")
        layout.addWidget(lbl_conf)

        conf_row = QHBoxLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        self.lbl_conf_val = QLabel("0.50")
        self.lbl_conf_val.setFixedWidth(36)
        conf_row.addWidget(self.slider_conf)
        conf_row.addWidget(self.lbl_conf_val)
        layout.addLayout(conf_row)

        layout.addWidget(QLabel("DeepSORT 최대 소실 프레임"))
        self.spin_max_age = QSpinBox()
        self.spin_max_age.setRange(1, 120)
        self.spin_max_age.setValue(30)
        layout.addWidget(self.spin_max_age)

        return grp

    def _build_display_group(self) -> QGroupBox:
        grp = QGroupBox("표시 옵션")
        layout = QVBoxLayout(grp)
        self.chk_bbox = QCheckBox("바운딩 박스")
        self.chk_bbox.setChecked(True)
        self.chk_track_id = QCheckBox("트랙 ID")
        self.chk_track_id.setChecked(True)
        self.chk_keypoints = QCheckBox("관절 키포인트 (포즈 모델)")
        self.chk_keypoints.setChecked(True)
        self.chk_kpt_behavior = QCheckBox("키포인트 행동 분류")
        self.chk_kpt_behavior.setChecked(True)
        # 가판대·결제기를 각각 독립 토글 (model/objects.pt 있을 때만 활성)
        _has_obj = bool(_default_stand_model_path())
        self.chk_stand = QCheckBox("물품 가판대 감지")
        self.chk_pos = QCheckBox("결제기(POS) 감지")
        for chk in (self.chk_stand, self.chk_pos):
            chk.setChecked(_has_obj)
            chk.setEnabled(_has_obj)
            if not _has_obj:
                chk.setToolTip("model/objects.pt 가 없습니다 (객체 모델 미학습)")
        layout.addWidget(self.chk_bbox)
        layout.addWidget(self.chk_track_id)
        layout.addWidget(self.chk_keypoints)
        layout.addWidget(self.chk_kpt_behavior)
        layout.addWidget(self.chk_stand)
        layout.addWidget(self.chk_pos)

        # 체크박스 → 실행 중인 모든 스레드에 실시간 반영
        # bool 단순 속성 쓰기는 GIL이 원자성을 보장하므로 별도 락 불필요
        self.chk_bbox.stateChanged.connect(
            lambda s: self._set_thread_attr('show_bbox', bool(s)))
        self.chk_track_id.stateChanged.connect(
            lambda s: self._set_thread_attr('show_track_id', bool(s)))
        self.chk_keypoints.stateChanged.connect(
            lambda s: self._set_thread_attr('show_keypoints', bool(s)))
        self.chk_kpt_behavior.stateChanged.connect(
            lambda s: self._set_thread_attr('show_kpt_behavior', bool(s)))
        self.chk_stand.stateChanged.connect(
            lambda s: self._set_thread_attr('show_stand', bool(s)))
        self.chk_pos.stateChanged.connect(
            lambda s: self._set_thread_attr('show_pos', bool(s)))
        return grp

    def _build_stats_group(self) -> QGroupBox:
        grp = QGroupBox("실시간 통계 (전체 합산)")
        layout = QVBoxLayout(grp)

        self.lbl_count = QLabel("감지된 인원: 0명")
        self.lbl_count.setFont(QFont("Malgun Gothic", 13, QFont.Bold))
        self.lbl_count.setStyleSheet("color: #00d4aa;")

        self.lbl_behavior = QLabel("이상행동: —")
        self.lbl_behavior.setWordWrap(True)
        self.lbl_behavior.setFont(QFont("Malgun Gothic", 13, QFont.Bold))
        self.lbl_behavior.setStyleSheet("color: #00d4aa;")

        self.lbl_fps = QLabel("FPS: —")
        self.lbl_fps.setFont(QFont("Malgun Gothic", 11))
        self.lbl_fps.setStyleSheet("color: #aaaaaa;")

        layout.addWidget(self.lbl_count)
        layout.addWidget(self.lbl_behavior)
        layout.addWidget(self.lbl_fps)
        return grp

    # ------------------------------------------------------------------ 소스 추가 이벤트
    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "동영상 파일 선택", "",
            "동영상 파일 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;모든 파일 (*)"
        )
        if path:
            self._add_source(path, f"파일: {os.path.basename(path)}")

    def _on_add_webcam(self):
        idx = self.spin_cam.value()
        for src in self._sources.values():
            if src["thread"].source == idx:
                QMessageBox.information(
                    self, "이미 연결됨", f"웹캠 {idx} 는 이미 연결되어 있습니다.")
                return
        self._add_source(idx, f"웹캠 {idx}")

    def _on_rtsp(self):
        url = self.edit_rtsp.text().strip()
        if not url:
            QMessageBox.warning(self, "URL 없음", "RTSP URL을 입력하세요.")
            return
        if not url.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            QMessageBox.warning(self, "URL 형식 오류", "rtsp:// 로 시작하는 URL을 입력하세요.")
            return
        for src in self._sources.values():
            if src["thread"].source == url:
                QMessageBox.information(self, "이미 연결됨", "해당 RTSP 소스는 이미 연결되어 있습니다.")
                return
        self._add_source(url, f"IP 카메라: {url.split('@')[-1][:30]}")

    # ------------------------------------------------------------------ 재생 제어
    def _on_pause_toggled(self, sid: int, paused: bool):
        src = self._sources.get(sid)
        if not src:
            return
        if paused:
            src["thread"].pause()
        else:
            src["thread"].resume()

    def _on_browse_kpt_behavior(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "키포인트 행동 분류 모델 선택", "",
            "PyTorch 모델 (*.pth);;모든 파일 (*)"
        )
        if path:
            self.edit_kpt_behavior_model.setText(path)

    def _on_conf_changed(self, value: int):
        conf = value / 100.0
        self.lbl_conf_val.setText(f"{conf:.2f}")
        for src in self._sources.values():
            src["thread"].set_conf(conf)

    def _set_thread_attr(self, name: str, value: bool):
        for src in self._sources.values():
            setattr(src["thread"], name, value)

    # ------------------------------------------------------------------ 스레드 콜백
    def _on_frame(self, sid: int, qimg):
        src = self._sources.get(sid)
        if src:
            src["tile"].display.set_frame(QPixmap.fromImage(qimg))

    def _on_stats(self, sid: int, count: int, fps: float):
        src = self._sources.get(sid)
        if src is None:
            return
        src["count"], src["fps"] = count, fps
        src["tile"].set_stats(count, fps)
        self._update_aggregate_stats()

    def _on_behavior(self, sid: int, class_id: int, conf: float):
        src = self._sources.get(sid)
        if src is None:
            return
        src["behavior"] = (class_id, conf)
        self._update_behavior_label()

    def _on_progress(self, sid: int, cur: int, total: int):
        src = self._sources.get(sid)
        if src:
            src["tile"].set_progress(cur, total)

    def _on_error(self, sid: int, msg: str):
        src = self._sources.get(sid)
        title = src["title"] if src else f"소스 {sid}"
        self._remove_source(sid)
        QMessageBox.critical(self, "오류", f"[{title}]\n{msg}")

    def _on_thread_finished(self, sid: int):
        src = self._sources.get(sid)
        if src:
            src["tile"].set_finished()

    # ------------------------------------------------------------------ 집계 표시
    def _update_aggregate_stats(self):
        total = sum(s["count"] for s in self._sources.values())
        fps_list = [s["fps"] for s in self._sources.values() if s["fps"] > 0]
        self.lbl_count.setText(f"감지된 인원: {total}명")
        if fps_list:
            self.lbl_fps.setText(f"FPS: {sum(fps_list) / len(fps_list):.1f} (평균)")
        else:
            self.lbl_fps.setText("FPS: —")

    def _update_behavior_label(self):
        from core.labels import label_kr
        # 모든 소스 중 가장 신뢰도 높은 이상행동을 대표로 표시
        best_abn = None     # (conf, class_id, title)
        best_norm = None    # (conf,)
        for src in self._sources.values():
            cid, conf = src["behavior"]
            if cid > 0:
                if best_abn is None or conf > best_abn[0]:
                    best_abn = (conf, cid, src["title"])
            elif cid == 0:
                if best_norm is None or conf > best_norm[0]:
                    best_norm = (conf,)

        if best_abn is not None:
            conf, cid, title = best_abn
            self.lbl_behavior.setText(
                f"이상행동: {label_kr(cid)} ({conf * 100:.0f}%) — {title}")
            self.lbl_behavior.setStyleSheet("color: #ff4d4d;")
        elif best_norm is not None:
            self.lbl_behavior.setText(f"이상행동: 정상 ({best_norm[0] * 100:.0f}%)")
            self.lbl_behavior.setStyleSheet("color: #00d4aa;")
        elif self._sources:
            self.lbl_behavior.setText("이상행동: 분석 중...")
            self.lbl_behavior.setStyleSheet("color: #aaaaaa;")
        else:
            self.lbl_behavior.setText("이상행동: —")
            self.lbl_behavior.setStyleSheet("color: #00d4aa;")

    # ------------------------------------------------------------------ 소스 관리
    def _resolve_model_path(self) -> str:
        # 콤보 항목은 "yolo11n-pose.pt  (경량·권장)" 형태 → 첫 토큰이 실제 파일명
        return self.combo_model.currentText().split()[0]

    def _resolve_imgsz(self) -> int:
        return int(self.combo_imgsz.currentText().split()[0])

    def _resolve_interval(self) -> int:
        return int(self.combo_interval.currentText().split()[0])

    def _add_source(self, source, title: str):
        if len(self._sources) >= MAX_SOURCES:
            QMessageBox.warning(
                self, "소스 한도 초과",
                f"동시 연결은 최대 {MAX_SOURCES}개까지 지원합니다."
                + ("" if _HAS_CUDA else "\n(CPU 전용 환경 부하 제한)"))
            return

        model = self._resolve_model_path()
        conf = self.slider_conf.value() / 100.0
        max_age = self.spin_max_age.value()
        imgsz = self._resolve_imgsz()
        interval = self._resolve_interval()

        # 키포인트 행동 분류 모델 경로 (파일이 존재할 때만 전달)
        kpt_model = self.edit_kpt_behavior_model.text().strip()
        if not (kpt_model and os.path.isfile(kpt_model)):
            kpt_model = ""
            self.status_bar.showMessage(
                "이상행동 모델(kpt_behavior.pth) 미지정 — 감지·추적만 동작합니다.")

        # 물품 가판대 모델 경로 (model/stand.pt 존재 시 전달, 표시는 체크박스로 토글)
        stand_model = _default_stand_model_path()

        sid = self._next_id
        self._next_id += 1

        thread = VideoThread(source, model, conf, max_age,
                             kpt_model_path=kpt_model, imgsz=imgsz,
                             detect_interval=interval, stand_model_path=stand_model,
                             source_name=title)
        thread.show_bbox = self.chk_bbox.isChecked()
        thread.show_track_id = self.chk_track_id.isChecked()
        thread.show_keypoints = self.chk_keypoints.isChecked()
        thread.show_kpt_behavior = self.chk_kpt_behavior.isChecked()
        thread.show_stand = self.chk_stand.isChecked()
        thread.show_pos = self.chk_pos.isChecked()

        tile = VideoTile(sid, title)
        tile.close_requested.connect(self._remove_source)
        tile.pause_toggled.connect(self._on_pause_toggled)

        # sid 를 기본 인자로 바인딩해 소스별 콜백 분기
        thread.frame_ready.connect(lambda img, s=sid: self._on_frame(s, img))
        thread.stats_updated.connect(lambda c, f, s=sid: self._on_stats(s, c, f))
        thread.behavior_ready.connect(lambda cid, cf, s=sid: self._on_behavior(s, cid, cf))
        thread.error_occurred.connect(lambda m, s=sid: self._on_error(s, m))
        thread.finished_signal.connect(lambda s=sid: self._on_thread_finished(s))

        # 동영상 파일 소스만 재생바(탐색) 활성화
        if thread.is_file:
            tile.enable_seek()
            thread.progress_updated.connect(
                lambda cur, total, s=sid: self._on_progress(s, cur, total))
            tile.seek_requested.connect(
                lambda s, frac: self._sources[s]["thread"].seek_to_fraction(frac)
                if s in self._sources else None)

        # 자유 이동·크기조절 가능한 MDI 서브창에 담는다
        sub = _SourceSubWindow(sid)
        sub.setWidget(tile)
        sub.setWindowTitle(title)
        sub.closed.connect(self._remove_source)
        self.mdi.addSubWindow(sub)
        sub.resize(520, 400)
        sub.show()

        self._sources[sid] = {
            "thread": thread, "tile": tile, "subwindow": sub, "title": title,
            "count": 0, "fps": 0.0, "behavior": (-1, 0.0),
        }

        thread.start()
        self.status_bar.showMessage(f"{title} 연결됨 — 처리 중...")
        self._update_source_count()

    def _remove_source(self, sid: int):
        src = self._sources.pop(sid, None)
        if src is None:
            return
        thread = src["thread"]
        if thread.isRunning():
            thread.stop()
        sub = src.get("subwindow")
        if sub is not None:
            self.mdi.removeSubWindow(sub)
            sub.deleteLater()
        src["tile"].deleteLater()

        self._update_source_count()
        self._update_aggregate_stats()
        self._update_behavior_label()
        if not self._sources:
            self.lbl_count.setText("감지된 인원: 0명")
            self.lbl_fps.setText("FPS: —")

    def _update_source_count(self):
        self.lbl_source.setText(f"연결된 소스: {len(self._sources)} / {MAX_SOURCES}")

    def closeEvent(self, event):
        for sid in list(self._sources):
            self._remove_source(sid)
        event.accept()

    # ------------------------------------------------------------------ 스타일
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #12121e;
                color: #e0e0f0;
                font-family: 'Malgun Gothic', sans-serif;
            }
            QGroupBox {
                border: 1px solid #2e2e4e;
                border-radius: 6px;
                margin-top: 10px;
                padding: 6px;
                font-weight: bold;
                color: #aaaacc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2a2a4a;
                color: #e0e0f0;
                border: 1px solid #3a3a6a;
                border-radius: 5px;
                padding: 4px 10px;
            }
            QPushButton:hover { background-color: #3a3a6a; }
            QPushButton:pressed { background-color: #1a1a3a; }
            QPushButton:disabled { color: #555566; border-color: #2a2a4a; }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2e2e4e;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00d4aa;
                width: 14px; height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #00d4aa; border-radius: 2px; }
            QComboBox {
                background-color: #1e1e3e;
                border: 1px solid #3a3a6a;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0f0;
            }
            QComboBox::drop-down { border: none; }
            QSpinBox {
                background-color: #1e1e3e;
                border: 1px solid #3a3a6a;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0f0;
            }
            QCheckBox { color: #c0c0e0; }
            QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px;
                border: 1px solid #3a3a6a; background: #1e1e3e; }
            QCheckBox::indicator:checked { background: #00d4aa; border-color: #00d4aa; }
            QRadioButton { color: #c0c0e0; }
            QRadioButton::indicator { width: 13px; height: 13px; border-radius: 7px;
                border: 1px solid #3a3a6a; background: #1e1e3e; }
            QRadioButton::indicator:checked { background: #00d4aa; border-color: #00d4aa; }
            QRadioButton:disabled { color: #555566; }
            QLineEdit {
                background-color: #1e1e3e;
                border: 1px solid #3a3a6a;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0f0;
            }
            QLineEdit:disabled { color: #555566; border-color: #2a2a4a; }
            QLineEdit:focus { border-color: #00d4aa; }
            QStatusBar { background-color: #0e0e1e; color: #8888aa; }
            QLabel { color: #c0c0e0; }
            QMenuBar { background-color: #0e0e1e; color: #c0c0e0; }
            QMenuBar::item { padding: 4px 12px; background: transparent; }
            QMenuBar::item:selected { background: #2a2a4a; border-radius: 4px; }
            QMenu { background-color: #1e1e3e; color: #e0e0f0;
                    border: 1px solid #3a3a6a; }
            QMenu::item:selected { background-color: #2a2a4a; }
            QTableWidget { background-color: #1a1a2e; color: #e0e0f0;
                           gridline-color: #2e2e4e; }
            QHeaderView::section { background-color: #2a2a4a; color: #e0e0f0;
                                   border: none; padding: 4px; }
        """)
