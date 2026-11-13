# -*- coding: utf-8 -*-
"""녹화 기록 보기 대화상자 — 상단 메뉴바 '녹화 기록' 에서 연다.

DB 대신 recordings/recordings.json 을 읽어 목록을 표시하고,
선택한 이벤트 영상을 OS 기본 플레이어로 재생하거나 삭제한다.
"""

import os
import sys
import subprocess

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QMessageBox, QLabel
)
from PyQt5.QtCore import Qt

from core import recorder


def _open_with_os(path: str):
    """OS 기본 연결 프로그램으로 파일/폴더 열기."""
    if sys.platform == "win32":
        os.startfile(path)               # noqa: S606 (Windows 전용)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class RecordingsDialog(QDialog):
    _COLS = ["발생 시각", "행동", "신뢰도", "소스", "길이(초)"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("녹화 기록")
        self.resize(720, 440)
        self._records = []
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.lbl_info = QLabel("이상행동 발생 시 자동 녹화된 영상 목록입니다.")
        self.lbl_info.setStyleSheet("color: #aaaacc;")
        layout.addWidget(self.lbl_info)

        self.table = QTableWidget(0, len(self._COLS))
        self.table.setHorizontalHeaderLabels(self._COLS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.cellDoubleClicked.connect(lambda *_: self._play())
        layout.addWidget(self.table)

        btns = QHBoxLayout()
        self.btn_play = QPushButton("▶ 재생")
        self.btn_play.clicked.connect(self._play)
        self.btn_folder = QPushButton("폴더 열기")
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_delete = QPushButton("삭제")
        self.btn_delete.clicked.connect(self._delete)
        self.btn_refresh = QPushButton("새로고침")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_close = QPushButton("닫기")
        self.btn_close.clicked.connect(self.accept)
        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_folder)
        btns.addWidget(self.btn_delete)
        btns.addStretch()
        btns.addWidget(self.btn_refresh)
        btns.addWidget(self.btn_close)
        layout.addLayout(btns)

    def refresh(self):
        self._records = recorder.load_records()
        self.table.setRowCount(len(self._records))
        for row, rec in enumerate(self._records):
            conf = rec.get("confidence", 0)
            vals = [
                rec.get("timestamp", "—"),
                rec.get("behavior_kr", rec.get("behavior_en", "—")),
                f"{float(conf) * 100:.0f}%" if conf else "—",
                rec.get("source", "—"),
                str(rec.get("duration_sec", "—")),
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
        if not self._records:
            self.lbl_info.setText("녹화된 이상행동 기록이 없습니다.")
        else:
            self.lbl_info.setText(f"총 {len(self._records)}건 — 행을 더블클릭하면 재생됩니다.")

    def _selected(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._records):
            return None
        return self._records[row]

    def _play(self):
        rec = self._selected()
        if rec is None:
            QMessageBox.information(self, "선택 없음", "재생할 기록을 선택하세요.")
            return
        path = rec.get("video_path", "")
        if not (path and os.path.exists(path)):
            QMessageBox.warning(self, "파일 없음", f"영상 파일을 찾을 수 없습니다:\n{path}")
            return
        try:
            _open_with_os(path)
        except Exception as e:
            QMessageBox.critical(self, "재생 실패", str(e))

    def _open_folder(self):
        os.makedirs(recorder.RECORD_DIR, exist_ok=True)
        _open_with_os(recorder.RECORD_DIR)

    def _delete(self):
        rec = self._selected()
        if rec is None:
            QMessageBox.information(self, "선택 없음", "삭제할 기록을 선택하세요.")
            return
        ok = QMessageBox.question(
            self, "삭제 확인",
            f"이 기록과 영상 파일을 삭제할까요?\n\n{rec.get('timestamp')} — "
            f"{rec.get('behavior_kr')}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok == QMessageBox.Yes:
            recorder.delete_record(rec.get("id"))
            self.refresh()
