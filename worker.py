from __future__ import annotations

import traceback
from pathlib import Path
from threading import Event
from typing import List

import numpy as np
from PySide6 import QtCore, QtGui

from models import StitchParams
from stitcher import StitchCancelled, run_stitching


class StitchWorker(QtCore.QThread):
    progress = QtCore.Signal(int)
    log = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    preview_ready = QtCore.Signal(QtGui.QImage)

    def __init__(self, paths: List[str], output_path: str, params: StitchParams, parent=None) -> None:
        super().__init__(parent)
        self._paths = paths
        self._output_path = output_path
        self._params = params
        self.cancel_flag = Event()

    def run(self) -> None:
        try:
            run_stitching(
                paths=self._paths,
                output_path=self._output_path,
                params=self._params,
                on_progress=self.progress.emit,
                on_log=self.log.emit,
                cancel_flag=self.cancel_flag,
                on_preview=self._emit_preview,
            )
            self.finished.emit(self._output_path)
        except StitchCancelled:
            self.log.emit("Stitching cancelled")
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def stop(self) -> None:
        self.cancel_flag.set()

    def _emit_preview(self, preview: np.ndarray) -> None:
        h, w = preview.shape[:2]
        image = QtGui.QImage(preview.data, w, h, QtGui.QImage.Format_RGB888)
        self.preview_ready.emit(image.copy())
