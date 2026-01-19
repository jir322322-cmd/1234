from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

from models import StitchParams
from worker import StitchWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tile Stitcher")
        self.resize(1100, 700)

        self._paths: List[str] = []
        self._worker: StitchWorker | None = None

        self._build_ui()
        self._update_start_state()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        input_group = QtWidgets.QGroupBox("Входные файлы")
        input_layout = QtWidgets.QVBoxLayout(input_group)
        input_buttons = QtWidgets.QHBoxLayout()
        self.btn_select_files = QtWidgets.QPushButton("Выбрать файлы...")
        self.btn_select_folder = QtWidgets.QPushButton("Выбрать папку...")
        self.btn_clear = QtWidgets.QPushButton("Очистить список")
        input_buttons.addWidget(self.btn_select_files)
        input_buttons.addWidget(self.btn_select_folder)
        input_buttons.addWidget(self.btn_clear)
        input_layout.addLayout(input_buttons)
        self.lbl_count = QtWidgets.QLabel("Выбрано файлов: 0")
        input_layout.addWidget(self.lbl_count)

        output_group = QtWidgets.QGroupBox("Путь сохранения")
        output_layout = QtWidgets.QHBoxLayout(output_group)
        self.output_edit = QtWidgets.QLineEdit()
        self.btn_output = QtWidgets.QPushButton("Куда сохранить...")
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.btn_output)

        params_group = QtWidgets.QGroupBox("Параметры")
        params_layout = QtWidgets.QGridLayout(params_group)
        self.spin_bg = QtWidgets.QSpinBox()
        self.spin_bg.setRange(200, 255)
        self.spin_bg.setValue(245)
        self.spin_angle = QtWidgets.QDoubleSpinBox()
        self.spin_angle.setRange(0.0, 20.0)
        self.spin_angle.setValue(7.0)
        self.spin_angle.setSingleStep(0.1)
        self.spin_overlap = QtWidgets.QSpinBox()
        self.spin_overlap.setRange(0, 30)
        self.spin_overlap.setValue(15)
        self.combo_compress = QtWidgets.QComboBox()
        self.combo_compress.addItems(["deflate", "lzw", "none"])
        self.chk_debug = QtWidgets.QCheckBox("Debug режим")

        params_layout.addWidget(QtWidgets.QLabel("bg-threshold"), 0, 0)
        params_layout.addWidget(self.spin_bg, 0, 1)
        params_layout.addWidget(QtWidgets.QLabel("max-angle"), 1, 0)
        params_layout.addWidget(self.spin_angle, 1, 1)
        params_layout.addWidget(QtWidgets.QLabel("overlap-max"), 2, 0)
        params_layout.addWidget(self.spin_overlap, 2, 1)
        params_layout.addWidget(QtWidgets.QLabel("compression"), 3, 0)
        params_layout.addWidget(self.combo_compress, 3, 1)
        params_layout.addWidget(self.chk_debug, 4, 0, 1, 2)

        controls_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Старт")
        self.btn_stop = QtWidgets.QPushButton("Остановить")
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)

        left_layout.addWidget(input_group)
        left_layout.addWidget(output_group)
        left_layout.addWidget(params_group)
        left_layout.addLayout(controls_layout)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(QtWidgets.QLabel("Логи"))
        left_layout.addWidget(self.log_text)

        preview_group = QtWidgets.QGroupBox("Превью раскладки")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_label = QtWidgets.QLabel("Нет превью")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.btn_save_preview = QtWidgets.QPushButton("Сохранить превью")
        self.btn_save_preview.setEnabled(False)
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.btn_save_preview)

        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(preview_group, 1)

        self.btn_select_files.clicked.connect(self._select_files)
        self.btn_select_folder.clicked.connect(self._select_folder)
        self.btn_clear.clicked.connect(self._clear_files)
        self.btn_output.clicked.connect(self._select_output)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_save_preview.clicked.connect(self._save_preview)

    def _select_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Выбрать JPG тайлы",
            str(Path.cwd()),
            "JPG Files (*.jpg)"
        )
        if files:
            self._paths = files
            self._update_file_count()
            self._update_start_state()

    def _select_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбрать папку", str(Path.cwd()))
        if folder:
            paths = sorted(Path(folder).glob("*.jpg"))
            self._paths = [str(path) for path in paths]
            self._update_file_count()
            self._update_start_state()

    def _clear_files(self) -> None:
        self._paths = []
        self._update_file_count()
        self._update_start_state()

    def _select_output(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить TIFF",
            str(Path.cwd() / "output.tif"),
            "TIFF Files (*.tif *.tiff)"
        )
        if file_path:
            self.output_edit.setText(file_path)
            self._update_start_state()

    def _update_file_count(self) -> None:
        self.lbl_count.setText(f"Выбрано файлов: {len(self._paths)}")

    def _update_start_state(self) -> None:
        has_input = len(self._paths) > 0
        has_output = bool(self.output_edit.text().strip())
        self.btn_start.setEnabled(has_input and has_output and self._worker is None)

    def _collect_params(self) -> StitchParams:
        params = StitchParams(
            bg_threshold=self.spin_bg.value(),
            max_angle=self.spin_angle.value(),
            overlap_max=self.spin_overlap.value(),
            compression=self.combo_compress.currentText(),
            debug=self.chk_debug.isChecked(),
        )
        output_path = Path(self.output_edit.text()).resolve()
        params.debug_dir = output_path.parent / "debug"
        return params

    def _start(self) -> None:
        if not self._paths:
            return
        output_path = self.output_edit.text().strip()
        if not output_path:
            return

        self.progress.setValue(0)
        self.log_text.clear()
        self._worker = StitchWorker(self._paths, output_path, self._collect_params())
        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(self._append_log)
        self._worker.error.connect(self._show_error)
        self._worker.finished.connect(self._finished)
        self._worker.preview_ready.connect(self._show_preview)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save_preview.setEnabled(False)

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self.btn_stop.setEnabled(False)
            self._append_log("Stopping...")

    def _append_log(self, message: str) -> None:
        self.log_text.append(message)

    def _show_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Ошибка", message)
        self._worker = None
        self.btn_stop.setEnabled(False)
        self._update_start_state()

    def _finished(self, output_path: str) -> None:
        self._append_log(f"Готово: {output_path}")
        self._worker = None
        self.btn_stop.setEnabled(False)
        self._update_start_state()

    def _show_preview(self, image: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        ))
        self.preview_label.setText("")
        self.btn_save_preview.setEnabled(True)
        self._preview_image = image

    def _save_preview(self) -> None:
        if not hasattr(self, "_preview_image"):
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить превью",
            str(Path.cwd() / "preview.png"),
            "PNG Files (*.png)"
        )
        if file_path:
            self._preview_image.save(file_path)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
