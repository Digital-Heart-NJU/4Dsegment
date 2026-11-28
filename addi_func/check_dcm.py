
import os
import sys
import numpy as np
import pydicom

from pathlib import Path

try:
    from pydicom.pixels import apply_modality_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit,
    QFileDialog, QSlider, QDoubleSpinBox,
    QMessageBox
)

from PySide6.QtCore import Qt, QLibraryInfo, QTimer

from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# ==========================================================
# 1. 基本配置
# ==========================================================

DEFAULT_ROOT = r"/media/mengao/T7/Dataset_XI-AN_XINSIJUN/hcm"


# ==========================================================
# 2. 文件夹扫描
# ==========================================================

def discover_patients(root):

    root = Path(root)

    if not root.exists():
        return []

    return sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower()
    )


def contains_files(folder):

    for _, _, files in os.walk(folder):

        if files:
            return True

    return False


def discover_series(patient_folder):

    patient_folder = Path(patient_folder)

    result = []

    # 第一层：采录时间文件夹
    for date_folder in sorted(patient_folder.iterdir()):

        if not date_folder.is_dir():
            continue

        # 第二层：序列文件夹，例如101、102、103
        for series_folder in sorted(date_folder.iterdir()):

            if not series_folder.is_dir():
                continue

            if contains_files(series_folder):

                result.append({
                    "name": series_folder.name,
                    "date": date_folder.name,
                    "path": str(series_folder)
                })

    return result


def iter_files(folder):

    for root, _, files in os.walk(folder):

        for filename in files:

            yield os.path.join(root, filename)


# ==========================================================
# 3. DICOM 元数据辅助函数
# ==========================================================

def first_item(dataset, sequence_name):

    if dataset is None:
        return None

    sequence = getattr(dataset, sequence_name, None)

    if sequence is not None and len(sequence) > 0:

        return sequence[0]

    return None


def get_orientation(ds, frame_group):

    shared = first_item(
        ds, "SharedFunctionalGroupsSequence"
    )

    for owner in (frame_group, shared):

        item = first_item(
            owner, "PlaneOrientationSequence"
        )

        if item is not None:

            orientation = getattr(
                item, "ImageOrientationPatient", None
            )

            if orientation is not None:
                return np.asarray(orientation, dtype=float)

    orientation = getattr(
        ds, "ImageOrientationPatient", None
    )

    if orientation is not None:

        return np.asarray(orientation, dtype=float)

    return None


def get_temporal_key(ds, frame_group):

    frame_content = first_item(
        frame_group, "FrameContentSequence"
    )

    # Enhanced DICOM 时间维度
    for owner in (frame_content, frame_group, ds):

        if owner is None:
            continue

        value = getattr(
            owner, "TemporalPositionIndex", None
        )

        if value is not None:
            return ("TP", float(value))

    # 传统 DICOM 时间维度
    value = getattr(
        ds, "TemporalPositionIdentifier", None
    )

    if value is not None:
        return ("TP", float(value))

    # TriggerTime
    value = getattr(ds, "TriggerTime", None)

    if value is not None:
        return ("TT", round(float(value), 3))

    return None


def get_slice_position(ds, frame_group, frame_index,
                       frame_count):

    frame_content = first_item(
        frame_group, "FrameContentSequence"
    )

    # 首先尝试 Enhanced DICOM 的逐帧空间位置
    position_item = first_item(
        frame_group, "PlanePositionSequence"
    )

    if position_item is None:

        position_item = first_item(
            frame_group, "PlanePositionVolumeSequence"
        )

    position = None

    if position_item is not None:

        position = getattr(
            position_item, "ImagePositionPatient", None
        )

    orientation = get_orientation(ds, frame_group)

    if position is not None and orientation is not None:

        row = orientation[:3]
        col = orientation[3:6]

        normal = np.cross(row, col)

        normal = normal / np.linalg.norm(normal)

        return float(
            np.dot(np.asarray(position, dtype=float), normal)
        )

    # Enhanced DICOM 层面编号
    if frame_content is not None:

        value = getattr(
            frame_content, "InStackPositionNumber", None
        )

        if value is not None:
            return float(value)

    # 传统单帧 DICOM
    if frame_count == 1:

        position = getattr(
            ds, "ImagePositionPatient", None
        )

        if position is not None and orientation is not None:

            normal = np.cross(
                orientation[:3],
                orientation[3:6]
            )

            normal = normal / np.linalg.norm(normal)

            return float(
                np.dot(np.asarray(position, dtype=float), normal)
            )

        instance = getattr(ds, "InstanceNumber", None)

        if instance is not None:
            return float(instance)

    # 没有空间位置信息时，使用文件内部帧顺序
    return float(frame_index)


# ==========================================================
# 4. DICOM 图像读取与 3D/4D 数据组织
# ==========================================================

def split_frames(arr, samples_per_pixel):

    arr = np.asarray(arr)

    def rgb_to_gray(image):

        image = image[..., :3].astype(np.float32)

        return (
            0.299 * image[..., 0] +
            0.587 * image[..., 1] +
            0.114 * image[..., 2]
        )

    if samples_per_pixel > 1:

        if arr.ndim == 3:

            return [rgb_to_gray(arr)]

        if arr.ndim == 4:

            return [rgb_to_gray(frame) for frame in arr]

        raise ValueError("不支持的彩色 DICOM 数据维度")

    if arr.ndim == 2:

        return [arr]

    if arr.ndim == 3:

        return [arr[i] for i in range(arr.shape[0])]

    raise ValueError(
        "无法识别的 DICOM 像素数据维度：" + str(arr.shape)
    )


def get_float_value(value):

    if value is None:
        return None

    try:

        if hasattr(value, "__iter__") and not isinstance(
            value, (str, bytes)
        ):

            value = value[0]

        return float(value)

    except Exception:

        return None


def load_dicom_series(folder):

    warnings = []

    # ------------------------------------------------------
    # 先扫描 DICOM 文件头，按 SeriesInstanceUID 分组
    # 避免将不同 DICOM Series 混合
    # ------------------------------------------------------

    header_groups = {}

    for path in iter_files(folder):

        try:

            header = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True
            )

            if not hasattr(header, "Rows"):
                continue

            if not hasattr(header, "Columns"):
                continue

            if (
                not hasattr(header, "SOPClassUID")
                and not hasattr(header, "SeriesInstanceUID")
            ):
                continue

            uid = str(
                getattr(header, "SeriesInstanceUID", "UNKNOWN")
            )

            header_groups.setdefault(uid, []).append(path)

        except Exception:

            continue

    if not header_groups:

        raise RuntimeError(
            "当前序列文件夹中没有找到有效的 DICOM 影像。"
        )

    # 如果一个文件夹存在多个 SeriesInstanceUID，
    # 默认选择影像文件数量最多的 Series。
    selected_uid, selected_paths = max(
        header_groups.items(),
        key=lambda item: len(item[1])
    )

    if len(header_groups) > 1:

        warnings.append(
            "该文件夹中检测到多个 SeriesInstanceUID，"
            "当前自动选择影像文件数量最多的一组。"
        )

    records = []

    first_ds = None

    failed_count = 0

    # ------------------------------------------------------
    # 正式读取像素数据
    # ------------------------------------------------------

    for path in selected_paths:

        try:

            ds = pydicom.dcmread(
                path,
                force=True
            )

            if (
                "PixelData" not in ds
                and "FloatPixelData" not in ds
            ):
                continue

            raw = ds.pixel_array

            samples = int(
                getattr(ds, "SamplesPerPixel", 1)
            )

            # 灰度影像进行 Modality LUT / Rescale
            if samples == 1:

                raw = apply_modality_lut(raw, ds)

            frames = split_frames(raw, samples)

            if first_ds is None:
                first_ds = ds

            per_frame = ds.get(
                "PerFrameFunctionalGroupsSequence", None
            )

            frame_count = len(frames)

            for i, frame in enumerate(frames):

                frame_group = None

                if (
                    per_frame is not None
                    and i < len(per_frame)
                ):

                    frame_group = per_frame[i]

                temporal_key = get_temporal_key(
                    ds, frame_group
                )

                z_position = get_slice_position(
                    ds,
                    frame_group,
                    i,
                    frame_count
                )

                records.append({

                    "image": np.asarray(
                        frame, dtype=np.float32
                    ),

                    "z": z_position,

                    "time": temporal_key

                })

        except Exception:

            failed_count += 1

            continue

    if not records:

        raise RuntimeError(
            "DICOM 文件读取失败，可能存在不支持的压缩格式，"
            "或者文件缺少有效像素数据。"
        )

    if failed_count > 0:

        warnings.append(
            "有 %d 个文件读取失败，已跳过。" % failed_count
        )

    # ------------------------------------------------------
    # 检查图像尺寸
    # ------------------------------------------------------

    shapes = set(
        tuple(r["image"].shape) for r in records
    )

    if len(shapes) > 1:

        raise RuntimeError(
            "当前序列中存在不同尺寸的影像，"
            "可能混入了其他序列，无法直接构建体数据。"
        )

    # ------------------------------------------------------
    # 按时间维度分组
    # ------------------------------------------------------

    time_groups = {}

    for record in records:

        key = record["time"]

        time_groups.setdefault(key, []).append(record)

    def time_sort_key(key):

        if key is None:
            return ("", 0)

        return (str(key[0]), float(key[1]))

    time_keys = sorted(
        time_groups.keys(),
        key=time_sort_key
    )

    volumes = []

    # ------------------------------------------------------
    # 3D 数据：T = 1
    # 4D 数据：根据明确的时间标签组织 T 维度
    # ------------------------------------------------------

    if len(time_keys) == 1:

        ordered = sorted(
            time_groups[time_keys[0]],
            key=lambda r: r["z"]
        )

        volume = np.stack(
            [r["image"] for r in ordered],
            axis=0
        )

        volume = volume[np.newaxis, ...]

    else:

        # 对不同时间帧进行空间位置匹配
        position_maps = []

        for key in time_keys:

            group = sorted(
                time_groups[key],
                key=lambda r: r["z"]
            )

            position_map = {}

            for record in group:

                z_key = round(float(record["z"]), 3)

                if z_key not in position_map:

                    position_map[z_key] = record

            position_maps.append(position_map)

        common_positions = set(position_maps[0].keys())

        for position_map in position_maps[1:]:

            common_positions &= set(position_map.keys())

        if common_positions:

            z_keys = sorted(common_positions)

            for position_map in position_maps:

                frame_volume = np.stack(
                    [
                        position_map[z]["image"]
                        for z in z_keys
                    ],
                    axis=0
                )

                volumes.append(frame_volume)

            if any(
                len(position_map) > len(z_keys)
                for position_map in position_maps
            ):

                warnings.append(
                    "不同时间帧的空间层面不完全一致，"
                    "显示时采用各时间帧共同的空间位置。"
                )

        else:

            # 空间位置无法对应时，按各时间组内部的
            # 排序结果进行索引匹配
            sorted_groups = []

            for key in time_keys:

                sorted_groups.append(
                    sorted(
                        time_groups[key],
                        key=lambda r: r["z"]
                    )
                )

            n_slices = min(
                len(group) for group in sorted_groups
            )

            if n_slices == 0:
                raise RuntimeError("无法匹配时间帧与空间层面。")

            for group in sorted_groups:

                frame_volume = np.stack(
                    [
                        r["image"]
                        for r in group[:n_slices]
                    ],
                    axis=0
                )

                volumes.append(frame_volume)

            warnings.append(
                "不同时间帧的空间坐标不完全匹配，"
                "已按层面排序进行索引匹配，请检查影像。"
            )

        volume = np.stack(volumes, axis=0)

    # 最终维度统一为 T, Z, Y, X
    volume = volume.astype(np.float32, copy=False)

    # ------------------------------------------------------
    # 窗宽窗位
    # ------------------------------------------------------

    wl = get_float_value(
        first_ds.get("WindowCenter", None)
    )

    ww = get_float_value(
        first_ds.get("WindowWidth", None)
    )

    if ww is None or ww <= 0 or wl is None:

        low, high = np.percentile(
            volume[0, 0],
            [1, 99]
        )

        if ww is None or ww <= 0:
            ww = max(float(high - low), 1.0)

        if wl is None:
            wl = float((high + low) / 2)

    photometric = str(
        first_ds.get(
            "PhotometricInterpretation", "MONOCHROME2"
        )
    )

    metadata = {

        "modality": str(
            first_ds.get("Modality", "Unknown")
        ),

        "description": str(
            first_ds.get("SeriesDescription", "")
        ),

        "series_uid": selected_uid,

        "photometric": photometric

    }

    return volume, ww, wl, metadata, warnings


# ==========================================================
# 5. GUI 主程序
# ==========================================================

class DicomViewer(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "DICOM Multi-Patient Viewer"
        )

        self.resize(1250, 900)

        self.volume = None

        self.current_slice = 0

        self.current_time = 0

        self.invert = False

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.advance_image)

        self.init_ui()

        if os.path.isdir(DEFAULT_ROOT):

            self.refresh_patients()

        else:

            self.status_label.setText(
                "请先选择 DICOM 数据根目录。"
            )

    # ------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------

    def init_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # 数据目录
        root_layout = QHBoxLayout()

        root_layout.addWidget(
            QLabel("DICOM 根目录：")
        )

        self.root_edit = QLineEdit(
            DEFAULT_ROOT
        )

        root_layout.addWidget(
            self.root_edit
        )

        browse_btn = QPushButton("选择目录")

        browse_btn.clicked.connect(
            self.browse_root
        )

        root_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("刷新")

        refresh_btn.clicked.connect(
            self.refresh_patients
        )

        root_layout.addWidget(refresh_btn)

        main_layout.addLayout(root_layout)

        # 患者与序列选择
        select_layout = QHBoxLayout()

        select_layout.addWidget(
            QLabel("患者：")
        )

        self.patient_combo = QComboBox()

        self.patient_combo.setMinimumWidth(300)

        self.patient_combo.currentIndexChanged.connect(
            self.on_patient_changed
        )

        select_layout.addWidget(
            self.patient_combo
        )

        select_layout.addWidget(
            QLabel("序列文件夹：")
        )

        self.series_combo = QComboBox()

        self.series_combo.setMinimumWidth(300)

        self.series_combo.currentIndexChanged.connect(
            self.on_series_changed
        )

        select_layout.addWidget(
            self.series_combo
        )

        main_layout.addLayout(select_layout)

        # 状态信息
        self.status_label = QLabel(
            "等待选择患者和序列..."
        )

        main_layout.addWidget(
            self.status_label
        )

        # Matplotlib 显示区域
        self.figure = Figure(
            figsize=(8, 7)
        )

        self.canvas = FigureCanvasQTAgg(
            self.figure
        )

        self.ax = self.figure.add_subplot(111)

        self.ax.axis("off")

        main_layout.addWidget(
            self.canvas,
            stretch=1
        )

        # 切片控制
        slice_layout = QHBoxLayout()

        self.slice_label = QLabel(
            "切片：0 / 0"
        )

        slice_layout.addWidget(
            self.slice_label
        )

        self.slice_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.slice_slider.valueChanged.connect(
            self.on_slice_changed
        )

        slice_layout.addWidget(
            self.slice_slider,
            stretch=1
        )

        main_layout.addLayout(slice_layout)

        # 时间帧控制
        time_layout = QHBoxLayout()

        self.time_label = QLabel(
            "时间帧：0 / 0"
        )

        time_layout.addWidget(
            self.time_label
        )

        self.time_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.time_slider.valueChanged.connect(
            self.on_time_changed
        )

        time_layout.addWidget(
            self.time_slider,
            stretch=1
        )

        main_layout.addLayout(time_layout)

        # 自动播放：4D 影像默认播放时间帧，3D 影像默认播放切片。
        playback_layout = QHBoxLayout()

        self.auto_display_btn = QPushButton("自动播放 / Auto Display")
        self.auto_display_btn.setCheckable(True)
        self.auto_display_btn.setEnabled(False)
        self.auto_display_btn.toggled.connect(self.toggle_auto_display)
        playback_layout.addWidget(self.auto_display_btn)

        playback_layout.addWidget(QLabel("播放内容："))
        self.playback_axis_combo = QComboBox()
        self.playback_axis_combo.setEnabled(False)
        self.playback_axis_combo.currentIndexChanged.connect(
            lambda _: self.stop_auto_display()
        )
        playback_layout.addWidget(self.playback_axis_combo)

        playback_layout.addWidget(QLabel("播放速度："))
        self.playback_speed_spin = QDoubleSpinBox()
        self.playback_speed_spin.setRange(0.2, 30.0)
        self.playback_speed_spin.setDecimals(1)
        self.playback_speed_spin.setValue(5.0)
        self.playback_speed_spin.setSuffix(" fps")
        self.playback_speed_spin.valueChanged.connect(
            self.update_playback_speed
        )
        self.update_playback_speed(self.playback_speed_spin.value())
        playback_layout.addWidget(self.playback_speed_spin)
        playback_layout.addStretch()
        main_layout.addLayout(playback_layout)

        # 窗宽窗位
        window_layout = QHBoxLayout()

        window_layout.addWidget(
            QLabel("窗宽 WW：")
        )

        self.ww_spin = QDoubleSpinBox()

        self.ww_spin.setRange(
            0.1, 10000000
        )

        self.ww_spin.setDecimals(1)

        self.ww_spin.valueChanged.connect(
            lambda _: self.update_image()
        )

        window_layout.addWidget(
            self.ww_spin
        )

        window_layout.addWidget(
            QLabel("窗位 WL：")
        )

        self.wl_spin = QDoubleSpinBox()

        self.wl_spin.setRange(
            -10000000, 10000000
        )

        self.wl_spin.setDecimals(1)

        self.wl_spin.valueChanged.connect(
            lambda _: self.update_image()
        )

        window_layout.addWidget(
            self.wl_spin
        )

        reset_btn = QPushButton(
            "恢复默认窗宽窗位"
        )

        reset_btn.clicked.connect(
            self.reset_window
        )

        window_layout.addWidget(
            reset_btn
        )

        main_layout.addLayout(window_layout)

    # ------------------------------------------------------
    # 选择根目录
    # ------------------------------------------------------

    def browse_root(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "选择 DICOM 数据根目录",
            self.root_edit.text()
        )

        if folder:

            self.root_edit.setText(folder)

            self.refresh_patients()

    # ------------------------------------------------------
    # 扫描患者
    # ------------------------------------------------------

    def refresh_patients(self):

        self.stop_auto_display()

        root = self.root_edit.text().strip()

        patients = discover_patients(root)

        self.patient_combo.blockSignals(True)

        self.patient_combo.clear()

        for patient in patients:

            self.patient_combo.addItem(
                patient.name,
                str(patient)
            )

        self.patient_combo.blockSignals(False)

        if patients:

            self.status_label.setText(
                "发现 %d 位患者。" % len(patients)
            )

            self.on_patient_changed(0)

        else:

            self.series_combo.clear()

            self.clear_view()

            self.status_label.setText(
                "未发现患者文件夹，请检查根目录。"
            )

    # ------------------------------------------------------
    # 选择患者
    # ------------------------------------------------------

    def on_patient_changed(self, index):

        self.stop_auto_display()

        self.series_combo.blockSignals(True)

        self.series_combo.clear()

        if index < 0:
            self.series_combo.blockSignals(False)
            return

        patient_path = self.patient_combo.itemData(
            index
        )

        entries = discover_series(
            patient_path
        )

        for entry in entries:

            label = (
                entry["name"]
                + "（采录时间："
                + entry["date"]
                + "）"
            )

            self.series_combo.addItem(
                label,
                entry["path"]
            )

        self.series_combo.blockSignals(False)

        if entries:

            self.on_series_changed(0)

        else:

            self.clear_view()

            self.status_label.setText(
                "该患者没有发现序列文件夹。"
            )

    # ------------------------------------------------------
    # 选择序列并加载
    # ------------------------------------------------------

    def on_series_changed(self, index):

        self.stop_auto_display()
        self.auto_display_btn.setEnabled(False)
        self.playback_axis_combo.setEnabled(False)

        if index < 0:
            return

        folder = self.series_combo.itemData(
            index
        )

        if not folder:
            return

        QApplication.setOverrideCursor(
            Qt.CursorShape.WaitCursor
        )

        try:

            self.status_label.setText(
                "正在读取 DICOM，请稍候..."
            )

            QApplication.processEvents()

            (
                volume,
                ww,
                wl,
                metadata,
                warnings
            ) = load_dicom_series(folder)

            self.volume = volume

            self.current_slice = 0

            self.current_time = 0

            self.invert = (
                metadata["photometric"] == "MONOCHROME1"
            )

            t_count, z_count, height, width = (
                volume.shape
            )

            self.playback_axis_combo.clear()
            if t_count > 1:
                self.playback_axis_combo.addItem("时间帧", "time")
            if z_count > 1:
                self.playback_axis_combo.addItem("切片", "slice")
            self.playback_axis_combo.setEnabled(
                self.playback_axis_combo.count() > 1
            )
            self.auto_display_btn.setEnabled(
                self.playback_axis_combo.count() > 0
            )

            # 设置切片滑块
            self.slice_slider.blockSignals(True)

            self.slice_slider.setRange(
                0, z_count - 1
            )

            self.slice_slider.setValue(0)

            self.slice_slider.setEnabled(
                z_count > 1
            )

            self.slice_slider.blockSignals(False)

            # 设置时间滑块
            self.time_slider.blockSignals(True)

            self.time_slider.setRange(
                0, t_count - 1
            )

            self.time_slider.setValue(0)

            self.time_slider.setEnabled(
                t_count > 1
            )

            self.time_slider.setVisible(
                t_count > 1
            )

            self.time_label.setVisible(
                t_count > 1
            )

            self.time_slider.blockSignals(False)

            # 初始化窗宽窗位
            self.ww_spin.blockSignals(True)

            self.wl_spin.blockSignals(True)

            self.ww_spin.setValue(ww)

            self.wl_spin.setValue(wl)

            self.ww_spin.blockSignals(False)

            self.wl_spin.blockSignals(False)

            self.update_image()

            self.status_label.setText(
                "患者：%s | 序列：%s | "
                "维度：T=%d, Z=%d, Y=%d, X=%d | "
                "Modality：%s | %s"
                % (
                    self.patient_combo.currentText(),
                    self.series_combo.currentText(),
                    t_count,
                    z_count,
                    height,
                    width,
                    metadata["modality"],
                    metadata["description"]
                )
            )

            if warnings:

                QMessageBox.warning(
                    self,
                    "DICOM 数据提示",
                    "\n".join(warnings)
                )

        except Exception as e:

            self.volume = None

            self.clear_view()

            QMessageBox.critical(
                self,
                "DICOM 读取错误",
                str(e)
            )

        finally:

            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------
    # 自动播放控制
    # ------------------------------------------------------

    def toggle_auto_display(self, playing):

        if playing:
            if self.volume is None or self.playback_axis_combo.count() == 0:
                self.auto_display_btn.setChecked(False)
                return
            self.auto_display_btn.setText("暂停 / Pause")
            self.playback_timer.start()
        else:
            self.playback_timer.stop()
            self.auto_display_btn.setText("自动播放 / Auto Display")

    def stop_auto_display(self):

        self.playback_timer.stop()
        self.auto_display_btn.setChecked(False)

    def update_playback_speed(self, fps):

        self.playback_timer.setInterval(max(1, round(1000 / fps)))

    def advance_image(self):

        if self.volume is None:
            self.stop_auto_display()
            return

        if self.playback_axis_combo.currentData() == "time":
            self.time_slider.setValue(
                (self.current_time + 1) % self.volume.shape[0]
            )
        else:
            self.slice_slider.setValue(
                (self.current_slice + 1) % self.volume.shape[1]
            )

    def closeEvent(self, event):

        self.stop_auto_display()
        super().closeEvent(event)

    # ------------------------------------------------------
    # 切片变化
    # ------------------------------------------------------

    def on_slice_changed(self, value):

        self.current_slice = value

        self.update_image()

    # ------------------------------------------------------
    # 时间帧变化
    # ------------------------------------------------------

    def on_time_changed(self, value):

        self.current_time = value

        self.update_image()

    # ------------------------------------------------------
    # 窗宽窗位恢复
    # ------------------------------------------------------

    def reset_window(self):

        if self.volume is None:
            return

        image = self.volume[0, 0]

        low, high = np.percentile(
            image, [1, 99]
        )

        self.ww_spin.setValue(
            max(float(high - low), 1.0)
        )

        self.wl_spin.setValue(
            float((high + low) / 2)
        )

    # ------------------------------------------------------
    # 显示图像
    # ------------------------------------------------------

    def update_image(self):

        if self.volume is None:
            return

        t_count, z_count, _, _ = (
            self.volume.shape
        )

        image = self.volume[
            self.current_time,
            self.current_slice
        ]

        ww = max(
            self.ww_spin.value(),
            0.1
        )

        wl = self.wl_spin.value()

        self.ax.clear()

        cmap = (
            "gray_r" if self.invert else "gray"
        )

        self.ax.imshow(
            image,
            cmap=cmap,
            vmin=wl - ww / 2,
            vmax=wl + ww / 2,
            interpolation="nearest"
        )

        self.ax.axis("off")

        self.ax.set_title(
            "Time Frame: %d / %d     "
            "Slice: %d / %d"
            % (
                self.current_time + 1,
                t_count,
                self.current_slice + 1,
                z_count
            ),
            fontsize=12
        )

        self.figure.tight_layout()

        self.canvas.draw_idle()

        self.slice_label.setText(
            "切片：%d / %d"
            % (
                self.current_slice + 1,
                z_count
            )
        )

        self.time_label.setText(
            "时间帧：%d / %d"
            % (
                self.current_time + 1,
                t_count
            )
        )

    # ------------------------------------------------------
    # 清空图像
    # ------------------------------------------------------

    def clear_view(self):

        self.stop_auto_display()
        self.auto_display_btn.setEnabled(False)
        self.playback_axis_combo.clear()
        self.playback_axis_combo.setEnabled(False)

        self.volume = None

        self.ax.clear()

        self.ax.axis("off")

        self.canvas.draw_idle()

        self.slice_label.setText(
            "切片：0 / 0"
        )

        self.time_label.setText(
            "时间帧：0 / 0"
        )

        self.slice_slider.setRange(0, 0)

        self.time_slider.setRange(0, 0)


# ==========================================================
# 6. 程序入口
# ==========================================================

if __name__ == "__main__":

    # Conda's plugin search can select a different Qt version from PySide6.
    # Use the platform plugins shipped with the binding imported above.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
        Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        / "platforms"
    )

    app = QApplication(sys.argv)

    viewer = DicomViewer()

    viewer.show()

    sys.exit(app.exec())
