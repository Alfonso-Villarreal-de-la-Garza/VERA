import sys
import os
import cv2
import yaml
import json
import shutil
import math
import time
import random
import numpy as np
from pathlib import Path

# PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QInputDialog,
    QMessageBox, QLineEdit, QGroupBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QScrollArea, QGridLayout, QSizePolicy,
    QTextEdit, QFileDialog, QSlider, QCheckBox, QSplitter,
)
from PySide6.QtCore import Qt, QThread, Signal, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import (
    QFont, QImage, QPixmap, QPainter, QPen, QColor, QIcon, QBrush,
)

# Deep Learning
import albumentations as A
from ultralytics import YOLO

# xArm Connection (Graceful fallback if not installed)
try:
    from xarm.wrapper import XArmAPI
    XARM_AVAILABLE = True
except ImportError:
    XARM_AVAILABLE = False


# =============================================================================
#  COLOUR PALETTE & STYLE CONSTANTS
# =============================================================================
DARK_BG      = "#1e1e2e"
SIDEBAR_BG   = "#11111b"
CARD_BG      = "#313244"
ACCENT       = "#89b4fa"
ACCENT_HOVER = "#b4befe"
SUCCESS      = "#a6e3a1"
WARNING      = "#f9e2af"
DANGER       = "#f38ba8"
TEXT_PRIMARY  = "#cdd6f4"
TEXT_DIM      = "#9399b2"
BORDER        = "#45475a"

GLOBAL_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}}
QLabel {{
    background-color: transparent;
    border: none;
    font-weight: bold;
    font-size: 13px;
}}
QFrame#sidebar {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
}}
QPushButton#nav {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 14px 24px;
    font-size: 14px;
    font-weight: normal;
    color: {TEXT_DIM};
    border-radius: 10px;
    margin: 4px 12px;
}}
QPushButton#nav:hover {{
    background-color: {CARD_BG};
    color: {TEXT_PRIMARY};
}}
QPushButton#nav:checked {{
    background-color: {ACCENT};
    color: {DARK_BG};
    font-weight: bold;
}}
QPushButton#action {{
    background-color: {ACCENT};
    color: {DARK_BG};
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton#action:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton#action:disabled {{
    background-color: {BORDER};
    color: {TEXT_DIM};
}}
QPushButton#danger {{
    background-color: {DANGER};
    color: {DARK_BG};
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton#step_active {{
    background-color: {WARNING};
    color: {DARK_BG};
    border: none;
    border-radius: 8px;
    padding: 14px 24px;
    font-size: 14px;
    font-weight: 700;
}}
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    margin-top: 24px;
    padding: 24px 16px 16px 16px;
    font-weight: bold;
    color: {ACCENT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 20px;
    padding: 0 10px;
    background-color: {DARK_BG};
    border-radius: 4px;
    font-size: 13px;
}}
QLabel#title {{
    font-size: 24px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    margin-bottom: 4px;
}}
QLabel#subtitle {{
    font-size: 14px;
    font-weight: normal;
    color: {TEXT_DIM};
    margin-bottom: 12px;
}}
QLabel#status {{
    font-size: 13px;
    color: {WARNING};
    padding: 6px 0;
    font-weight: bold;
}}
QLabel#step_instruction {{
    font-size: 16px;
    font-weight: bold;
    color: {WARNING};
    padding: 12px;
    background-color: {CARD_BG};
    border: 2px solid {WARNING};
    border-radius: 10px;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {DARK_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
    font-weight: normal;
    selection-background-color: {ACCENT};
    qproperty-alignment: AlignCenter;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
    background-color: {SIDEBAR_BG};
}}
QProgressBar {{
    background-color: {DARK_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-size: 12px;
    font-weight: bold;
    height: 24px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 6px;
}}
QTextEdit {{
    background-color: {SIDEBAR_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    color: {TEXT_PRIMARY};
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    font-weight: normal;
    padding: 12px;
    line-height: 1.5;
}}
QScrollBar:vertical {{
    border: none;
    background: {SIDEBAR_BG};
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    min-height: 20px;
    border-radius: 5px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollArea {{
    border: none;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {BORDER};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
"""


# =============================================================================
#  BOUNDING BOX DRAWING WIDGET
# =============================================================================
class BoundingBoxWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: #11111b; border: 2px solid {BORDER}; border-radius: 12px;")
        self.setMinimumHeight(300)
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_pixmap: QPixmap | None = None
        self.original_image_cv: np.ndarray | None = None
        self.boxes: list[dict] = []

    def _display_rect(self) -> QRect:
        if not self.pixmap(): return QRect()
        pw, ph = self.pixmap().width(), self.pixmap().height()
        lw, lh = self.width(), self.height()
        return QRect((lw - pw) // 2, (lh - ph) // 2, pw, ph)

    def _widget_to_image(self, pt: QPoint):
        if self.original_image_cv is None or not self.pixmap(): return None
        dr = self._display_rect()
        if dr.width() == 0 or dr.height() == 0: return None
        rx = max(0.0, min(1.0, (pt.x() - dr.x()) / dr.width()))
        ry = max(0.0, min(1.0, (pt.y() - dr.y()) / dr.height()))
        ih, iw = self.original_image_cv.shape[:2]
        return rx * iw, ry * ih

    def set_image(self, cv_img: np.ndarray):
        self.original_image_cv = cv_img.copy()
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.current_pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
        self._refresh_display()
        self.boxes.clear()
        self.start_point = QPoint()
        self.end_point = QPoint()

    def _refresh_display(self):
        if self.current_pixmap:
            self.setPixmap(self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_pixmap:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = self.start_point

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            tl = self._widget_to_image(self.start_point)
            br = self._widget_to_image(self.end_point)
            if tl is None or br is None: return
            x1, y1 = min(tl[0], br[0]), min(tl[1], br[1])
            x2, y2 = max(tl[0], br[0]), max(tl[1], br[1])
            if (x2 - x1) < 4 or (y2 - y1) < 4: return
            ih, iw = self.original_image_cv.shape[:2]
            label, ok = QInputDialog.getText(self, "Label Part", "Enter part name:")
            if ok and label.strip():
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                self.boxes.append({
                    "rect": QRect(self.start_point, self.end_point).normalized(),
                    "yolo": [cx, cy, bw, bh], "label": label.strip().lower(),
                })
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for box in self.boxes:
            painter.setPen(QPen(QColor(SUCCESS), 2))
            painter.drawRect(box["rect"])
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(box["rect"].topLeft() + QPoint(5, 15), box["label"])
        if self.drawing and not self.start_point.isNull():
            painter.setPen(QPen(QColor(DANGER), 2, Qt.DashLine))
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())
        painter.end()

    def clear_boxes(self):
        self.boxes.clear()
        self.update()


# =============================================================================
#  CLICKABLE VIDEO WIDGET
# =============================================================================
class ClickableVideoWidget(QLabel):
    pixel_clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: #11111b; border: 2px solid {BORDER}; border-radius: 12px;")
        self.original_shape = None

    def set_image(self, cv_img: np.ndarray):
        self.original_shape = cv_img.shape[:2]
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, event):
        if not self.pixmap() or not self.original_shape: return
        pw, ph = self.pixmap().width(), self.pixmap().height()
        lw, lh = self.width(), self.height()
        x_off, y_off = (lw - pw) // 2, (lh - ph) // 2
        rx = (event.pos().x() - x_off) / pw
        ry = (event.pos().y() - y_off) / ph
        if 0 <= rx <= 1 and 0 <= ry <= 1:
            ih, iw = self.original_shape
            self.pixel_clicked.emit(rx * iw, ry * ih)


# =============================================================================
#  YOLO TRAINING WORKER
# =============================================================================
class YoloTrainerWorker(QThread):
    status_update = Signal(str)
    progress_update = Signal(int)
    training_finished = Signal(str)

    def __init__(self, dataset, project_path, epochs=30, augment_factor=25, imgsz=640):
        super().__init__()
        self.dataset = dataset
        self.dataset_dir = os.path.join(project_path, "sme_dataset")
        self.project_path = project_path
        self.epochs = epochs
        self.augment_factor = augment_factor
        self.imgsz = imgsz
        self.unique_classes = sorted(set(
            box["label"] for data in self.dataset for box in data["boxes"]
        ))
        self.class_to_id = {name: idx for idx, name in enumerate(self.unique_classes)}

    def run(self):
        try:
            self._prepare_dataset()
            best_path = self._train()
            self.training_finished.emit(best_path)
        except Exception as e:
            self.status_update.emit(f"ERROR: {e}")

    def _prepare_dataset(self):
        self.status_update.emit("Preparing directories …")
        if os.path.exists(self.dataset_dir): shutil.rmtree(self.dataset_dir)
        for split in ("train", "val"):
            os.makedirs(f"{self.dataset_dir}/images/{split}", exist_ok=True)
            os.makedirs(f"{self.dataset_dir}/labels/{split}", exist_ok=True)

        transform = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomScale(scale_limit=0.15, p=0.4),
            A.PadIfNeeded(min_height=self.imgsz, min_width=self.imgsz,
                          border_mode=cv2.BORDER_REFLECT_101, p=0.0),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                     min_visibility=0.3, min_area=256))

        self.status_update.emit("Augmenting images …")
        img_counter = 0
        total_base = len(self.dataset)
        for idx, data in enumerate(self.dataset):
            base_img = data["image"]
            bboxes = [b["yolo"] for b in data["boxes"]]
            labels = [b["label"] for b in data["boxes"]]
            self._save(base_img, bboxes, labels, img_counter, "train")
            img_counter += 1
            for aug_i in range(self.augment_factor):
                try:
                    t = transform(image=base_img, bboxes=bboxes, class_labels=labels)
                    if len(t["bboxes"]) == 0: continue
                    split = "val" if aug_i % 5 == 0 else "train"
                    self._save(t["image"], t["bboxes"], t["class_labels"], img_counter, split)
                    img_counter += 1
                except Exception: pass
            self.progress_update.emit(int(((idx + 1) / total_base) * 30))
        self.status_update.emit(f"Dataset ready – {img_counter} images.")

    def _save(self, img, bboxes, labels, index, split):
        cv2.imwrite(f"{self.dataset_dir}/images/{split}/img_{index:05d}.jpg", img)
        with open(f"{self.dataset_dir}/labels/{split}/img_{index:05d}.txt", "w") as f:
            for bbox, label in zip(bboxes, labels):
                cid = self.class_to_id[label]
                f.write(f"{cid} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    def _train(self) -> str:
        self.status_update.emit(f"Training YOLOv8n – {self.epochs} epochs …")
        yaml_path = os.path.join(self.dataset_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({
                "path": os.path.abspath(self.dataset_dir), "train": "images/train",
                "val": "images/val",
                "names": {idx: name for name, idx in self.class_to_id.items()},
            }, f, default_flow_style=False)
        model = YOLO("yolov8n.pt")
        model.train(data=yaml_path, epochs=self.epochs, imgsz=self.imgsz, batch=-1,
                     device="cpu", patience=10, verbose=False, project=self.project_path,
                     name="sme_model", exist_ok=True, lr0=0.01, lrf=0.01, warmup_epochs=3,
                     augment=True, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, flipud=0.1, fliplr=0.5,
                     mosaic=0.8, mixup=0.1)
        self.progress_update.emit(95)
        best_pt = os.path.join(self.project_path, "sme_model", "weights", "best.pt")
        if not os.path.isfile(best_pt):
            last_pt = os.path.join(self.project_path, "sme_model", "weights", "last.pt")
            if os.path.isfile(last_pt): best_pt = last_pt
        self.status_update.emit("Training complete ✓")
        self.progress_update.emit(100)
        return best_pt


# =============================================================================
#  WAYPOINT RECORD
# =============================================================================
class Waypoint:
    def __init__(self, label="", x=0, y=0, z=200, roll=-179.8, pitch=0.0, yaw=0.0,
                 gripper="open", source="manual", speed=100, move_type="moveL"):
        self.label = label
        self.x = x; self.y = y; self.z = z
        self.roll = roll; self.pitch = pitch; self.yaw = yaw
        self.gripper = gripper; self.source = source
        self.speed = speed; self.move_type = move_type

    def as_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(d):
        return Waypoint(**{k: v for k, v in d.items() if k in Waypoint.__init__.__code__.co_varnames})


# =============================================================================
#  JOG PANEL
# =============================================================================
class JogPanel(QGroupBox):
    jog_requested = Signal(str, float)

    def __init__(self, title="Fine-Tune Position"):
        super().__init__(title)
        layout = QGridLayout(self)
        layout.setSpacing(12)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 50.0); self.step_spin.setValue(5.0); self.step_spin.setSuffix(" mm")
        layout.addWidget(QLabel("Step:"), 0, 0)
        layout.addWidget(self.step_spin, 0, 1, 1, 2)
        for label, row in [("X", 1), ("Y", 2), ("Z", 3)]:
            minus = QPushButton(f"  −{label}  "); minus.setObjectName("action")
            plus = QPushButton(f"  +{label}  "); plus.setObjectName("action")
            minus.clicked.connect(lambda _, a=label: self.jog_requested.emit(a, -self.step_spin.value()))
            plus.clicked.connect(lambda _, a=label: self.jog_requested.emit(a, self.step_spin.value()))
            layout.addWidget(QLabel(f"{label}:"), row, 0)
            layout.addWidget(minus, row, 1)
            layout.addWidget(plus, row, 2)


# =============================================================================
#  ROUTINE DATA MODEL
# =============================================================================
class Routine:
    def __init__(self, name="", routine_type="pick_place", waypoints=None, pallet_config=None):
        self.name = name
        self.routine_type = routine_type
        self.waypoints: list[dict] = waypoints or []
        self.pallet_config: dict | None = pallet_config

    def as_dict(self):
        return {"name": self.name, "routine_type": self.routine_type,
                "waypoints": self.waypoints, "pallet_config": self.pallet_config}

    @staticmethod
    def from_dict(d):
        return Routine(**d)


# =============================================================================
#  MAIN APPLICATION
# =============================================================================
class RobotTeachApp(QMainWindow):
    APPROACH_Z = 200.0
    MOVE_SPEED = 100
    MOVE_SPEED_SLOW = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SME Robot Programmer")
        self.setMinimumSize(1280, 780)

        self.project_dir = os.path.abspath("SME_Vision_Project")
        os.makedirs(self.project_dir, exist_ok=True)
        self.routines_dir = os.path.join(self.project_dir, "routines")
        os.makedirs(self.routines_dir, exist_ok=True)
        self.model_path = os.path.join(self.project_dir, "sme_model", "weights", "best.pt")
        self.homography_path = os.path.join(self.project_dir, "homografia_H.npy")

        self.vision_dataset: list[dict] = []
        self.waypoints: list[Waypoint] = []
        self.current_model: YOLO | None = None
        self.cam_index_top = 0
        self.cam_index_side = 1

        self.robot = None
        self.H_matrix = None
        self.H_inv = None
        self.calib_image_pts = []
        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self._calib_update_frame)
        self.calib_cap = None

        if os.path.exists(self.homography_path):
            self.H_matrix = np.load(self.homography_path)
            self.H_inv = np.linalg.inv(self.H_matrix)

        # Guided workflow state
        self._guide_step = 0
        self._guide_waypoints: list[dict] = []
        self._guide_detections: list[dict] = []
        self._guide_current_frame = None

        # Palletizing state
        self._pal_guide_step = 0
        self._pal_waypoints: list[dict] = []

        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()

    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(240)
        sb_layout = QVBoxLayout(sidebar); sb_layout.setContentsMargins(0, 30, 0, 30); sb_layout.setSpacing(10)

        logo = QLabel("  ⚙  SME Robot"); logo.setObjectName("title")
        logo.setStyleSheet(f"color: {ACCENT}; padding: 0 12px 24px 12px; font-size: 20px;")
        sb_layout.addWidget(logo)

        self.nav_buttons: list[QPushButton] = []
        pages = [
            ("🏠  Dashboard", 0), ("📐  Calibration", 1), ("👁  Vision Training", 2),
            ("🤖  Pick & Place", 3), ("📦  Palletizing", 4), ("▶  Run Production", 5),
        ]
        for text, idx in pages:
            btn = QPushButton(text); btn.setObjectName("nav"); btn.setCheckable(True)
            btn.clicked.connect(lambda _, i=idx: self._nav(i))
            sb_layout.addWidget(btn); self.nav_buttons.append(btn)
        sb_layout.addStretch()

        cam_box = QGroupBox("Camera Indices"); cam_lay = QGridLayout(cam_box)
        cam_lay.setContentsMargins(20, 20, 20, 20)
        self.spin_cam_top = QSpinBox(); self.spin_cam_top.setRange(0, 10); self.spin_cam_top.setValue(0)
        self.spin_cam_side = QSpinBox(); self.spin_cam_side.setRange(0, 10); self.spin_cam_side.setValue(1)
        cam_lay.addWidget(QLabel("Top:"), 0, 0); cam_lay.addWidget(self.spin_cam_top, 0, 1)
        cam_lay.addWidget(QLabel("Side:"), 1, 0); cam_lay.addWidget(self.spin_cam_side, 1, 1)
        self.spin_cam_top.valueChanged.connect(lambda v: setattr(self, "cam_index_top", v))
        self.spin_cam_side.valueChanged.connect(lambda v: setattr(self, "cam_index_side", v))
        sb_layout.addWidget(cam_box)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_dashboard())
        self.stack.addWidget(self._page_calibration())
        self.stack.addWidget(self._page_vision())
        self.stack.addWidget(self._page_pick_place())
        self.stack.addWidget(self._page_palletize())
        self.stack.addWidget(self._page_production())

        root.addWidget(sidebar); root.addWidget(self.stack, stretch=1)
        self._nav(0)

    def _nav(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons): btn.setChecked(i == index)
        if index != 1:
            if self.calib_timer.isActive():
                self.calib_timer.stop()
                if self.calib_cap: self.calib_cap.release(); self.calib_cap = None

    # =================================================================
    #  PAGE 0: DASHBOARD
    # =================================================================
    def _page_dashboard(self) -> QWidget:
        page = QWidget(); lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 40, 40, 40); lay.setSpacing(20)

        title = QLabel("System Dashboard"); title.setObjectName("title"); lay.addWidget(title)
        sub = QLabel("Low-code robot programming for small & medium enterprises.\n"
                      "No PLC needed – direct xArm + USB camera workflow.")
        sub.setObjectName("subtitle"); sub.setWordWrap(True); lay.addWidget(sub)

        robot_box = QGroupBox("xArm Connection"); rb_lay = QHBoxLayout(robot_box)
        rb_lay.setContentsMargins(20, 30, 20, 20)
        self.ip_input = QLineEdit(); self.ip_input.setText("192.168.1.197")
        self.btn_connect = QPushButton("🔌 Connect Robot"); self.btn_connect.setObjectName("action")
        self.btn_connect.clicked.connect(self._connect_robot)
        rb_lay.addWidget(QLabel("Robot IP:")); rb_lay.addWidget(self.ip_input)
        rb_lay.addWidget(self.btn_connect); rb_lay.addStretch()
        lay.addWidget(robot_box)

        grid = QGridLayout(); grid.setSpacing(24)
        self.dash_model_lbl = QLabel(f"Model: {'✓ Found' if os.path.isfile(self.model_path) else '✗ Not trained'}")
        self.dash_dataset_lbl = QLabel("Dataset images: 0")
        self.dash_wp_lbl = QLabel("Waypoints saved: 0")
        self.dash_routines_lbl = QLabel(f"Routines: {len(self._list_routines())}")
        for i, (t, w) in enumerate([("Vision Model", self.dash_model_lbl), ("Training Data", self.dash_dataset_lbl),
                                     ("Teach Points", self.dash_wp_lbl), ("Routines", self.dash_routines_lbl)]):
            card = QGroupBox(t); cl = QVBoxLayout(card); cl.setContentsMargins(20, 30, 20, 20); cl.addWidget(w)
            grid.addWidget(card, 0, i)
        lay.addLayout(grid)

        info_box = QGroupBox("Quick Start Guide"); info_lay = QVBoxLayout(info_box)
        info_lay.setContentsMargins(20, 30, 20, 20)
        info_text = QLabel(
            "1.  Calibration → Map camera pixels to robot mm coordinates.\n"
            "2.  Vision Training → Capture, label, and train YOLO.\n"
            "3.  Pick & Place → Guided step-by-step waypoint teaching.\n"
            "4.  Palletizing → Define grid, teach primary waypoint.\n"
            "5.  Run Production → Select and execute saved routines."
        )
        info_text.setStyleSheet("line-height: 1.5; font-size: 14px; font-weight: normal;")
        info_lay.addWidget(info_text)
        lay.addWidget(info_box); lay.addStretch()
        return page

    def _connect_robot(self):
        if not XARM_AVAILABLE:
            QMessageBox.warning(self, "Library Missing", "xarm SDK not found. Simulation mode.")
            return
        ip = self.ip_input.text().strip()
        try:
            self.robot = XArmAPI(ip)
            self.robot.clean_warn(); self.robot.clean_error()
            self.robot.motion_enable(True); self.robot.set_mode(0); self.robot.set_state(0)
            self.btn_connect.setText("✅ Connected")
            self.btn_connect.setStyleSheet(f"background-color: {SUCCESS}; color: {DARK_BG};")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))

    # =================================================================
    #  PAGE 1: CALIBRATION
    # =================================================================
    def _page_calibration(self) -> QWidget:
        page = QWidget(); lay = QVBoxLayout(page)
        lay.setContentsMargins(30, 30, 30, 30); lay.setSpacing(16)
        title = QLabel("📐 Camera-to-Robot Calibration"); title.setObjectName("title"); lay.addWidget(title)
        desc = QLabel("Place the asymmetric circle pattern under the top camera.")
        desc.setObjectName("subtitle"); lay.addWidget(desc)
        self.calib_status = QLabel(f"Homography: {'✅ Loaded' if self.H_matrix is not None else '❌ Missing'}")
        self.calib_status.setObjectName("status"); lay.addWidget(self.calib_status)

        content = QHBoxLayout(); content.setSpacing(24)
        left = QVBoxLayout()
        self.calib_viewer = ClickableVideoWidget(); self.calib_viewer.setMinimumHeight(350)
        self.calib_viewer.pixel_clicked.connect(self._calib_on_image_click)
        left.addWidget(self.calib_viewer, stretch=1)
        cam_ctrl = QHBoxLayout()
        self.btn_calib_feed = QPushButton("▶ Start Live Feed"); self.btn_calib_feed.setObjectName("action")
        self.btn_calib_feed.clicked.connect(self._calib_toggle_feed)
        self.btn_calib_capture = QPushButton("📷 Capture Pattern"); self.btn_calib_capture.setObjectName("action")
        self.btn_calib_capture.clicked.connect(self._calib_capture_pattern)
        cam_ctrl.addWidget(self.btn_calib_feed); cam_ctrl.addWidget(self.btn_calib_capture)
        left.addLayout(cam_ctrl)
        self.calib_log = QTextEdit(); self.calib_log.setReadOnly(True); self.calib_log.setMaximumHeight(150)
        left.addWidget(self.calib_log); content.addLayout(left, stretch=2)

        right = QVBoxLayout(); right.setSpacing(16)
        param_box = QGroupBox("Pattern & Offset (mm)"); pg = QGridLayout(param_box)
        pg.setContentsMargins(16, 26, 16, 16)
        self.calib_cols = QSpinBox(); self.calib_cols.setRange(2, 20); self.calib_cols.setValue(4)
        self.calib_rows = QSpinBox(); self.calib_rows.setRange(2, 20); self.calib_rows.setValue(11)
        self.calib_sq = QDoubleSpinBox(); self.calib_sq.setRange(1.0, 100.0); self.calib_sq.setValue(15.0)
        self.calib_off_x = QDoubleSpinBox(); self.calib_off_x.setRange(-500, 500); self.calib_off_x.setValue(200.0)
        self.calib_off_y = QDoubleSpinBox(); self.calib_off_y.setRange(-500, 500); self.calib_off_y.setValue(0.0)
        for i, (lbl, w) in enumerate([("Cols:", self.calib_cols), ("Rows:", self.calib_rows),
                                       ("Spacing:", self.calib_sq), ("Offset X:", self.calib_off_x),
                                       ("Offset Y:", self.calib_off_y)]):
            pg.addWidget(QLabel(lbl), i, 0); pg.addWidget(w, i, 1)
        right.addWidget(param_box)
        btn_compute = QPushButton("🧮 Compute Homography"); btn_compute.setObjectName("action")
        btn_compute.clicked.connect(self._calib_compute)
        right.addWidget(btn_compute); right.addStretch()
        content.addLayout(right, stretch=1)
        lay.addLayout(content, stretch=1)
        return page

    def _calib_toggle_feed(self):
        if self.calib_timer.isActive():
            self.calib_timer.stop()
            if self.calib_cap: self.calib_cap.release(); self.calib_cap = None
            self.btn_calib_feed.setText("▶ Start Live Feed")
        else:
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold, params.maxThreshold, params.thresholdStep = 0, 255, 5
            params.filterByArea, params.minArea, params.maxArea = True, 150, 50000
            params.filterByCircularity, params.minCircularity = True, 0.55
            params.filterByColor, params.blobColor = True, 0
            self.calib_detector = cv2.SimpleBlobDetector_create(params)
            self.calib_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.calib_cap = cv2.VideoCapture(self.cam_index_top)
            if self.calib_cap.isOpened():
                self.calib_timer.start(30); self.btn_calib_feed.setText("⏹ Stop Live Feed")
            else: QMessageBox.warning(self, "Camera", "Failed to open top camera.")

    def _calib_update_frame(self):
        if not self.calib_cap: return
        ret, frame = self.calib_cap.read()
        if not ret: return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = self.calib_clahe.apply(gray)
        kps = self.calib_detector.detect(gray_eq)
        display = cv2.drawKeypoints(frame, kps, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cols, rows = self.calib_cols.value(), self.calib_rows.value()
        found, centers = cv2.findCirclesGrid(gray_eq, (cols, rows),
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING, blobDetector=self.calib_detector)
        if found: cv2.drawChessboardCorners(display, (cols, rows), centers, found)
        self.calib_viewer.set_image(display)

    def _calib_capture_pattern(self):
        if not self.calib_cap: QMessageBox.warning(self, "Warning", "Start live feed first."); return
        ret, frame = self.calib_cap.read()
        if not ret: return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = self.calib_clahe.apply(gray)
        cols, rows = self.calib_cols.value(), self.calib_rows.value()
        found, centers = cv2.findCirclesGrid(gray_eq, (cols, rows),
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING, blobDetector=self.calib_detector)
        if found:
            self.calib_image_pts.append(centers.reshape(-1, 2))
            self.calib_log.append(f"✅ Capture {len(self.calib_image_pts)}: {len(centers)} points.")
        else: self.calib_log.append("❌ Pattern not found.")

    def _calib_compute(self):
        if not self.calib_image_pts:
            QMessageBox.warning(self, "No Data", "Capture at least one pattern."); return
        cols, rows, sq = self.calib_cols.value(), self.calib_rows.value(), self.calib_sq.value()
        objp = np.zeros((rows * cols, 2), np.float32)
        for i in range(rows):
            for j in range(cols):
                objp[i * cols + j] = (j * sq * 2 + (i % 2) * sq, i * sq)
        H, _ = cv2.findHomography(np.vstack(self.calib_image_pts), np.vstack([objp] * len(self.calib_image_pts)))
        self.H_matrix = H; self.H_inv = np.linalg.inv(H)
        np.save(self.homography_path, H)
        self.calib_status.setText("Homography: ✅ Loaded")
        self.calib_log.append("✅ Homography saved."); self.calib_image_pts.clear()

    def _calib_on_image_click(self, px, py):
        if self.H_inv is None: return
        p = cv2.perspectiveTransform(np.array([[[px, py]]], dtype=np.float32), self.H_inv)
        wx = p[0][0][0] + self.calib_off_x.value()
        wy = p[0][0][1] + self.calib_off_y.value()
        self.calib_log.append(f"Pixel ({px:.0f},{py:.0f}) → Robot ({wx:.1f},{wy:.1f}) mm")

    # =================================================================
    #  PAGE 2: VISION TRAINING
    # =================================================================
    def _page_vision(self) -> QWidget:
        page = QWidget(); lay = QVBoxLayout(page)
        lay.setContentsMargins(30, 30, 30, 30); lay.setSpacing(16)
        title = QLabel("👁 Vision Training"); title.setObjectName("title"); lay.addWidget(title)
        desc = QLabel("Capture images, draw bounding boxes, train YOLO.")
        desc.setObjectName("subtitle"); lay.addWidget(desc)
        self.vis_status = QLabel("Ready."); self.vis_status.setObjectName("status"); lay.addWidget(self.vis_status)
        self.vis_progress = QProgressBar(); self.vis_progress.setValue(0); lay.addWidget(self.vis_progress)
        self.vis_viewer = BoundingBoxWidget(); lay.addWidget(self.vis_viewer, stretch=1)

        ctrl = QHBoxLayout(); ctrl.setSpacing(12)
        btn_cap = QPushButton("📷 Capture"); btn_cap.setObjectName("action"); btn_cap.clicked.connect(self._vis_capture)
        self.btn_vis_save = QPushButton("💾 Save"); self.btn_vis_save.setObjectName("action")
        self.btn_vis_save.setEnabled(False); self.btn_vis_save.clicked.connect(self._vis_save)
        self.btn_vis_undo = QPushButton("↩ Undo Box"); self.btn_vis_undo.setObjectName("action")
        self.btn_vis_undo.clicked.connect(self._vis_undo_box)
        self.btn_vis_train = QPushButton("🚀 Train YOLO"); self.btn_vis_train.setObjectName("action")
        self.btn_vis_train.clicked.connect(self._vis_train)
        self.btn_vis_test = QPushButton("🔍 Test"); self.btn_vis_test.setObjectName("action")
        self.btn_vis_test.setEnabled(os.path.isfile(self.model_path))
        self.btn_vis_test.clicked.connect(self._vis_test)
        for b in (btn_cap, self.btn_vis_save, self.btn_vis_undo, self.btn_vis_train, self.btn_vis_test):
            ctrl.addWidget(b)
        lay.addLayout(ctrl)

        param_box = QGroupBox("Training Parameters"); pg = QHBoxLayout(param_box)
        pg.setContentsMargins(20, 30, 20, 20); pg.setSpacing(15)
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(5, 200); self.spin_epochs.setValue(30)
        self.spin_aug = QSpinBox(); self.spin_aug.setRange(5, 100); self.spin_aug.setValue(25)
        self.spin_imgsz = QSpinBox(); self.spin_imgsz.setRange(320, 1280); self.spin_imgsz.setSingleStep(32); self.spin_imgsz.setValue(640)
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.05, 0.95); self.spin_conf.setSingleStep(0.05); self.spin_conf.setValue(0.25)
        for lbl, w in [("Epochs:", self.spin_epochs), ("Aug×:", self.spin_aug),
                       ("ImgSz:", self.spin_imgsz), ("Conf:", self.spin_conf)]:
            pg.addWidget(QLabel(lbl)); pg.addWidget(w)
        lay.addWidget(param_box)
        self.vis_log = QTextEdit(); self.vis_log.setReadOnly(True); self.vis_log.setMaximumHeight(100)
        lay.addWidget(self.vis_log)
        return page

    def _vis_capture(self):
        cap = cv2.VideoCapture(self.cam_index_top); ret, frame = cap.read(); cap.release()
        if ret:
            self.vis_viewer.set_image(frame); self.btn_vis_save.setEnabled(True)
            self.vis_status.setText("Draw bounding boxes, then save.")
        else: QMessageBox.warning(self, "Camera", "Could not read from camera.")

    def _vis_undo_box(self):
        if self.vis_viewer.boxes: self.vis_viewer.boxes.pop(); self.vis_viewer.update()

    def _vis_save(self):
        if self.vis_viewer.original_image_cv is None or not self.vis_viewer.boxes:
            QMessageBox.information(self, "Info", "Draw at least one box."); return
        self.vision_dataset.append({"image": self.vis_viewer.original_image_cv.copy(),
                                     "boxes": list(self.vis_viewer.boxes)})
        self.vis_log.append(f"Saved image {len(self.vision_dataset)} ({len(self.vis_viewer.boxes)} boxes).")
        self.dash_dataset_lbl.setText(f"Dataset images: {len(self.vision_dataset)}")
        self.vis_viewer.clear_boxes(); self.btn_vis_save.setEnabled(False)

    def _vis_train(self):
        if len(self.vision_dataset) < 1:
            QMessageBox.warning(self, "Empty", "Capture at least 1 image."); return
        self.btn_vis_train.setEnabled(False); self.vis_progress.setValue(0)
        self.trainer = YoloTrainerWorker(self.vision_dataset, self.project_dir,
            self.spin_epochs.value(), self.spin_aug.value(), self.spin_imgsz.value())
        self.trainer.status_update.connect(self.vis_status.setText)
        self.trainer.progress_update.connect(self.vis_progress.setValue)
        self.trainer.training_finished.connect(self._vis_train_done)
        self.trainer.start()

    def _vis_train_done(self, best_pt):
        self.model_path = best_pt; self.dash_model_lbl.setText("Model: ✓ Trained")
        self.btn_vis_test.setEnabled(True); self.btn_vis_train.setEnabled(True)
        self.current_model = YOLO(self.model_path)
        self.vis_log.append(f"Done: {self.model_path}")

    def _vis_test(self):
        if not os.path.isfile(self.model_path): return
        cap = cv2.VideoCapture(self.cam_index_top); ret, frame = cap.read(); cap.release()
        if not ret: return
        model = YOLO(self.model_path); results = model(frame, conf=self.spin_conf.value())
        self.vis_viewer.set_image(results[0].plot())
        self.vis_status.setText(f"{len(results[0].boxes)} detections.")

    # =================================================================
    #  PAGE 3: GUIDED PICK & PLACE
    # =================================================================
    PP_STEPS = [
        "Step 1/7 — 📷 Capture image and select the PICK part from detections.",
        "Step 2/7 — Robot moved ABOVE pick. Use jog to fine-tune XY. Press Save.",
        "Step 3/7 — Lower the robot to PICK height. Use jog Z. Press Save (gripper will CLOSE).",
        "Step 4/7 — Robot ascending… Capture image for PLACE target.",
        "Step 5/7 — Select the PLACE location from detections.",
        "Step 6/7 — Robot moved ABOVE place. Use jog to fine-tune XY. Press Save.",
        "Step 7/7 — Lower robot to PLACE height. Use jog Z. Press Save (gripper will OPEN).",
    ]

    def _page_pick_place(self) -> QWidget:
        page = QWidget(); lay = QVBoxLayout(page)
        lay.setContentsMargins(30, 30, 30, 30); lay.setSpacing(12)

        title = QLabel("🤖 Guided Pick & Place"); title.setObjectName("title"); lay.addWidget(title)
        desc = QLabel("The robot guides you step-by-step to teach a pick & place routine.")
        desc.setObjectName("subtitle"); lay.addWidget(desc)

        self.pp_instruction = QLabel("Press 'Start New Routine' to begin.")
        self.pp_instruction.setObjectName("step_instruction"); self.pp_instruction.setWordWrap(True)
        lay.addWidget(self.pp_instruction)

        content = QHBoxLayout(); content.setSpacing(20)

        # Left: camera + detections
        left = QVBoxLayout(); left.setSpacing(10)
        self.pp_viewer = QLabel("Camera preview"); self.pp_viewer.setAlignment(Qt.AlignCenter)
        self.pp_viewer.setStyleSheet(f"background:#11111b; border:2px solid {BORDER}; border-radius:12px;")
        self.pp_viewer.setMinimumHeight(300)
        left.addWidget(self.pp_viewer, stretch=1)

        self.pp_det_list = QTextEdit(); self.pp_det_list.setReadOnly(True); self.pp_det_list.setMinimumHeight(100)
        left.addWidget(self.pp_det_list)

        det_sel = QHBoxLayout()
        det_sel.addWidget(QLabel("Detection #:"))
        self.pp_det_idx = QSpinBox(); self.pp_det_idx.setRange(0, 99); self.pp_det_idx.setValue(0)
        det_sel.addWidget(self.pp_det_idx)
        self.btn_pp_select = QPushButton("✅ Select & Move"); self.btn_pp_select.setObjectName("action")
        self.btn_pp_select.clicked.connect(self._pp_select_detection); self.btn_pp_select.setEnabled(False)
        det_sel.addWidget(self.btn_pp_select)
        left.addLayout(det_sel)
        content.addLayout(left, stretch=2)

        # Right: jog + save
        right = QVBoxLayout(); right.setSpacing(12)

        coord_box = QGroupBox("Current Target (mm)")
        cg = QGridLayout(coord_box); cg.setContentsMargins(16, 26, 16, 16)
        self.pp_x = QDoubleSpinBox(); self.pp_x.setRange(-1000, 1000); self.pp_x.setDecimals(1)
        self.pp_y = QDoubleSpinBox(); self.pp_y.setRange(-1000, 1000); self.pp_y.setDecimals(1)
        self.pp_z = QDoubleSpinBox(); self.pp_z.setRange(-100, 500); self.pp_z.setDecimals(1); self.pp_z.setValue(200)
        for i, (lbl, w) in enumerate([("X:", self.pp_x), ("Y:", self.pp_y), ("Z:", self.pp_z)]):
            cg.addWidget(QLabel(lbl), i, 0); cg.addWidget(w, i, 1)
        right.addWidget(coord_box)

        self.pp_jog = JogPanel("Jog Robot"); self.pp_jog.jog_requested.connect(self._pp_jog_apply)
        right.addWidget(self.pp_jog)

        btn_box = QVBoxLayout(); btn_box.setSpacing(8)
        self.btn_pp_start = QPushButton("🚀 Start New Routine"); self.btn_pp_start.setObjectName("action")
        self.btn_pp_start.clicked.connect(self._pp_start_routine); btn_box.addWidget(self.btn_pp_start)

        self.btn_pp_capture = QPushButton("📷 Capture & Detect"); self.btn_pp_capture.setObjectName("action")
        self.btn_pp_capture.clicked.connect(self._pp_capture_detect); self.btn_pp_capture.setEnabled(False)
        btn_box.addWidget(self.btn_pp_capture)

        self.btn_pp_save_wp = QPushButton("💾 Save Waypoint"); self.btn_pp_save_wp.setObjectName("step_active")
        self.btn_pp_save_wp.clicked.connect(self._pp_save_step); self.btn_pp_save_wp.setEnabled(False)
        btn_box.addWidget(self.btn_pp_save_wp)

        self.btn_pp_add_cycle = QPushButton("➕ Add Another Cycle")
        self.btn_pp_add_cycle.setObjectName("action"); self.btn_pp_add_cycle.setEnabled(False)
        self.btn_pp_add_cycle.clicked.connect(self._pp_add_cycle); btn_box.addWidget(self.btn_pp_add_cycle)

        self.btn_pp_finish = QPushButton("✅ Finish & Save Routine")
        self.btn_pp_finish.setObjectName("action"); self.btn_pp_finish.setEnabled(False)
        self.btn_pp_finish.clicked.connect(self._pp_finish_routine); btn_box.addWidget(self.btn_pp_finish)

        right.addLayout(btn_box); right.addStretch()
        content.addLayout(right, stretch=1)
        lay.addLayout(content, stretch=1)

        self.pp_log = QTextEdit(); self.pp_log.setReadOnly(True); self.pp_log.setMaximumHeight(100)
        lay.addWidget(self.pp_log)
        return page

    def _pp_start_routine(self):
        self._guide_step = 0; self._guide_waypoints = []; self._guide_detections = []
        self.pp_log.clear(); self.pp_det_list.clear()
        self.btn_pp_start.setEnabled(False); self.btn_pp_capture.setEnabled(True)
        self.btn_pp_save_wp.setEnabled(False); self.btn_pp_select.setEnabled(False)
        self.btn_pp_add_cycle.setEnabled(False); self.btn_pp_finish.setEnabled(False)
        self._pp_update_instruction()
        self.pp_log.append("--- New Pick & Place routine ---")

    def _pp_update_instruction(self):
        if self._guide_step < len(self.PP_STEPS):
            self.pp_instruction.setText(self.PP_STEPS[self._guide_step])
        else:
            self.pp_instruction.setText("✅ Cycle complete! Add another or save.")

    def _pp_capture_detect(self):
        if self.H_inv is None:
            QMessageBox.warning(self, "Error", "Calibrate camera first."); return
        if self.current_model is None:
            if os.path.isfile(self.model_path): self.current_model = YOLO(self.model_path)
            else: QMessageBox.warning(self, "Error", "Train YOLO first."); return

        cap = cv2.VideoCapture(self.cam_index_top); ret, frame = cap.read(); cap.release()
        if not ret: return

        self._guide_current_frame = frame.copy()
        results = self.current_model.predict(frame, conf=self.spin_conf.value())[0]
        annotated = frame.copy()
        self._guide_detections = []; self.pp_det_list.clear()

        if len(results.boxes) == 0:
            self.pp_det_list.append("No detections."); self._show_cv_on_label(annotated, self.pp_viewer); return

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            uc, vc = (x1 + x2) / 2, (y1 + y2) / 2
            p = cv2.perspectiveTransform(np.array([[[uc, vc]]], dtype=np.float32), self.H_inv)
            rx = p[0][0][0] + self.calib_off_x.value()
            ry = p[0][0][1] + self.calib_off_y.value()
            label = self.current_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            self._guide_detections.append({"label": label, "rx": rx, "ry": ry, "conf": conf})
            cv2.circle(annotated, (int(uc), int(vc)), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"[{i}] {label}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.pp_det_list.append(f"[{i}] {label}: X={rx:.1f}, Y={ry:.1f} (conf={conf:.2f})")

        self._show_cv_on_label(annotated, self.pp_viewer)
        self.pp_det_idx.setRange(0, len(self._guide_detections) - 1)
        self.btn_pp_select.setEnabled(True); self.btn_pp_capture.setEnabled(False)

    def _pp_select_detection(self):
        idx = self.pp_det_idx.value()
        if idx >= len(self._guide_detections): return
        det = self._guide_detections[idx]
        self.pp_x.setValue(det["rx"]); self.pp_y.setValue(det["ry"]); self.pp_z.setValue(self.APPROACH_Z)
        self._robot_move_to(det["rx"], det["ry"], self.APPROACH_Z, speed=self.MOVE_SPEED, move_type="moveJ")
        self.btn_pp_select.setEnabled(False); self.btn_pp_save_wp.setEnabled(True)
        if self._guide_step == 0: self._guide_step = 1
        elif self._guide_step == 4: self._guide_step = 5
        self._pp_update_instruction()
        self.pp_log.append(f"Selected [{idx}] '{det['label']}' → above ({det['rx']:.1f}, {det['ry']:.1f})")

    def _pp_save_step(self):
        x, y, z = self.pp_x.value(), self.pp_y.value(), self.pp_z.value()

        if self._guide_step == 1:
            self._guide_waypoints.append({"label": "approach_pick", "x": x, "y": y, "z": z,
                                           "gripper": "open", "move_type": "moveJ"})
            self._guide_step = 2; self.pp_z.setValue(z - 100)
            self.pp_log.append(f"Saved approach_pick. Now lower to pick height.")

        elif self._guide_step == 2:
            self._guide_waypoints.append({"label": "pick", "x": x, "y": y, "z": z,
                                           "gripper": "close", "move_type": "moveL"})
            self._gripper_action("close")
            ap = self._guide_waypoints[-2]
            self._guide_waypoints.append({"label": "ascend_after_pick", "x": ap["x"], "y": ap["y"],
                                           "z": ap["z"], "gripper": "close", "move_type": "moveL"})
            self._robot_move_to(ap["x"], ap["y"], ap["z"], speed=self.MOVE_SPEED_SLOW, move_type="moveL")
            self.pp_x.setValue(ap["x"]); self.pp_y.setValue(ap["y"]); self.pp_z.setValue(ap["z"])
            self._guide_step = 4
            self.btn_pp_save_wp.setEnabled(False); self.btn_pp_capture.setEnabled(True)
            self.pp_log.append("Picked. Robot ascending. Capture for PLACE.")

        elif self._guide_step == 5:
            self._guide_waypoints.append({"label": "approach_place", "x": x, "y": y, "z": z,
                                           "gripper": "close", "move_type": "moveJ"})
            self._guide_step = 6; self.pp_z.setValue(z - 100)
            self.pp_log.append(f"Saved approach_place. Now lower to place height.")

        elif self._guide_step == 6:
            self._guide_waypoints.append({"label": "place", "x": x, "y": y, "z": z,
                                           "gripper": "open", "move_type": "moveL"})
            self._gripper_action("open")
            ap = self._guide_waypoints[-2]
            self._guide_waypoints.append({"label": "ascend_after_place", "x": ap["x"], "y": ap["y"],
                                           "z": ap["z"], "gripper": "open", "move_type": "moveL"})
            self._robot_move_to(ap["x"], ap["y"], ap["z"], speed=self.MOVE_SPEED_SLOW, move_type="moveL")
            self._guide_step = 7
            self.btn_pp_save_wp.setEnabled(False)
            self.btn_pp_add_cycle.setEnabled(True); self.btn_pp_finish.setEnabled(True)
            self.pp_log.append("✅ Cycle complete!")

        self._pp_update_instruction()

    def _pp_add_cycle(self):
        self._guide_step = 0
        self.btn_pp_add_cycle.setEnabled(False); self.btn_pp_finish.setEnabled(False)
        self.btn_pp_capture.setEnabled(True)
        self._pp_update_instruction()
        self.pp_log.append("--- Adding another cycle ---")

    def _pp_finish_routine(self):
        name, ok = QInputDialog.getText(self, "Routine Name", "Enter a name for this routine:")
        if not ok or not name.strip(): return
        name = name.strip().replace(" ", "_")
        routine = Routine(name=name, routine_type="pick_place", waypoints=self._guide_waypoints)
        self._save_routine(routine)
        self.pp_log.append(f"✅ Routine '{name}' saved ({len(self._guide_waypoints)} waypoints).")
        self.btn_pp_start.setEnabled(True); self.btn_pp_add_cycle.setEnabled(False)
        self.btn_pp_finish.setEnabled(False); self.btn_pp_capture.setEnabled(False)
        self.pp_instruction.setText("Routine saved! Start another or go to Production.")
        self.dash_routines_lbl.setText(f"Routines: {len(self._list_routines())}")

    def _pp_jog_apply(self, axis, delta):
        mapping = {"X": self.pp_x, "Y": self.pp_y, "Z": self.pp_z}
        spin = mapping.get(axis)
        if spin: spin.setValue(spin.value() + delta)
        self._robot_move_to(self.pp_x.value(), self.pp_y.value(), self.pp_z.value(),
                            speed=self.MOVE_SPEED_SLOW, move_type="moveL")

    # =================================================================
    #  PAGE 4: GUIDED PALLETIZING
    # =================================================================
    PAL_STEPS = [
        "Step 1/5 — Configure grid. Then Capture & select PICK part.",
        "Step 2/5 — Robot above pick. Jog XY. Save approach_pick.",
        "Step 3/5 — Lower to pick. Jog Z. Save pick (gripper CLOSE).",
        "Step 4/5 — Robot ascending. Jog to PALLET ORIGIN (first cell). Save approach_place.",
        "Step 5/5 — Lower to place. Jog Z. Save place (gripper OPEN). Grid auto-generated.",
    ]

    def _page_palletize(self) -> QWidget:
        page = QWidget(); lay = QVBoxLayout(page)
        lay.setContentsMargins(30, 30, 30, 30); lay.setSpacing(12)

        title = QLabel("📦 Guided Palletizing"); title.setObjectName("title"); lay.addWidget(title)
        desc = QLabel("Teach one pick + one place (pallet origin). Grid auto-generated from offsets.")
        desc.setObjectName("subtitle"); lay.addWidget(desc)

        self.pal_instruction = QLabel("Press 'Start' to begin.")
        self.pal_instruction.setObjectName("step_instruction"); self.pal_instruction.setWordWrap(True)
        lay.addWidget(self.pal_instruction)

        content = QHBoxLayout(); content.setSpacing(20)

        # Left
        left = QVBoxLayout(); left.setSpacing(10)
        self.pal_viewer = QLabel("Camera"); self.pal_viewer.setAlignment(Qt.AlignCenter)
        self.pal_viewer.setStyleSheet(f"background:#11111b; border:2px solid {BORDER}; border-radius:12px;")
        self.pal_viewer.setMinimumHeight(250)
        left.addWidget(self.pal_viewer, stretch=1)

        self.pal_det_list = QTextEdit(); self.pal_det_list.setReadOnly(True); self.pal_det_list.setMaximumHeight(100)
        left.addWidget(self.pal_det_list)

        det_sel = QHBoxLayout()
        det_sel.addWidget(QLabel("Detection #:"))
        self.pal_det_idx = QSpinBox(); self.pal_det_idx.setRange(0, 99)
        det_sel.addWidget(self.pal_det_idx)
        self.btn_pal_select = QPushButton("✅ Select"); self.btn_pal_select.setObjectName("action")
        self.btn_pal_select.clicked.connect(self._pal_select_detection); self.btn_pal_select.setEnabled(False)
        det_sel.addWidget(self.btn_pal_select)
        left.addLayout(det_sel)
        content.addLayout(left, stretch=2)

        # Right
        right = QVBoxLayout(); right.setSpacing(10)

        grid_box = QGroupBox("Grid & Object Size")
        gg = QGridLayout(grid_box); gg.setContentsMargins(12, 24, 12, 12)
        self.pal_obj_w = QDoubleSpinBox(); self.pal_obj_w.setRange(10, 500); self.pal_obj_w.setValue(50); self.pal_obj_w.setSuffix(" mm")
        self.pal_obj_d = QDoubleSpinBox(); self.pal_obj_d.setRange(10, 500); self.pal_obj_d.setValue(50); self.pal_obj_d.setSuffix(" mm")
        self.pal_obj_h = QDoubleSpinBox(); self.pal_obj_h.setRange(10, 500); self.pal_obj_h.setValue(20); self.pal_obj_h.setSuffix(" mm")
        self.pal_cols = QSpinBox(); self.pal_cols.setRange(1, 10); self.pal_cols.setValue(3)
        self.pal_rows = QSpinBox(); self.pal_rows.setRange(1, 10); self.pal_rows.setValue(3)
        self.pal_layers = QSpinBox(); self.pal_layers.setRange(1, 5); self.pal_layers.setValue(1)
        self.pal_gap_x = QDoubleSpinBox(); self.pal_gap_x.setRange(0, 50); self.pal_gap_x.setValue(5); self.pal_gap_x.setSuffix(" mm")
        self.pal_gap_y = QDoubleSpinBox(); self.pal_gap_y.setRange(0, 50); self.pal_gap_y.setValue(5); self.pal_gap_y.setSuffix(" mm")
        for i, (lbl, w) in enumerate([("Obj W:", self.pal_obj_w), ("Obj D:", self.pal_obj_d),
                                       ("Obj H:", self.pal_obj_h), ("Cols:", self.pal_cols),
                                       ("Rows:", self.pal_rows), ("Layers:", self.pal_layers),
                                       ("Gap X:", self.pal_gap_x), ("Gap Y:", self.pal_gap_y)]):
            gg.addWidget(QLabel(lbl), i // 2, (i % 2) * 2)
            gg.addWidget(w, i // 2, (i % 2) * 2 + 1)
        right.addWidget(grid_box)

        coord_box = QGroupBox("Position (mm)")
        cg = QGridLayout(coord_box); cg.setContentsMargins(12, 24, 12, 12)
        self.pal_x = QDoubleSpinBox(); self.pal_x.setRange(-1000, 1000); self.pal_x.setDecimals(1)
        self.pal_y = QDoubleSpinBox(); self.pal_y.setRange(-1000, 1000); self.pal_y.setDecimals(1)
        self.pal_z = QDoubleSpinBox(); self.pal_z.setRange(-100, 500); self.pal_z.setDecimals(1); self.pal_z.setValue(200)
        for i, (lbl, w) in enumerate([("X:", self.pal_x), ("Y:", self.pal_y), ("Z:", self.pal_z)]):
            cg.addWidget(QLabel(lbl), i, 0); cg.addWidget(w, i, 1)
        right.addWidget(coord_box)

        self.pal_jog = JogPanel("Jog"); self.pal_jog.jog_requested.connect(self._pal_jog_apply)
        right.addWidget(self.pal_jog)

        btn_box = QVBoxLayout(); btn_box.setSpacing(6)
        self.btn_pal_start = QPushButton("🚀 Start Palletizing"); self.btn_pal_start.setObjectName("action")
        self.btn_pal_start.clicked.connect(self._pal_start); btn_box.addWidget(self.btn_pal_start)

        self.btn_pal_capture = QPushButton("📷 Capture & Detect"); self.btn_pal_capture.setObjectName("action")
        self.btn_pal_capture.clicked.connect(self._pal_capture); self.btn_pal_capture.setEnabled(False)
        btn_box.addWidget(self.btn_pal_capture)

        self.btn_pal_save_wp = QPushButton("💾 Save Waypoint"); self.btn_pal_save_wp.setObjectName("step_active")
        self.btn_pal_save_wp.clicked.connect(self._pal_save_step); self.btn_pal_save_wp.setEnabled(False)
        btn_box.addWidget(self.btn_pal_save_wp)

        self.btn_pal_finish = QPushButton("✅ Save Routine"); self.btn_pal_finish.setObjectName("action")
        self.btn_pal_finish.clicked.connect(self._pal_finish); self.btn_pal_finish.setEnabled(False)
        btn_box.addWidget(self.btn_pal_finish)

        right.addLayout(btn_box); right.addStretch()
        content.addLayout(right, stretch=1)
        lay.addLayout(content, stretch=1)

        self.pal_log = QTextEdit(); self.pal_log.setReadOnly(True); self.pal_log.setMaximumHeight(100)
        lay.addWidget(self.pal_log)
        return page

    def _pal_start(self):
        self._pal_guide_step = 0; self._pal_waypoints = []; self._guide_detections = []
        self.pal_log.clear(); self.pal_det_list.clear()
        self.btn_pal_start.setEnabled(False); self.btn_pal_capture.setEnabled(True)
        self.btn_pal_save_wp.setEnabled(False); self.btn_pal_select.setEnabled(False)
        self.btn_pal_finish.setEnabled(False)
        self._pal_update_instruction()

    def _pal_update_instruction(self):
        if self._pal_guide_step < len(self.PAL_STEPS):
            self.pal_instruction.setText(self.PAL_STEPS[self._pal_guide_step])
        else:
            self.pal_instruction.setText("✅ Done! Save the routine.")

    def _pal_capture(self):
        if self.H_inv is None: QMessageBox.warning(self, "Error", "Calibrate first."); return
        if self.current_model is None:
            if os.path.isfile(self.model_path): self.current_model = YOLO(self.model_path)
            else: QMessageBox.warning(self, "Error", "Train YOLO first."); return

        cap = cv2.VideoCapture(self.cam_index_top); ret, frame = cap.read(); cap.release()
        if not ret: return
        results = self.current_model.predict(frame, conf=self.spin_conf.value())[0]
        annotated = frame.copy(); self._guide_detections = []; self.pal_det_list.clear()

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            uc, vc = (x1 + x2) / 2, (y1 + y2) / 2
            p = cv2.perspectiveTransform(np.array([[[uc, vc]]], dtype=np.float32), self.H_inv)
            rx = p[0][0][0] + self.calib_off_x.value()
            ry = p[0][0][1] + self.calib_off_y.value()
            label = self.current_model.names[int(box.cls[0])]
            self._guide_detections.append({"label": label, "rx": rx, "ry": ry})
            cv2.circle(annotated, (int(uc), int(vc)), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"[{i}] {label}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.pal_det_list.append(f"[{i}] {label}: X={rx:.1f}, Y={ry:.1f}")

        self._show_cv_on_label(annotated, self.pal_viewer)
        if self._guide_detections:
            self.pal_det_idx.setRange(0, len(self._guide_detections) - 1)
            self.btn_pal_select.setEnabled(True); self.btn_pal_capture.setEnabled(False)

    def _pal_select_detection(self):
        idx = self.pal_det_idx.value()
        if idx >= len(self._guide_detections): return
        det = self._guide_detections[idx]
        self.pal_x.setValue(det["rx"]); self.pal_y.setValue(det["ry"]); self.pal_z.setValue(self.APPROACH_Z)
        self._robot_move_to(det["rx"], det["ry"], self.APPROACH_Z, speed=self.MOVE_SPEED, move_type="moveJ")
        self.btn_pal_select.setEnabled(False); self.btn_pal_save_wp.setEnabled(True)
        self._pal_guide_step = 1; self._pal_update_instruction()

    def _pal_save_step(self):
        x, y, z = self.pal_x.value(), self.pal_y.value(), self.pal_z.value()

        if self._pal_guide_step == 1:
            self._pal_waypoints.append({"label": "approach_pick", "x": x, "y": y, "z": z,
                                         "gripper": "open", "move_type": "moveJ"})
            self._pal_guide_step = 2; self.pal_z.setValue(z - 100)
            self.pal_log.append("Saved approach_pick. Lower to pick.")

        elif self._pal_guide_step == 2:
            self._pal_waypoints.append({"label": "pick", "x": x, "y": y, "z": z,
                                         "gripper": "close", "move_type": "moveL"})
            self._gripper_action("close")
            ap = self._pal_waypoints[0]
            self._pal_waypoints.append({"label": "ascend_after_pick", "x": ap["x"], "y": ap["y"],
                                         "z": ap["z"], "gripper": "close", "move_type": "moveL"})
            self._robot_move_to(ap["x"], ap["y"], ap["z"], speed=self.MOVE_SPEED_SLOW, move_type="moveL")
            self.pal_x.setValue(ap["x"]); self.pal_y.setValue(ap["y"]); self.pal_z.setValue(ap["z"])
            self._pal_guide_step = 3
            self.pal_log.append("Picked. Jog to pallet origin.")

        elif self._pal_guide_step == 3:
            self._pal_waypoints.append({"label": "approach_place", "x": x, "y": y, "z": z,
                                         "gripper": "close", "move_type": "moveJ"})
            self._pal_guide_step = 4; self.pal_z.setValue(z - 100)
            self.pal_log.append("Saved approach_place. Lower to place.")

        elif self._pal_guide_step == 4:
            self._pal_waypoints.append({"label": "place_primary", "x": x, "y": y, "z": z,
                                         "gripper": "open", "move_type": "moveL"})
            self._gripper_action("open")
            self._pal_guide_step = 5
            self.btn_pal_save_wp.setEnabled(False); self.btn_pal_finish.setEnabled(True)
            self.pal_log.append(f"Primary place saved. Grid will auto-generate.")

        self._pal_update_instruction()

    def _pal_finish(self):
        name, ok = QInputDialog.getText(self, "Routine Name", "Name:")
        if not ok or not name.strip(): return
        name = name.strip().replace(" ", "_")
        pallet_config = {
            "obj_w": self.pal_obj_w.value(), "obj_d": self.pal_obj_d.value(), "obj_h": self.pal_obj_h.value(),
            "cols": self.pal_cols.value(), "rows": self.pal_rows.value(), "layers": self.pal_layers.value(),
            "gap_x": self.pal_gap_x.value(), "gap_y": self.pal_gap_y.value(),
        }
        routine = Routine(name=name, routine_type="palletizing",
                          waypoints=self._pal_waypoints, pallet_config=pallet_config)
        self._save_routine(routine)
        self.pal_log.append(f"✅ Routine '{name}' saved.")
        self.btn_pal_start.setEnabled(True); self.btn_pal_finish.setEnabled(False)
        self.pal_instruction.setText("Routine saved!")
        self.dash_routines_lbl.setText(f"Routines: {len(self._list_routines())}")

    def _pal_jog_apply(self, axis, delta):
        mapping = {"X": self.pal_x, "Y": self.pal_y, "Z": self.pal_z}
        spin = mapping.get(axis)
        if spin: spin.setValue(spin.value() + delta)
        self._robot_move_to(self.pal_x.value(), self.pal_y.value(), self.pal_z.value(),
                            speed=self.MOVE_SPEED_SLOW, move_type="moveL")

    # =================================================================
    #  PAGE 5: PRODUCTION
    # =================================================================
    def _page_production(self) -> QWidget:
        page = QWidget(); lay = QVBoxLayout(page)
        lay.setContentsMargins(30, 30, 30, 30); lay.setSpacing(16)

        title = QLabel("▶ Run Production"); title.setObjectName("title"); lay.addWidget(title)
        desc = QLabel("Select a saved routine and execute it.")
        desc.setObjectName("subtitle"); desc.setWordWrap(True); lay.addWidget(desc)

        self.prod_status = QLabel("Idle."); self.prod_status.setObjectName("status"); lay.addWidget(self.prod_status)

        cam_frame = QHBoxLayout(); cam_frame.setSpacing(24)
        self.prod_cam1 = QLabel("Top Camera"); self.prod_cam1.setAlignment(Qt.AlignCenter)
        self.prod_cam1.setStyleSheet(f"background:#11111b; border:2px solid {BORDER}; border-radius:12px; font-weight:normal;")
        self.prod_cam1.setMinimumHeight(240)
        self.prod_cam2 = QLabel("Side Camera"); self.prod_cam2.setAlignment(Qt.AlignCenter)
        self.prod_cam2.setStyleSheet(f"background:#11111b; border:2px solid {BORDER}; border-radius:12px; font-weight:normal;")
        self.prod_cam2.setMinimumHeight(240)
        cam_frame.addWidget(self.prod_cam1); cam_frame.addWidget(self.prod_cam2)
        lay.addLayout(cam_frame, stretch=1)

        mode_box = QGroupBox("Routine Selection")
        ml = QHBoxLayout(mode_box); ml.setContentsMargins(16, 26, 16, 16); ml.setSpacing(16)
        self.prod_type_filter = QComboBox()
        self.prod_type_filter.addItems(["All", "pick_place", "palletizing"])
        self.prod_type_filter.currentTextChanged.connect(self._prod_refresh_routines)
        ml.addWidget(QLabel("Type:")); ml.addWidget(self.prod_type_filter)
        self.prod_routine_combo = QComboBox()
        ml.addWidget(QLabel("Routine:")); ml.addWidget(self.prod_routine_combo)
        btn_refresh = QPushButton("🔄"); btn_refresh.setObjectName("action")
        btn_refresh.clicked.connect(self._prod_refresh_routines); ml.addWidget(btn_refresh)
        self.prod_cycles = QSpinBox(); self.prod_cycles.setRange(1, 9999); self.prod_cycles.setValue(1)
        ml.addWidget(QLabel("Cycles:")); ml.addWidget(self.prod_cycles)
        btn_run = QPushButton("🚀 RUN"); btn_run.setObjectName("action"); btn_run.clicked.connect(self._prod_run)
        btn_stop = QPushButton("🛑 E-STOP"); btn_stop.setObjectName("danger"); btn_stop.clicked.connect(self._prod_stop)
        ml.addWidget(btn_run); ml.addWidget(btn_stop)
        lay.addWidget(mode_box)

        self.prod_log = QTextEdit(); self.prod_log.setReadOnly(True)
        lay.addWidget(self.prod_log, stretch=1)
        self._prod_refresh_routines()
        return page

    def _prod_refresh_routines(self):
        self.prod_routine_combo.clear()
        tf = self.prod_type_filter.currentText()
        for r in self._list_routines():
            if tf == "All" or r["routine_type"] == tf:
                self.prod_routine_combo.addItem(f"{r['name']} ({r['routine_type']})", r["name"])

    def _prod_run(self):
        if self.prod_routine_combo.count() == 0:
            QMessageBox.warning(self, "No Routine", "Save a routine first."); return
        routine_name = self.prod_routine_combo.currentData()
        routine = self._load_routine(routine_name)
        if not routine: return
        cycles = self.prod_cycles.value()
        self.prod_status.setText(f"RUNNING: {routine.name} × {cycles}")
        self.prod_status.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 14px;")
        self.prod_log.append(f"=== {routine.name} ({routine.routine_type}) × {cycles} ===")

        if routine.routine_type == "palletizing" and routine.pallet_config:
            self._prod_run_palletizing(routine, cycles)
        else:
            self._prod_run_pick_place(routine, cycles)

    def _prod_run_pick_place(self, routine: Routine, cycles: int):
        for cycle in range(cycles):
            self.prod_log.append(f"\n--- Cycle {cycle + 1}/{cycles} ---")
            for wp in routine.waypoints:
                self.prod_log.append(f"  → {wp['label']}: ({wp['x']:.1f},{wp['y']:.1f},{wp['z']:.1f}) g={wp.get('gripper','?')}")
                self._robot_move_to(wp["x"], wp["y"], wp["z"], speed=self.MOVE_SPEED,
                                    move_type=wp.get("move_type", "moveL"))
                g = wp.get("gripper")
                if g in ("open", "close"): self._gripper_action(g)
                QApplication.processEvents()
        self.prod_log.append("\n✅ Done."); self.prod_status.setText("Idle.")

    def _prod_run_palletizing(self, routine: Routine, cycles: int):
        cfg = routine.pallet_config
        cols, rows, layers = cfg["cols"], cfg["rows"], cfg["layers"]
        step_x = cfg["obj_w"] + cfg["gap_x"]
        step_y = cfg["obj_d"] + cfg["gap_y"]
        step_z = cfg["obj_h"]

        place_wp = approach_wp = None
        for wp in routine.waypoints:
            if wp["label"] == "place_primary": place_wp = wp
            if wp["label"] == "approach_place": approach_wp = wp
        if not place_wp or not approach_wp:
            self.prod_log.append("ERROR: missing place waypoints."); return

        pick_wps = []
        for wp in routine.waypoints:
            if wp["label"] == "approach_place": break
            pick_wps.append(wp)

        cell = 0; total = min(cycles, cols * rows * layers)
        for L in range(layers):
            for R in range(rows):
                for C in range(cols):
                    if cell >= total: break
                    self.prod_log.append(f"\n--- Cell {cell+1}/{total} L{L}R{R}C{C} ---")
                    for wp in pick_wps:
                        self._robot_move_to(wp["x"], wp["y"], wp["z"], speed=self.MOVE_SPEED,
                                            move_type=wp.get("move_type", "moveL"))
                        g = wp.get("gripper")
                        if g in ("open", "close"): self._gripper_action(g)
                        QApplication.processEvents()
                    ox = place_wp["x"] + C * step_x
                    oy = place_wp["y"] + R * step_y
                    oz = place_wp["z"] + L * step_z
                    az = approach_wp["z"] + L * step_z
                    self._robot_move_to(ox, oy, az, speed=self.MOVE_SPEED, move_type="moveJ")
                    self._robot_move_to(ox, oy, oz, speed=self.MOVE_SPEED_SLOW, move_type="moveL")
                    self._gripper_action("open")
                    self._robot_move_to(ox, oy, az, speed=self.MOVE_SPEED_SLOW, move_type="moveL")
                    cell += 1; QApplication.processEvents()
                if cell >= total: break
            if cell >= total: break
        self.prod_log.append(f"\n✅ Palletized {cell} cells."); self.prod_status.setText("Idle.")

    def _prod_stop(self):
        self.prod_status.setText("⚠ E-STOP")
        if self.robot:
            self.robot.set_state(4)
            self.prod_log.append("E-STOP SENT")

    # =================================================================
    #  ROBOT HELPERS
    # =================================================================
    def _robot_move_to(self, x, y, z, roll=-180, pitch=0, yaw=0, speed=100, move_type="moveL"):
        if not self.robot: return
        try:
            self.robot.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                    speed=speed, mvacc=500, wait=True)
        except Exception as e:
            print(f"[Motion error] {e}")

    def _gripper_action(self, action):
        if not self.robot: return
        try:
            pos = 800 if action == "open" else 0
            self.robot.set_gripper_position(pos, speed=3000, wait=True)
        except Exception as e:
            print(f"[Gripper error] {e}")

    # =================================================================
    #  ROUTINE PERSISTENCE
    # =================================================================
    def _save_routine(self, routine: Routine):
        with open(os.path.join(self.routines_dir, f"{routine.name}.json"), "w") as f:
            json.dump(routine.as_dict(), f, indent=2)

    def _load_routine(self, name) -> Routine | None:
        path = os.path.join(self.routines_dir, f"{name}.json")
        if not os.path.isfile(path): return None
        with open(path) as f: return Routine.from_dict(json.load(f))

    def _list_routines(self) -> list[dict]:
        routines = []
        if not os.path.isdir(self.routines_dir): return routines
        for fname in sorted(os.listdir(self.routines_dir)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self.routines_dir, fname)) as f:
                        d = json.load(f)
                        routines.append({"name": d["name"], "routine_type": d["routine_type"]})
                except Exception: pass
        return routines

    # =================================================================
    #  SHARED
    # =================================================================
    def _show_cv_on_label(self, cv_img, label):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        pix = QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotTeachApp()
    window.show()
    sys.exit(app.exec())