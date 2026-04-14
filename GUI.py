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
    QTextEdit, QFileDialog, QSlider, QCheckBox, QSplitter, QDialog,
    QDialogButtonBox, QButtonGroup, QListWidget, QListWidgetItem,
    QFormLayout,
)
from PySide6.QtCore import Qt, QThread, Signal, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import (
    QFont, QImage, QPixmap, QPainter, QPen, QColor, QIcon, QBrush,
)

# Deep Learning
import albumentations as A
from ultralytics import YOLO

try:
    from xarm.wrapper import XArmAPI
    XARM_AVAILABLE = True
except ImportError:
    XARM_AVAILABLE = False

# =============================================================================
#  THEME COLOURS
# =============================================================================
DARK_BG="#1e1e2e"; SIDEBAR_BG="#11111b"; CARD_BG="#313244"
ACCENT="#89b4fa"; ACCENT_HOVER="#b4befe"; SUCCESS="#a6e3a1"
WARNING="#f9e2af"; DANGER="#f38ba8"; TEXT_PRIMARY="#cdd6f4"
TEXT_DIM="#9399b2"; BORDER="#45475a"

# =============================================================================
#  COMPACT GLOBAL STYLESHEET
# =============================================================================
GLOBAL_STYLE = f"""
QMainWindow, QWidget {{ background-color:{DARK_BG}; color:{TEXT_PRIMARY}; font-family:'Segoe UI','Roboto',sans-serif; font-size:12px; }}
QLabel {{ background-color:transparent; border:none; font-weight:bold; font-size:12px; }}
QFrame#sidebar {{ background-color:{SIDEBAR_BG}; border-right:1px solid {BORDER}; }}

QPushButton#nav {{ background-color:transparent; border:none; text-align:left; padding:10px 16px; font-size:13px; font-weight:normal; color:{TEXT_DIM}; border-radius:8px; margin:2px 8px; }}
QPushButton#nav:hover {{ background-color:{CARD_BG}; color:{TEXT_PRIMARY}; }}
QPushButton#nav:checked {{ background-color:{ACCENT}; color:{DARK_BG}; font-weight:bold; }}

QPushButton#action {{ background-color:{ACCENT}; color:{DARK_BG}; border:none; border-radius:6px; padding:7px 14px; font-size:12px; font-weight:600; }}
QPushButton#action:hover {{ background-color:{ACCENT_HOVER}; }}
QPushButton#action:disabled {{ background-color:rgba(69,71,90,0.25); color:rgba(147,153,178,0.3); border:1px solid rgba(69,71,90,0.2); }}
QPushButton#danger {{ background-color:{DANGER}; color:{DARK_BG}; border:none; border-radius:6px; padding:7px 14px; font-size:12px; font-weight:600; }}
QPushButton#danger:disabled {{ background-color:rgba(69,71,90,0.25); color:rgba(147,153,178,0.3); border:1px solid rgba(69,71,90,0.2); }}

QPushButton#btn_next {{ background-color:{DANGER}; color:white; border:2px solid {WARNING}; border-radius:6px; padding:7px 14px; font-size:12px; font-weight:700; }}
QPushButton#btn_next:hover {{ background-color:#eba0ac; }}
QPushButton#btn_locked {{ background-color:rgba(49,50,68,0.30); color:rgba(147,153,178,0.25); border:1px dashed rgba(69,71,90,0.25); border-radius:6px; padding:7px 14px; font-size:12px; }}

QPushButton#step_active {{ background-color:{WARNING}; color:{DARK_BG}; border:none; border-radius:6px; padding:7px 14px; font-size:12px; font-weight:700; }}

QPushButton#estop {{ background-color:#ee2222; color:white; border:3px solid #ff5555; border-radius:8px; padding:10px 14px; font-size:14px; font-weight:900; letter-spacing:2px; }}
QPushButton#estop:hover {{ background-color:#ff3333; }}

QGroupBox {{ background-color:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; margin-top:16px; padding:18px 10px 10px 10px; font-weight:bold; color:{ACCENT}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:14px; padding:0 6px; background-color:{DARK_BG}; border-radius:3px; font-size:11px; }}

QLabel#title {{ font-size:17px; font-weight:bold; color:{TEXT_PRIMARY}; }}
QLabel#subtitle {{ font-size:11px; font-weight:normal; color:{TEXT_DIM}; }}
QLabel#status {{ font-size:12px; color:{WARNING}; font-weight:bold; }}
QLabel#step_instruction {{ font-size:13px; font-weight:bold; color:{WARNING}; padding:8px 12px; background-color:{CARD_BG}; border:2px solid {WARNING}; border-radius:8px; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background-color:{DARK_BG}; border:1px solid {BORDER}; border-radius:6px; padding:5px 8px; color:{TEXT_PRIMARY}; font-size:12px; font-weight:normal; selection-background-color:{ACCENT}; qproperty-alignment:AlignCenter; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border:1px solid {ACCENT}; background-color:{SIDEBAR_BG}; }}

QProgressBar {{ background-color:{DARK_BG}; border:1px solid {BORDER}; border-radius:6px; text-align:center; color:{TEXT_PRIMARY}; font-size:11px; font-weight:bold; height:18px; }}
QProgressBar::chunk {{ background-color:{ACCENT}; border-radius:4px; }}

QTextEdit {{ background-color:{SIDEBAR_BG}; border:1px solid {BORDER}; border-radius:8px; color:{TEXT_PRIMARY}; font-family:'Consolas','Monaco',monospace; font-size:11px; font-weight:normal; padding:6px; }}

QScrollBar:vertical {{ border:none; background:{SIDEBAR_BG}; width:8px; }}
QScrollBar::handle:vertical {{ background:{BORDER}; min-height:16px; border-radius:4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
QScrollArea {{ border:none; }}

QSlider::groove:horizontal {{ height:5px; background:{BORDER}; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{ACCENT}; width:14px; height:14px; margin:-5px 0; border-radius:7px; }}

QDialog {{ background-color:{DARK_BG}; color:{TEXT_PRIMARY}; }}
QListWidget {{ background-color:{SIDEBAR_BG}; border:1px solid {BORDER}; border-radius:6px; color:{TEXT_PRIMARY}; font-size:12px; padding:2px; }}
QListWidget::item {{ padding:4px 6px; border-radius:3px; }}
QListWidget::item:selected {{ background-color:{ACCENT}; color:{DARK_BG}; }}
"""

# =============================================================================
#  STEP INDICATOR WIDGET
# =============================================================================
class StepIndicator(QWidget):
    """Horizontal bar of numbered badges showing workflow progress."""
    def __init__(self, step_short_labels, parent=None):
        super().__init__(parent)
        self._labels = step_short_labels
        self._current = -1
        self._badges = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        for i, text in enumerate(step_short_labels):
            badge = QLabel(f"  {i+1}. {text}  ")
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedHeight(24)
            badge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._badges.append(badge)
            lay.addWidget(badge)
        self._apply_styles()

    def set_step(self, idx):
        self._current = idx
        self._apply_styles()

    def set_all_done(self):
        self._current = len(self._labels)
        self._apply_styles()

    def reset(self):
        self._current = -1
        self._apply_styles()

    def _apply_styles(self):
        for i, badge in enumerate(self._badges):
            if i < self._current:
                badge.setStyleSheet(
                    f"background:{SUCCESS};color:{DARK_BG};border-radius:4px;"
                    f"font-size:11px;font-weight:600;padding:1px 4px;")
            elif i == self._current:
                badge.setStyleSheet(
                    f"background:{DANGER};color:white;border-radius:4px;"
                    f"font-size:11px;font-weight:700;padding:1px 4px;"
                    f"border:2px solid {WARNING};")
            else:
                badge.setStyleSheet(
                    f"background:{SIDEBAR_BG};color:{TEXT_DIM};border-radius:4px;"
                    f"font-size:11px;padding:1px 4px;"
                    f"border:1px solid {BORDER};")


# =============================================================================
#  DIALOGS
# =============================================================================
class ApproachZDialog(QDialog):
    def __init__(self, parent=None, default_z=200.0):
        super().__init__(parent)
        self.setWindowTitle("Set Approach Z Height"); self.setMinimumWidth(340)
        lay = QVBoxLayout(self); lay.setSpacing(12); lay.setContentsMargins(20,20,20,20)
        info = QLabel("Z height for approach moves above pick/place."); info.setObjectName("subtitle"); info.setWordWrap(True); lay.addWidget(info)
        row = QHBoxLayout(); row.addWidget(QLabel("Approach Z:"))
        self.spin = QDoubleSpinBox(); self.spin.setRange(50,500); self.spin.setDecimals(1); self.spin.setValue(default_z); self.spin.setSuffix(" mm")
        row.addWidget(self.spin); lay.addLayout(row)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject); lay.addWidget(btns)
    def get_z(self): return self.spin.value()


# =============================================================================
#  PALLET CONFIG DIALOG — forces operator to review/enter all measurements
# =============================================================================
class PalletConfigDialog(QDialog):
    """Dialog that forces operator to enter all pallet measurements + approach Z."""
    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Pallet Configuration"); self.setMinimumWidth(420)
        d = defaults or {}
        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(20,20,20,20)

        warn = QLabel("⚠ Review ALL measurements before starting.\nIncorrect values will cause collisions or misalignment.")
        warn.setObjectName("status"); warn.setWordWrap(True); lay.addWidget(warn)

        # Approach Z
        az_row = QHBoxLayout()
        az_row.addWidget(QLabel("Approach Z:"))
        self.spin_az = QDoubleSpinBox(); self.spin_az.setRange(50,500); self.spin_az.setDecimals(1)
        self.spin_az.setValue(d.get("approach_z", 200.0)); self.spin_az.setSuffix(" mm")
        az_row.addWidget(self.spin_az); lay.addLayout(az_row)

        # Object dimensions
        og = QGroupBox("Object Dimensions"); ogl = QGridLayout(og); ogl.setContentsMargins(10,20,10,10)
        self.spin_w = QDoubleSpinBox(); self.spin_w.setRange(1,500); self.spin_w.setDecimals(1)
        self.spin_w.setValue(d.get("obj_w",50)); self.spin_w.setSuffix(" mm")
        self.spin_d = QDoubleSpinBox(); self.spin_d.setRange(1,500); self.spin_d.setDecimals(1)
        self.spin_d.setValue(d.get("obj_d",50)); self.spin_d.setSuffix(" mm")
        self.spin_h = QDoubleSpinBox(); self.spin_h.setRange(1,500); self.spin_h.setDecimals(1)
        self.spin_h.setValue(d.get("obj_h",20)); self.spin_h.setSuffix(" mm")
        ogl.addWidget(QLabel("Width  (X):"),0,0);  ogl.addWidget(self.spin_w,0,1)
        ogl.addWidget(QLabel("Length (Y):"),1,0);   ogl.addWidget(self.spin_d,1,1)
        ogl.addWidget(QLabel("Height (Z):"),2,0);   ogl.addWidget(self.spin_h,2,1)
        lay.addWidget(og)

        # Gaps
        gg = QGroupBox("Gaps Between Pieces"); ggl = QGridLayout(gg); ggl.setContentsMargins(10,20,10,10)
        self.spin_gx = QDoubleSpinBox(); self.spin_gx.setRange(0,100); self.spin_gx.setDecimals(1)
        self.spin_gx.setValue(d.get("gap_x",5)); self.spin_gx.setSuffix(" mm")
        self.spin_gy = QDoubleSpinBox(); self.spin_gy.setRange(0,100); self.spin_gy.setDecimals(1)
        self.spin_gy.setValue(d.get("gap_y",5)); self.spin_gy.setSuffix(" mm")
        ggl.addWidget(QLabel("Gap X:"),0,0); ggl.addWidget(self.spin_gx,0,1)
        ggl.addWidget(QLabel("Gap Y:"),1,0); ggl.addWidget(self.spin_gy,1,1)
        lay.addWidget(gg)

        # Grid
        sg = QGroupBox("Grid Size"); sgl = QGridLayout(sg); sgl.setContentsMargins(10,20,10,10)
        self.spin_cols = QSpinBox(); self.spin_cols.setRange(1,20); self.spin_cols.setValue(d.get("cols",3))
        self.spin_rows = QSpinBox(); self.spin_rows.setRange(1,20); self.spin_rows.setValue(d.get("rows",3))
        self.spin_layers = QSpinBox(); self.spin_layers.setRange(1,10); self.spin_layers.setValue(d.get("layers",1))
        sgl.addWidget(QLabel("Columns:"),0,0); sgl.addWidget(self.spin_cols,0,1)
        sgl.addWidget(QLabel("Rows:"),1,0);    sgl.addWidget(self.spin_rows,1,1)
        sgl.addWidget(QLabel("Layers:"),2,0);  sgl.addWidget(self.spin_layers,2,1)
        lay.addWidget(sg)

        # Preview
        self.preview_lbl = QLabel(); self.preview_lbl.setObjectName("subtitle"); self.preview_lbl.setWordWrap(True)
        lay.addWidget(self.preview_lbl)
        for w in (self.spin_w, self.spin_d, self.spin_h, self.spin_gx, self.spin_gy,
                  self.spin_cols, self.spin_rows, self.spin_layers):
            w.valueChanged.connect(self._update_preview)
        self._update_preview()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject); lay.addWidget(btns)

    def _update_preview(self):
        px = self.spin_w.value() + self.spin_gx.value()
        py = self.spin_d.value() + self.spin_gy.value()
        total = self.spin_cols.value() * self.spin_rows.value() * self.spin_layers.value()
        tw = px * self.spin_cols.value() - self.spin_gx.value()
        td = py * self.spin_rows.value() - self.spin_gy.value()
        self.preview_lbl.setText(
            f"Pitch X={px:.1f}mm  Pitch Y={py:.1f}mm  Layer ΔZ={self.spin_h.value():.1f}mm\n"
            f"Total cells: {total}  |  Footprint: {tw:.0f} × {td:.0f} mm\n"
            f"Grid grows: Cols → +X  |  Rows → −Y  |  Layers → +Z")

    def get_config(self):
        return {
            "approach_z": self.spin_az.value(),
            "obj_w": self.spin_w.value(), "obj_d": self.spin_d.value(), "obj_h": self.spin_h.value(),
            "gap_x": self.spin_gx.value(), "gap_y": self.spin_gy.value(),
            "cols": self.spin_cols.value(), "rows": self.spin_rows.value(), "layers": self.spin_layers.value(),
        }


# =============================================================================
#  IMAGE / BOUNDING BOX WIDGETS
# =============================================================================
class BoundingBoxWidget(QLabel):
    def __init__(self):
        super().__init__(); self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color:#11111b;border:2px solid {BORDER};border-radius:10px;"); self.setMinimumHeight(180)
        self.drawing=False; self.start_point=QPoint(); self.end_point=QPoint()
        self.current_pixmap=None; self.original_image_cv=None; self.boxes=[]
    def _display_rect(self):
        if not self.pixmap(): return QRect()
        pw,ph=self.pixmap().width(),self.pixmap().height(); lw,lh=self.width(),self.height()
        return QRect((lw-pw)//2,(lh-ph)//2,pw,ph)
    def _widget_to_image(self, pt):
        if self.original_image_cv is None or not self.pixmap(): return None
        dr=self._display_rect()
        if dr.width()==0 or dr.height()==0: return None
        rx=max(0.0,min(1.0,(pt.x()-dr.x())/dr.width())); ry=max(0.0,min(1.0,(pt.y()-dr.y())/dr.height()))
        ih,iw=self.original_image_cv.shape[:2]; return rx*iw, ry*ih
    def set_image(self, cv_img):
        self.original_image_cv=cv_img.copy(); rgb=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB); h,w,ch=rgb.shape
        self.current_pixmap=QPixmap.fromImage(QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888).copy())
        self._refresh_display(); self.boxes.clear(); self.start_point=QPoint(); self.end_point=QPoint()
    def _refresh_display(self):
        if self.current_pixmap: self.setPixmap(self.current_pixmap.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def resizeEvent(self, e): super().resizeEvent(e); self._refresh_display()
    def mousePressEvent(self, e):
        if e.button()==Qt.LeftButton and self.current_pixmap: self.drawing=True; self.start_point=e.pos(); self.end_point=self.start_point
    def mouseMoveEvent(self, e):
        if self.drawing: self.end_point=e.pos(); self.update()
    def mouseReleaseEvent(self, e):
        if e.button()==Qt.LeftButton and self.drawing:
            self.drawing=False; self.end_point=e.pos()
            tl=self._widget_to_image(self.start_point); br=self._widget_to_image(self.end_point)
            if tl is None or br is None: return
            x1,y1=min(tl[0],br[0]),min(tl[1],br[1]); x2,y2=max(tl[0],br[0]),max(tl[1],br[1])
            if (x2-x1)<4 or (y2-y1)<4: return
            ih,iw=self.original_image_cv.shape[:2]
            label,ok=QInputDialog.getText(self,"Label Part","Enter part name:")
            if ok and label.strip():
                cx=((x1+x2)/2)/iw; cy=((y1+y2)/2)/ih; bw=(x2-x1)/iw; bh=(y2-y1)/ih
                self.boxes.append({"rect":QRect(self.start_point,self.end_point).normalized(),"yolo":[cx,cy,bw,bh],"label":label.strip().lower()})
            self.update()
    def paintEvent(self, e):
        super().paintEvent(e); painter=QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        for box in self.boxes:
            painter.setPen(QPen(QColor(SUCCESS),2)); painter.drawRect(box["rect"])
            painter.setPen(QPen(QColor(255,255,255),1)); painter.drawText(box["rect"].topLeft()+QPoint(5,15),box["label"])
        if self.drawing and not self.start_point.isNull():
            painter.setPen(QPen(QColor(DANGER),2,Qt.DashLine)); painter.drawRect(QRect(self.start_point,self.end_point).normalized())
        painter.end()
    def clear_boxes(self): self.boxes.clear(); self.update()


class ClickableVideoWidget(QLabel):
    pixel_clicked=Signal(float,float)
    def __init__(self):
        super().__init__(); self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color:#11111b;border:2px solid {BORDER};border-radius:10px;"); self.original_shape=None
    def set_image(self, cv_img):
        self.original_shape=cv_img.shape[:2]; rgb=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB); h,w,ch=rgb.shape
        self.setPixmap(QPixmap.fromImage(QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888).copy()).scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def mousePressEvent(self, e):
        if not self.pixmap() or not self.original_shape: return
        pw,ph=self.pixmap().width(),self.pixmap().height(); lw,lh=self.width(),self.height()
        xo,yo=(lw-pw)//2,(lh-ph)//2; rx=(e.pos().x()-xo)/pw; ry=(e.pos().y()-yo)/ph
        if 0<=rx<=1 and 0<=ry<=1:
            ih,iw=self.original_shape; self.pixel_clicked.emit(rx*iw,ry*ih)


# =============================================================================
#  YOLO TRAINER
# =============================================================================
class YoloTrainerWorker(QThread):
    status_update=Signal(str); progress_update=Signal(int); training_finished=Signal(str)
    def __init__(self, dataset, project_path, epochs=30, augment_factor=25, imgsz=640):
        super().__init__(); self.dataset=dataset; self.dataset_dir=os.path.join(project_path,"sme_dataset")
        self.project_path=project_path; self.epochs=epochs; self.augment_factor=augment_factor; self.imgsz=imgsz
        self.unique_classes=sorted(set(box["label"] for d in self.dataset for box in d["boxes"]))
        self.class_to_id={n:i for i,n in enumerate(self.unique_classes)}
    def run(self):
        try: self._prepare_dataset(); self.training_finished.emit(self._train())
        except Exception as e: self.status_update.emit(f"ERROR: {e}")
    def _prepare_dataset(self):
        self.status_update.emit("Preparing...")
        if os.path.exists(self.dataset_dir): shutil.rmtree(self.dataset_dir)
        for s in ("train","val"):
            os.makedirs(f"{self.dataset_dir}/images/{s}",exist_ok=True); os.makedirs(f"{self.dataset_dir}/labels/{s}",exist_ok=True)
        transform=A.Compose([A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.2),A.RandomBrightnessContrast(0.3,0.3,p=0.6),
            A.HueSaturationValue(15,30,20,p=0.5),A.Rotate(limit=30,border_mode=cv2.BORDER_REFLECT_101,p=0.6),
            A.GaussianBlur((3,5),p=0.3),A.GaussNoise(p=0.2),A.RandomScale(0.15,p=0.4),
            A.PadIfNeeded(self.imgsz,self.imgsz,border_mode=cv2.BORDER_REFLECT_101,p=0.0)],
            bbox_params=A.BboxParams(format="yolo",label_fields=["class_labels"],min_visibility=0.3,min_area=256))
        self.status_update.emit("Augmenting..."); ic=0
        for idx,data in enumerate(self.dataset):
            img=data["image"]; bbs=[b["yolo"] for b in data["boxes"]]; lbls=[b["label"] for b in data["boxes"]]
            self._save(img,bbs,lbls,ic,"train"); ic+=1
            for ai in range(self.augment_factor):
                try:
                    t=transform(image=img,bboxes=bbs,class_labels=lbls)
                    if not t["bboxes"]: continue
                    self._save(t["image"],t["bboxes"],t["class_labels"],ic,"val" if ai%5==0 else "train"); ic+=1
                except: pass
            self.progress_update.emit(int(((idx+1)/len(self.dataset))*30))
        self.status_update.emit(f"Dataset: {ic} images.")
    def _save(self, img, bbs, lbls, i, split):
        cv2.imwrite(f"{self.dataset_dir}/images/{split}/img_{i:05d}.jpg",img)
        with open(f"{self.dataset_dir}/labels/{split}/img_{i:05d}.txt","w") as f:
            for bb,l in zip(bbs,lbls): f.write(f"{self.class_to_id[l]} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
    def _train(self):
        self.status_update.emit(f"Training {self.epochs} epochs...")
        yp=os.path.join(self.dataset_dir,"dataset.yaml")
        with open(yp,"w") as f:
            yaml.dump({"path":os.path.abspath(self.dataset_dir),"train":"images/train","val":"images/val",
                        "names":{i:n for n,i in self.class_to_id.items()}},f,default_flow_style=False)
        YOLO("yolov8n.pt").train(data=yp,epochs=self.epochs,imgsz=self.imgsz,batch=-1,device="cpu",patience=10,
            verbose=False,project=self.project_path,name="sme_model",exist_ok=True,lr0=0.01,lrf=0.01,warmup_epochs=3,
            augment=True,hsv_h=0.015,hsv_s=0.5,hsv_v=0.3,flipud=0.1,fliplr=0.5,mosaic=0.8,mixup=0.1)
        self.progress_update.emit(95)
        bp=os.path.join(self.project_path,"sme_model","weights","best.pt")
        if not os.path.isfile(bp):
            lp=os.path.join(self.project_path,"sme_model","weights","last.pt")
            if os.path.isfile(lp): bp=lp
        self.status_update.emit("Done"); self.progress_update.emit(100); return bp


# =============================================================================
#  DATA CLASSES
# =============================================================================
class Waypoint:
    def __init__(self, label="",x=0,y=0,z=200,roll=-179.8,pitch=0.0,yaw=0.0,gripper="open",source="manual",speed=100,move_type="moveL"):
        self.label=label;self.x=x;self.y=y;self.z=z;self.roll=roll;self.pitch=pitch;self.yaw=yaw
        self.gripper=gripper;self.source=source;self.speed=speed;self.move_type=move_type
    def as_dict(self): return vars(self)
    @staticmethod
    def from_dict(d): return Waypoint(**{k:v for k,v in d.items() if k in Waypoint.__init__.__code__.co_varnames})

class Routine:
    def __init__(self, name="",routine_type="pick_place",waypoints=None,pallet_config=None,approach_z=200.0):
        self.name=name; self.routine_type=routine_type; self.waypoints=waypoints or []; self.pallet_config=pallet_config; self.approach_z=approach_z
    def as_dict(self):
        return {"name":self.name,"routine_type":self.routine_type,"waypoints":self.waypoints,"pallet_config":self.pallet_config,"approach_z":self.approach_z}
    @staticmethod
    def from_dict(d):
        return Routine(name=d.get("name",""),routine_type=d.get("routine_type","pick_place"),
                       waypoints=d.get("waypoints",[]),pallet_config=d.get("pallet_config"),approach_z=d.get("approach_z",200.0))


# =============================================================================
#  COMPACT JOG PANEL
# =============================================================================
class AdvancedJogPanel(QGroupBox):
    jog_step=Signal(str,float)
    jog_cont_start=Signal(str,int)
    jog_cont_stop=Signal()
    def __init__(self, title="Jog Robot"):
        super().__init__(title)
        main=QVBoxLayout(self); main.setSpacing(4); main.setContentsMargins(8,18,8,6)
        top=QHBoxLayout(); top.setSpacing(4)
        self.mode_combo=QComboBox(); self.mode_combo.addItems(["Step","Continuous"]); self.mode_combo.setFixedWidth(90)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed); top.addWidget(self.mode_combo)
        self.step_spin=QDoubleSpinBox(); self.step_spin.setRange(0.1,50.0); self.step_spin.setValue(5.0); self.step_spin.setSuffix(" mm"); self.step_spin.setFixedWidth(80)
        top.addWidget(self.step_spin)
        self.vel_slider=QSlider(Qt.Horizontal); self.vel_slider.setRange(5,200); self.vel_slider.setValue(50)
        self.vel_value_lbl=QLabel("50"); self.vel_value_lbl.setFixedWidth(24)
        self.vel_slider.valueChanged.connect(lambda v: self.vel_value_lbl.setText(str(v)))
        top.addWidget(self.vel_slider,1); top.addWidget(self.vel_value_lbl)
        main.addLayout(top)
        for axes_row in [("X","Y"),("Z",)]:
            row=QHBoxLayout(); row.setSpacing(4)
            for axis in axes_row:
                minus=QPushButton(f"-{axis}"); minus.setObjectName("action"); minus.setFocusPolicy(Qt.NoFocus); minus.setFixedHeight(30)
                plus=QPushButton(f"+{axis}"); plus.setObjectName("action"); plus.setFocusPolicy(Qt.NoFocus); plus.setFixedHeight(30)
                minus.clicked.connect(lambda c=False,a=axis: self._on_step_click(a,-1))
                plus.clicked.connect(lambda c=False,a=axis: self._on_step_click(a,+1))
                minus.pressed.connect(lambda a=axis: self._on_cont_press(a,-1))
                minus.released.connect(self._on_cont_release)
                plus.pressed.connect(lambda a=axis: self._on_cont_press(a,+1))
                plus.released.connect(self._on_cont_release)
                row.addWidget(minus,1); row.addWidget(plus,1)
            main.addLayout(row)
        self._on_mode_changed("Step")
    def _on_mode_changed(self, m):
        s=m=="Step"
        self.step_spin.setVisible(s); self.vel_slider.setVisible(not s); self.vel_value_lbl.setVisible(not s)
    def _on_step_click(self, axis, sign):
        if self.mode_combo.currentText()=="Step": self.jog_step.emit(axis,self.step_spin.value()*sign)
    def _on_cont_press(self, axis, sign):
        if self.mode_combo.currentText()=="Continuous": self.jog_cont_start.emit(axis,sign)
    def _on_cont_release(self):
        if self.mode_combo.currentText()=="Continuous": self.jog_cont_stop.emit()


# =============================================================================
#  GRIPPER PANELS
# =============================================================================
class GripperPanel(QGroupBox):
    gripper_requested=Signal(str)
    def __init__(self, title="Gripper"):
        super().__init__(title)
        lay=QHBoxLayout(self); lay.setContentsMargins(8,18,8,6); lay.setSpacing(6)
        o=QPushButton("Open"); o.setObjectName("action"); o.setFocusPolicy(Qt.NoFocus); o.setFixedHeight(28); o.clicked.connect(lambda: self.gripper_requested.emit("open"))
        c=QPushButton("Close"); c.setObjectName("danger"); c.setFocusPolicy(Qt.NoFocus); c.setFixedHeight(28); c.clicked.connect(lambda: self.gripper_requested.emit("close"))
        lay.addWidget(o,1); lay.addWidget(c,1)

class GripperControlPanel(QGroupBox):
    action_requested=Signal(str)
    def __init__(self, title="Gripper Control"):
        super().__init__(title)
        lay=QHBoxLayout(self); lay.setContentsMargins(8,18,8,6); lay.setSpacing(6)
        for label,action,obj in [("Enable","enable","action"),("Open","open","action"),("Close","close","danger"),("OFF","off","danger")]:
            b=QPushButton(label); b.setObjectName(obj); b.setFocusPolicy(Qt.NoFocus); b.setFixedHeight(28)
            b.clicked.connect(lambda c=False,a=action: self.action_requested.emit(a)); lay.addWidget(b,1)


# =============================================================================
#  MAIN APPLICATION
# =============================================================================
class RobotTeachApp(QMainWindow):
    APPROACH_Z_DEFAULT=200.0; MOVE_SPEED=100; MOVE_SPEED_SLOW=30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SME Robot Programmer"); self.setMinimumSize(1280,720)
        self.project_dir=os.path.abspath("SME_Vision_Project"); os.makedirs(self.project_dir,exist_ok=True)
        self.routines_dir=os.path.join(self.project_dir,"routines"); os.makedirs(self.routines_dir,exist_ok=True)
        self.home_position_path=os.path.join(self.project_dir,"home_position.json")
        self.model_path=os.path.join(self.project_dir,"sme_model","weights","best.pt")
        self.homography_path=os.path.join(self.project_dir,"homografia_H.npy")
        self.vision_dataset=[]; self.waypoints=[]; self.current_model=None; self.cam_index_top=0
        self.robot=None; self.H_matrix=None; self.H_inv=None; self.calib_image_pts=[]
        self.calib_timer=QTimer(); self.calib_timer.timeout.connect(self._calib_update_frame); self.calib_cap=None
        if os.path.exists(self.homography_path):
            self.H_matrix=np.load(self.homography_path); self.H_inv=np.linalg.inv(self.H_matrix)
        self._guide_step=0; self._guide_waypoints=[]; self._guide_detections=[]; self._guide_current_frame=None
        self._guide_approach_z=self.APPROACH_Z_DEFAULT
        self._pal_guide_step=0; self._pal_waypoints=[]; self._pal_approach_z=self.APPROACH_Z_DEFAULT
        self._motion_busy=False; self._calib_teach_mode=False
        # --- E-STOP flag: checked before every robot motion in production ---
        self._estop_flag=False
        self.jog_timer=QTimer(); self.jog_timer.timeout.connect(self._update_pose_display)
        self.prod_cam_timer=QTimer(); self.prod_cam_timer.timeout.connect(self._prod_update_cam); self.prod_cap=None
        self.setStyleSheet(GLOBAL_STYLE); self._build_ui()

    # -----------------------------------------------------------------
    #  BUTTON VISUAL HELPERS
    # -----------------------------------------------------------------
    def _restyle_btn(self, btn, obj_name):
        btn.setObjectName(obj_name)
        btn.style().unpolish(btn); btn.style().polish(btn); btn.update()

    def _pp_style_buttons(self):
        all_btns = [self.btn_pp_start, self.btn_pp_capture, self.btn_pp_recapture,
                    self.btn_pp_select, self.btn_pp_save_wp, self.btn_pp_add_cycle,
                    self.btn_pp_finish, self.btn_pp_cancel]
        next_set = set()
        if self._guide_step >= 5:
            next_set = {self.btn_pp_add_cycle, self.btn_pp_finish}
        elif self.btn_pp_save_wp.isEnabled():
            next_set = {self.btn_pp_save_wp}
        elif self.btn_pp_select.isEnabled():
            next_set = {self.btn_pp_select}
        elif self.btn_pp_capture.isEnabled():
            next_set = {self.btn_pp_capture}
        elif self.btn_pp_recapture.isEnabled():
            next_set = {self.btn_pp_recapture}
        elif self.btn_pp_start.isEnabled():
            next_set = {self.btn_pp_start}
        for btn in all_btns:
            if btn in next_set and btn.isEnabled():
                self._restyle_btn(btn, "btn_next")
            elif btn.isEnabled():
                orig = "danger" if btn is self.btn_pp_cancel else "action"
                self._restyle_btn(btn, orig)
            else:
                self._restyle_btn(btn, "btn_locked")
        si = min(self._guide_step, len(self.PP_STEPS))
        if si >= len(self.PP_STEPS):
            self.pp_step_ind.set_all_done()
        else:
            self.pp_step_ind.set_step(si)

    def _pal_style_buttons(self):
        all_btns = [self.btn_pal_start, self.btn_pal_capture, self.btn_pal_recapture,
                    self.btn_pal_select, self.btn_pal_save_wp, self.btn_pal_finish,
                    self.btn_pal_cancel]
        next_set = set()
        if self._pal_guide_step >= 5:
            next_set = {self.btn_pal_finish}
        elif self.btn_pal_save_wp.isEnabled():
            next_set = {self.btn_pal_save_wp}
        elif self.btn_pal_select.isEnabled():
            next_set = {self.btn_pal_select}
        elif self.btn_pal_capture.isEnabled():
            next_set = {self.btn_pal_capture}
        elif self.btn_pal_recapture.isEnabled():
            next_set = {self.btn_pal_recapture}
        elif self.btn_pal_start.isEnabled():
            next_set = {self.btn_pal_start}
        for btn in all_btns:
            if btn in next_set and btn.isEnabled():
                self._restyle_btn(btn, "btn_next")
            elif btn.isEnabled():
                orig = "danger" if btn is self.btn_pal_cancel else "action"
                self._restyle_btn(btn, orig)
            else:
                self._restyle_btn(btn, "btn_locked")
        si = min(self._pal_guide_step, len(self.PAL_STEPS))
        if si >= len(self.PAL_STEPS):
            self.pal_step_ind.set_all_done()
        else:
            self.pal_step_ind.set_step(si)

    # -----------------------------------------------------------------
    #  BUILD UI
    # -----------------------------------------------------------------
    def _build_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        sidebar=QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(190)
        sb=QVBoxLayout(sidebar); sb.setContentsMargins(0,16,0,16); sb.setSpacing(2)
        logo=QLabel("  SME Robot"); logo.setObjectName("title")
        logo.setStyleSheet(f"color:{ACCENT};padding:0 8px 12px 8px;font-size:16px;"); sb.addWidget(logo)
        self.nav_buttons=[]
        for text,idx in [("Dashboard",0),("Free Jog",1),("Calibration",2),("Vision Training",3),("Pick & Place",4),("Palletizing",5),("Production",6)]:
            btn=QPushButton(f"  {text}"); btn.setObjectName("nav"); btn.setCheckable(True)
            btn.clicked.connect(lambda _,i=idx: self._nav(i)); sb.addWidget(btn); self.nav_buttons.append(btn)
        sb.addStretch()
        cam_box=QGroupBox("Camera"); cl=QHBoxLayout(cam_box); cl.setContentsMargins(8,18,8,6)
        self.spin_cam_top=QSpinBox(); self.spin_cam_top.setRange(0,10); self.spin_cam_top.setValue(0)
        cl.addWidget(QLabel("Index:")); cl.addWidget(self.spin_cam_top)
        self.spin_cam_top.valueChanged.connect(lambda v: setattr(self,"cam_index_top",v)); sb.addWidget(cam_box)
        self.stack=QStackedWidget()
        self.stack.addWidget(self._page_dashboard())
        self.stack.addWidget(self._page_free_jog())
        self.stack.addWidget(self._page_calibration())
        self.stack.addWidget(self._page_vision())
        self.stack.addWidget(self._page_pick_place())
        self.stack.addWidget(self._page_palletize())
        self.stack.addWidget(self._page_production())
        root.addWidget(sidebar); root.addWidget(self.stack,stretch=1); self._nav(0)

    def _nav(self, index):
        self.stack.setCurrentIndex(index)
        for i,btn in enumerate(self.nav_buttons): btn.setChecked(i==index)
        if index!=2 and self.calib_timer.isActive():
            self.calib_timer.stop()
            if self.calib_cap: self.calib_cap.release(); self.calib_cap=None
        if index!=2:
            self.btn_calib_feed.setText("Live Feed"); self._calib_teach_mode=False
            self.btn_calib_teach.setChecked(False); self.btn_calib_teach.setText("Teach Reference Point")
        if index!=6 and self.prod_cam_timer.isActive():
            self.prod_cam_timer.stop()
            if self.prod_cap: self.prod_cap.release(); self.prod_cap=None
            self.btn_prod_cam.setText("Live Camera")
        if index==1: self._update_robot_status(); self.jog_timer.start(500)
        else:
            self.jog_timer.stop()
            # Exit manual mode if leaving Free Jog page while it's active
            if hasattr(self, 'btn_fj_manual') and self.btn_fj_manual.isChecked():
                if self.robot:
                    try: self.robot.set_mode(0); self.robot.set_state(0)
                    except: pass
                self.btn_fj_manual.setChecked(False)
                self.btn_fj_manual.setText("Manual Move (Freedrive)")
                self.btn_fj_manual.setStyleSheet("")

    # ===== PAGE 0: DASHBOARD =====
    def _page_dashboard(self):
        page=QWidget(); lay=QVBoxLayout(page); lay.setContentsMargins(24,24,24,24); lay.setSpacing(12)
        lay.addWidget(self._mk_title("System Dashboard"))
        lay.addWidget(self._mk_sub("Low-code robot programming — xArm Lite6 + USB camera."))
        rb=QGroupBox("xArm Connection"); rl=QHBoxLayout(rb); rl.setContentsMargins(10,20,10,10)
        self.ip_input=QLineEdit(); self.ip_input.setText("192.168.1.197")
        self.btn_connect=QPushButton("Connect"); self.btn_connect.setObjectName("action"); self.btn_connect.clicked.connect(self._connect_robot)
        rl.addWidget(QLabel("IP:")); rl.addWidget(self.ip_input); rl.addWidget(self.btn_connect); rl.addStretch(); lay.addWidget(rb)
        g=QGridLayout(); g.setSpacing(12)
        self.dash_model_lbl=QLabel(f"Model: {'Yes' if os.path.isfile(self.model_path) else 'No'}")
        self.dash_dataset_lbl=QLabel("Dataset: 0"); self.dash_wp_lbl=QLabel("Waypoints: 0")
        self.dash_routines_lbl=QLabel(f"Routines: {len(self._list_routines())}")
        for i,(t,w) in enumerate([("Vision",self.dash_model_lbl),("Data",self.dash_dataset_lbl),("Points",self.dash_wp_lbl),("Routines",self.dash_routines_lbl)]):
            c=QGroupBox(t); vl=QVBoxLayout(c); vl.setContentsMargins(10,20,10,10); vl.addWidget(w); g.addWidget(c,0,i)
        lay.addLayout(g)
        ib=QGroupBox("Quick Start"); il=QVBoxLayout(ib); il.setContentsMargins(10,20,10,10)
        il.addWidget(QLabel("1. Free Jog → 2. Calibration → 3. Vision → 4. Pick&Place / Palletizing → 5. Production"))
        lay.addWidget(ib); lay.addStretch(); return page

    def _connect_robot(self):
        if not XARM_AVAILABLE: QMessageBox.warning(self,"","xarm SDK not found."); return
        ip=self.ip_input.text().strip()
        try:
            self.robot=XArmAPI(ip); self.robot.clean_warn(); self.robot.clean_error()
            self.robot.motion_enable(True); self.robot.set_mode(0); self.robot.set_state(0)
            self.robot.set_tgpio_modbus_timeout(20)
            self.btn_connect.setText("Connected")
            self.btn_connect.setStyleSheet(f"background-color:{SUCCESS};color:{DARK_BG};border:none;border-radius:6px;padding:7px 14px;font-size:12px;font-weight:600;")
            self._update_robot_status()
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    # =========================================================================
    #  PAGE 1: FREE JOG  (with rotation jog)
    # =========================================================================
    def _page_free_jog(self):
        page=QWidget(); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        cw=QWidget(); lay=QVBoxLayout(cw); lay.setContentsMargins(16,16,16,16); lay.setSpacing(8)
        lay.addWidget(self._mk_title("Free Jog & Manual Control"))
        sr=QHBoxLayout()
        self.fj_conn_status=QLabel("Robot: Disconnected"); self.fj_conn_status.setObjectName("status")
        self.fj_enable_status=QLabel("Enabled: No"); self.fj_enable_status.setObjectName("status")
        sr.addWidget(self.fj_conn_status); sr.addWidget(self.fj_enable_status); sr.addStretch(); lay.addLayout(sr)
        pb=QGroupBox("Current TCP Pose"); pl=QGridLayout(pb); pl.setContentsMargins(8,18,8,6)
        self.fj_pose_x=QLabel("X: ---"); self.fj_pose_y=QLabel("Y: ---"); self.fj_pose_z=QLabel("Z: ---")
        self.fj_pose_roll=QLabel("Roll: ---"); self.fj_pose_pitch=QLabel("Pitch: ---"); self.fj_pose_yaw=QLabel("Yaw: ---")
        pl.addWidget(self.fj_pose_x,0,0); pl.addWidget(self.fj_pose_y,0,1); pl.addWidget(self.fj_pose_z,0,2)
        pl.addWidget(self.fj_pose_roll,1,0); pl.addWidget(self.fj_pose_pitch,1,1); pl.addWidget(self.fj_pose_yaw,1,2)
        lay.addWidget(pb)

        # --- Translation jog ---
        self.fj_jog=AdvancedJogPanel("Jog Translation (XYZ)")
        self.fj_jog.jog_step.connect(self._fj_jog_step)
        self.fj_jog.jog_cont_start.connect(self._fj_jog_cont_start)
        self.fj_jog.jog_cont_stop.connect(self._fj_jog_cont_stop)
        lay.addWidget(self.fj_jog)

        # --- Rotation jog (Roll / Pitch / Yaw) — Step + Continuous ---
        rg=QGroupBox("Jog Rotation (RPY)")
        rm=QVBoxLayout(rg); rm.setSpacing(4); rm.setContentsMargins(8,18,8,6)
        rtop=QHBoxLayout(); rtop.setSpacing(4)
        self.fj_rot_mode=QComboBox(); self.fj_rot_mode.addItems(["Step","Continuous"]); self.fj_rot_mode.setFixedWidth(90)
        self.fj_rot_mode.currentTextChanged.connect(self._fj_rot_mode_changed); rtop.addWidget(self.fj_rot_mode)
        self.fj_rot_step=QDoubleSpinBox(); self.fj_rot_step.setRange(0.1,30.0); self.fj_rot_step.setValue(2.0); self.fj_rot_step.setSuffix(" °"); self.fj_rot_step.setFixedWidth(80)
        rtop.addWidget(self.fj_rot_step)
        self.fj_rot_vel=QSlider(Qt.Horizontal); self.fj_rot_vel.setRange(5,100); self.fj_rot_vel.setValue(20)
        self.fj_rot_vel_lbl=QLabel("20"); self.fj_rot_vel_lbl.setFixedWidth(24)
        self.fj_rot_vel.valueChanged.connect(lambda v: self.fj_rot_vel_lbl.setText(str(v)))
        rtop.addWidget(self.fj_rot_vel,1); rtop.addWidget(self.fj_rot_vel_lbl)
        rm.addLayout(rtop)
        for axes_row in [("Roll","Pitch"),("Yaw",)]:
            row=QHBoxLayout(); row.setSpacing(4)
            for axis in axes_row:
                minus=QPushButton(f"-{axis}"); minus.setObjectName("action"); minus.setFocusPolicy(Qt.NoFocus); minus.setFixedHeight(30)
                plus=QPushButton(f"+{axis}"); plus.setObjectName("action"); plus.setFocusPolicy(Qt.NoFocus); plus.setFixedHeight(30)
                minus.clicked.connect(lambda c=False,a=axis: self._fj_jog_rot(a,-1))
                plus.clicked.connect(lambda c=False,a=axis: self._fj_jog_rot(a,+1))
                minus.pressed.connect(lambda a=axis: self._fj_jog_rot_cont_start(a,-1))
                minus.released.connect(self._fj_jog_rot_cont_stop)
                plus.pressed.connect(lambda a=axis: self._fj_jog_rot_cont_start(a,+1))
                plus.released.connect(self._fj_jog_rot_cont_stop)
                row.addWidget(minus,1); row.addWidget(plus,1)
            rm.addLayout(row)
        self._fj_rot_mode_changed("Step")
        lay.addWidget(rg)

        self.fj_gripper=GripperControlPanel("Gripper Control (TO 0 = power)")
        self.fj_gripper.action_requested.connect(self._gripper_full_action); lay.addWidget(self.fj_gripper)
        bg=QGridLayout(); bg.setSpacing(6)
        self.btn_fj_enable=QPushButton("Enable Robot"); self.btn_fj_enable.setObjectName("action"); self.btn_fj_enable.clicked.connect(self._fj_enable)
        self.btn_fj_disable=QPushButton("Disable Robot"); self.btn_fj_disable.setObjectName("action"); self.btn_fj_disable.clicked.connect(self._fj_disable)
        self.btn_fj_set_home=QPushButton("Set Current as Home"); self.btn_fj_set_home.setObjectName("action"); self.btn_fj_set_home.clicked.connect(self._fj_set_home)
        self.btn_fj_goto_home=QPushButton("Move to Home"); self.btn_fj_goto_home.setObjectName("action"); self.btn_fj_goto_home.clicked.connect(self._fj_goto_home)
        self.btn_fj_manual=QPushButton("Manual Move (Freedrive)"); self.btn_fj_manual.setObjectName("action"); self.btn_fj_manual.setCheckable(True); self.btn_fj_manual.clicked.connect(self._fj_manual_toggle)
        self.btn_fj_estop=QPushButton("⛔ EMERGENCY STOP"); self.btn_fj_estop.setObjectName("estop"); self.btn_fj_estop.clicked.connect(self._global_estop)
        bg.addWidget(self.btn_fj_enable,0,0); bg.addWidget(self.btn_fj_disable,0,1)
        bg.addWidget(self.btn_fj_set_home,1,0); bg.addWidget(self.btn_fj_goto_home,1,1)
        bg.addWidget(self.btn_fj_manual,2,0,1,2)
        bg.addWidget(self.btn_fj_estop,3,0,1,2)
        lay.addLayout(bg); lay.addStretch()
        scroll.setWidget(cw); ml=QVBoxLayout(page); ml.setContentsMargins(0,0,0,0); ml.addWidget(scroll); return page

    def _update_robot_status(self):
        if self.robot:
            self.fj_conn_status.setText("Robot: Connected")
            try:
                st=self.robot.get_state()[1]; self.fj_enable_status.setText(f"Enabled: {'Yes' if st in (0,1,2) else 'No'}")
            except: pass
        else: self.fj_conn_status.setText("Robot: Disconnected"); self.fj_enable_status.setText("Enabled: No")
    def _update_pose_display(self):
        if not self.robot: return
        try:
            _,pos=self.robot.get_position()
            if pos:
                self.fj_pose_x.setText(f"X: {pos[0]:.1f}"); self.fj_pose_y.setText(f"Y: {pos[1]:.1f}"); self.fj_pose_z.setText(f"Z: {pos[2]:.1f}")
                self.fj_pose_roll.setText(f"Roll: {pos[3]:.1f}"); self.fj_pose_pitch.setText(f"Pitch: {pos[4]:.1f}"); self.fj_pose_yaw.setText(f"Yaw: {pos[5]:.1f}")
        except: pass
    def _fj_jog_step(self, axis, delta):
        if self._motion_busy or not self.robot: return
        try:
            _,pos=self.robot.get_position()
            if not pos: return
            idx={"X":0,"Y":1,"Z":2}.get(axis);
            if idx is None: return
            np_=list(pos[:6]); np_[idx]+=delta
            self._robot_move_to_guarded(*np_[:3],roll=np_[3],pitch=np_[4],yaw=np_[5],speed=self.MOVE_SPEED_SLOW,wait=True)
        except Exception as e: print(f"[FJ Step] {e}")
    def _fj_jog_rot(self, axis, sign):
        """Step-jog a rotation axis by the configured degree increment."""
        if self.fj_rot_mode.currentText()!="Step": return
        if self._motion_busy or not self.robot: return
        try:
            _,pos=self.robot.get_position()
            if not pos: return
            idx={"Roll":3,"Pitch":4,"Yaw":5}.get(axis)
            if idx is None: return
            np_=list(pos[:6]); np_[idx]+=self.fj_rot_step.value()*sign
            self._robot_move_to_guarded(np_[0],np_[1],np_[2],roll=np_[3],pitch=np_[4],yaw=np_[5],speed=self.MOVE_SPEED_SLOW,wait=True)
        except Exception as e: print(f"[FJ Rot] {e}")
    def _fj_rot_mode_changed(self, m):
        s=m=="Step"
        self.fj_rot_step.setVisible(s); self.fj_rot_vel.setVisible(not s); self.fj_rot_vel_lbl.setVisible(not s)
    def _fj_jog_rot_cont_start(self, axis, sign):
        """Continuous rotation: send far target at configured angular velocity."""
        if self.fj_rot_mode.currentText()!="Continuous": return
        if not self.robot: return
        try:
            self._ensure_robot_ready(); _,pos=self.robot.get_position()
            if not pos: return
            idx={"Roll":3,"Pitch":4,"Yaw":5}.get(axis)
            if idx is None: return
            t=list(pos[:6]); t[idx]+=sign*90  # far target ±90°
            vel=self.fj_rot_vel.value()
            self.robot.set_position(x=t[0],y=t[1],z=t[2],roll=t[3],pitch=t[4],yaw=t[5],speed=vel,mvacc=1000,wait=False)
        except Exception as e: print(f"[FJ Rot Cont] {e}")
    def _fj_jog_rot_cont_stop(self):
        """Halt continuous rotation movement."""
        if self.fj_rot_mode.currentText()!="Continuous": return
        if not self.robot: return
        try: self.robot.set_state(4); time.sleep(0.02); self.robot.set_mode(0); self.robot.set_state(0)
        except Exception as e: print(f"[FJ Rot Stop] {e}")
    def _fj_jog_cont_start(self, axis, sign):
        if not self.robot: return
        try:
            self._ensure_robot_ready(); _,pos=self.robot.get_position()
            if not pos: return
            idx={"X":0,"Y":1,"Z":2}.get(axis);
            if idx is None: return
            t=list(pos[:6]); t[idx]+=sign*500
            self.robot.set_position(x=t[0],y=t[1],z=t[2],roll=t[3],pitch=t[4],yaw=t[5],speed=self.fj_jog.vel_slider.value(),mvacc=2000,wait=False)
        except Exception as e: print(f"[FJ Cont] {e}")
    def _fj_jog_cont_stop(self):
        if not self.robot: return
        try: self.robot.set_state(4); time.sleep(0.02); self.robot.set_mode(0); self.robot.set_state(0)
        except Exception as e: print(f"[FJ Stop] {e}")
    def _fj_enable(self):
        if not self.robot: QMessageBox.warning(self,"","Not connected."); return
        try:
            self.robot.clean_error(); self.robot.clean_warn(); self.robot.motion_enable(True); self.robot.set_mode(0); self.robot.set_state(0)
            self._estop_flag=False  # clear E-STOP on re-enable
            # Reset manual mode toggle if it was active
            if self.btn_fj_manual.isChecked():
                self.btn_fj_manual.setChecked(False)
                self.btn_fj_manual.setText("Manual Move (Freedrive)")
                self.btn_fj_manual.setStyleSheet("")
            self._update_robot_status()
        except Exception as e: QMessageBox.critical(self,"Error",str(e))
    def _fj_disable(self):
        if not self.robot: return
        try: self.robot.motion_enable(False); self._update_robot_status()
        except: pass
    def _fj_set_home(self):
        if not self.robot: return
        try:
            _,pos=self.robot.get_position()
            if pos:
                with open(self.home_position_path,"w") as f: json.dump({"x":pos[0],"y":pos[1],"z":pos[2],"roll":pos[3],"pitch":pos[4],"yaw":pos[5]},f)
                QMessageBox.information(self,"Home","Home position saved.")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))
    def _fj_goto_home(self):
        if not self.robot: return
        if not os.path.exists(self.home_position_path): QMessageBox.warning(self,"","No home set."); return
        with open(self.home_position_path) as f: h=json.load(f)
        self._robot_move_to(h["x"],h["y"],h["z"],h["roll"],h["pitch"],h["yaw"],speed=self.MOVE_SPEED)
    def _fj_manual_toggle(self):
        """Toggle xArm manual/freedrive mode (mode 2) — operator can move robot by hand."""
        if not self.robot:
            QMessageBox.warning(self,"","Not connected."); self.btn_fj_manual.setChecked(False); return
        try:
            if self.btn_fj_manual.isChecked():
                self.robot.set_mode(2); self.robot.set_state(0)
                self.btn_fj_manual.setText("⏹ Exit Manual Mode")
                self.btn_fj_manual.setStyleSheet(
                    f"background-color:{WARNING};color:{DARK_BG};border:none;border-radius:6px;"
                    f"padding:7px 14px;font-size:12px;font-weight:700;")
            else:
                self.robot.set_mode(0); self.robot.set_state(0)
                self.btn_fj_manual.setText("Manual Move (Freedrive)")
                self.btn_fj_manual.setStyleSheet("")  # reset to stylesheet default
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e)); self.btn_fj_manual.setChecked(False)

    # -----------------------------------------------------------------
    #  GLOBAL E-STOP — used from Free Jog AND Production
    # -----------------------------------------------------------------
    def _global_estop(self):
        """Immediately halt all motion, disable motors, and set flag to break production loops."""
        self._estop_flag=True
        if self.robot:
            try:
                self.robot.set_state(4)            # controller-level halt
                self.robot.motion_enable(False)    # disable motors — arm goes limp
            except: pass
        # Exit manual mode if active
        if hasattr(self, 'btn_fj_manual') and self.btn_fj_manual.isChecked():
            self.btn_fj_manual.setChecked(False)
            self.btn_fj_manual.setText("Manual Move (Freedrive)")
            self.btn_fj_manual.setStyleSheet("")
        self._update_robot_status()
        print("[E-STOP] TRIGGERED — motors disabled")

    # ===== HELPERS: jog from actual position =====
    def _jog_step_from_robot(self, axis, delta, spin_x, spin_y, spin_z):
        if self._motion_busy or not self.robot: return
        try:
            _,pos=self.robot.get_position()
            if not pos: return
            idx={"X":0,"Y":1,"Z":2}.get(axis)
            if idx is None: return
            np_=list(pos[:6]); np_[idx]+=delta
            spin_x.setValue(round(np_[0],1)); spin_y.setValue(round(np_[1],1)); spin_z.setValue(round(np_[2],1))
            self._robot_move_to_guarded(np_[0],np_[1],np_[2],roll=np_[3],pitch=np_[4],yaw=np_[5],speed=self.MOVE_SPEED_SLOW,wait=True)
        except Exception as e: print(f"[Jog Step] {e}")
    def _jog_cont_start_from_robot(self, axis, sign, vel):
        if not self.robot: return
        try:
            self._ensure_robot_ready(); _,pos=self.robot.get_position()
            if not pos: return
            idx={"X":0,"Y":1,"Z":2}.get(axis)
            if idx is None: return
            t=list(pos[:6]); t[idx]+=sign*500
            self.robot.set_position(x=t[0],y=t[1],z=t[2],roll=t[3],pitch=t[4],yaw=t[5],speed=vel,mvacc=2000,wait=False)
        except Exception as e: print(f"[Jog Cont] {e}")
    def _jog_cont_stop_common(self):
        if not self.robot: return
        try: self.robot.set_state(4); time.sleep(0.02); self.robot.set_mode(0); self.robot.set_state(0)
        except: pass
    def _sync_spinboxes_from_robot(self, spin_x, spin_y, spin_z):
        if not self.robot: return
        try:
            _,pos=self.robot.get_position()
            if pos: spin_x.setValue(round(pos[0],1)); spin_y.setValue(round(pos[1],1)); spin_z.setValue(round(pos[2],1))
        except: pass
    def _load_home(self):
        if os.path.exists(self.home_position_path):
            with open(self.home_position_path) as f: return json.load(f)
        return None
    def _go_home_and_append(self, waypoints):
        h=self._load_home()
        if h:
            waypoints.append({"label":"return_home","x":h["x"],"y":h["y"],"z":h["z"],"gripper":"open","move_type":"moveJ"})
            self._robot_move_to(h["x"],h["y"],h["z"],h["roll"],h["pitch"],h["yaw"],speed=self.MOVE_SPEED,move_type="moveJ")

    # ===== PAGE 2: CALIBRATION =====
    def _page_calibration(self):
        page=QWidget(); lay=QVBoxLayout(page); lay.setContentsMargins(16,16,16,16); lay.setSpacing(8)
        lay.addWidget(self._mk_title("Camera-to-Robot Calibration"))
        self.calib_status=QLabel(f"Homography: {'Ready' if self.H_matrix is not None else 'Not set'}")
        self.calib_status.setObjectName("status"); lay.addWidget(self.calib_status)
        content=QHBoxLayout(); content.setSpacing(12)
        left=QVBoxLayout(); left.setSpacing(6)
        self.calib_viewer=ClickableVideoWidget(); self.calib_viewer.setMinimumHeight(250)
        self.calib_viewer.pixel_clicked.connect(self._calib_on_image_click); left.addWidget(self.calib_viewer,stretch=1)
        cc=QHBoxLayout()
        self.btn_calib_feed=QPushButton("Live Feed"); self.btn_calib_feed.setObjectName("action"); self.btn_calib_feed.clicked.connect(self._calib_toggle_feed)
        self.btn_calib_capture=QPushButton("Capture"); self.btn_calib_capture.setObjectName("action"); self.btn_calib_capture.clicked.connect(self._calib_capture_pattern)
        cc.addWidget(self.btn_calib_feed); cc.addWidget(self.btn_calib_capture); left.addLayout(cc)
        self.calib_log=QTextEdit(); self.calib_log.setReadOnly(True); self.calib_log.setMaximumHeight(80); left.addWidget(self.calib_log)
        content.addLayout(left,stretch=2)
        right=QVBoxLayout(); right.setSpacing(8)
        pb_=QGroupBox("Pattern (mm)"); pg=QGridLayout(pb_); pg.setContentsMargins(8,18,8,6)
        self.calib_cols=QSpinBox(); self.calib_cols.setRange(2,20); self.calib_cols.setValue(4)
        self.calib_rows=QSpinBox(); self.calib_rows.setRange(2,20); self.calib_rows.setValue(11)
        self.calib_sq=QDoubleSpinBox(); self.calib_sq.setRange(1,100); self.calib_sq.setValue(15.0)
        for i,(l,w) in enumerate([("Cols:",self.calib_cols),("Rows:",self.calib_rows),("Sp:",self.calib_sq)]): pg.addWidget(QLabel(l),i,0); pg.addWidget(w,i,1)
        right.addWidget(pb_)
        bc=QPushButton("Compute Homography"); bc.setObjectName("action"); bc.clicked.connect(self._calib_compute); right.addWidget(bc)
        ob=QGroupBox("Robot ↔ Camera Offset (mm)"); og=QGridLayout(ob); og.setContentsMargins(8,18,8,6)
        self.calib_off_x=QDoubleSpinBox(); self.calib_off_x.setRange(-1000,1000); self.calib_off_x.setValue(200.0); self.calib_off_x.setDecimals(1)
        self.calib_off_y=QDoubleSpinBox(); self.calib_off_y.setRange(-1000,1000); self.calib_off_y.setValue(0.0); self.calib_off_y.setDecimals(1)
        og.addWidget(QLabel("Off X:"),0,0); og.addWidget(self.calib_off_x,0,1)
        og.addWidget(QLabel("Off Y:"),1,0); og.addWidget(self.calib_off_y,1,1)
        self.btn_calib_teach=QPushButton("Teach Reference Point"); self.btn_calib_teach.setObjectName("action")
        self.btn_calib_teach.setCheckable(True); self.btn_calib_teach.clicked.connect(self._calib_teach_toggle)
        og.addWidget(self.btn_calib_teach,2,0,1,2)
        ti=QLabel("1) Jog TCP to circle  2) Click button  3) Click circle on camera → offsets computed")
        ti.setObjectName("subtitle"); ti.setWordWrap(True); og.addWidget(ti,3,0,1,2)
        right.addWidget(ob); right.addStretch(); content.addLayout(right,stretch=1)
        lay.addLayout(content,stretch=1); return page

    def _calib_toggle_feed(self):
        if self.calib_timer.isActive():
            self.calib_timer.stop()
            if self.calib_cap: self.calib_cap.release(); self.calib_cap=None
            self.btn_calib_feed.setText("Live Feed")
        else:
            p=cv2.SimpleBlobDetector_Params()
            p.minThreshold,p.maxThreshold,p.thresholdStep=0,255,5
            p.filterByArea,p.minArea,p.maxArea=True,150,50000; p.filterByCircularity,p.minCircularity=True,0.55
            p.filterByColor,p.blobColor=True,0
            self.calib_detector=cv2.SimpleBlobDetector_create(p)
            self.calib_clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
            self.calib_cap=cv2.VideoCapture(self.cam_index_top)
            if self.calib_cap.isOpened(): self.calib_timer.start(30); self.btn_calib_feed.setText("Stop")
            else: QMessageBox.warning(self,"Camera","Failed.")
    def _calib_update_frame(self):
        if not self.calib_cap: return
        ret,frame=self.calib_cap.read()
        if not ret: return
        gray=self.calib_clahe.apply(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        kps=self.calib_detector.detect(gray)
        display=cv2.drawKeypoints(frame,kps,None,(180,180,180),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        c,r=self.calib_cols.value(),self.calib_rows.value()
        found,centers=cv2.findCirclesGrid(gray,(c,r),flags=cv2.CALIB_CB_ASYMMETRIC_GRID|cv2.CALIB_CB_CLUSTERING,blobDetector=self.calib_detector)
        if found:
            pts=centers.reshape(-1,2)
            for i in range(len(pts)-1):
                p1=tuple(pts[i].astype(int)); p2=tuple(pts[i+1].astype(int))
                cv2.line(display,p1,p2,(0,0,255),2,cv2.LINE_AA)
            for pt in pts: cv2.circle(display,tuple(pt.astype(int)),8,(0,0,255),2,cv2.LINE_AA)
            tcp_pt=tuple(pts[0].astype(int))
            cv2.circle(display,tcp_pt,14,(0,230,255),3,cv2.LINE_AA)
            cv2.circle(display,tcp_pt,5,(0,230,255),-1,cv2.LINE_AA)
            cv2.putText(display,"TCP ref",(tcp_pt[0]+16,tcp_pt[1]+6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,230,255),2,cv2.LINE_AA)
        self.calib_viewer.set_image(display)
    def _calib_capture_pattern(self):
        if not self.calib_cap: return
        ret,frame=self.calib_cap.read()
        if not ret: return
        gray=self.calib_clahe.apply(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        c,r=self.calib_cols.value(),self.calib_rows.value()
        found,centers=cv2.findCirclesGrid(gray,(c,r),flags=cv2.CALIB_CB_ASYMMETRIC_GRID|cv2.CALIB_CB_CLUSTERING,blobDetector=self.calib_detector)
        if found: self.calib_image_pts.append(centers.reshape(-1,2)); self.calib_log.append(f"Capture {len(self.calib_image_pts)} OK")
        else: self.calib_log.append("Pattern not found.")
    def _calib_compute(self):
        if not self.calib_image_pts: QMessageBox.warning(self,"","Capture first."); return
        c,r,sq=self.calib_cols.value(),self.calib_rows.value(),self.calib_sq.value()
        objp=np.zeros((r*c,2),np.float32)
        for i in range(r):
            for j in range(c): objp[i*c+j]=(j*sq*2+(i%2)*sq,i*sq)
        H,_=cv2.findHomography(np.vstack(self.calib_image_pts),np.vstack([objp]*len(self.calib_image_pts)))
        self.H_matrix=H; self.H_inv=np.linalg.inv(H); np.save(self.homography_path,H)
        self.calib_status.setText("Homography: Ready"); self.calib_log.append("Saved."); self.calib_image_pts.clear()
    def _calib_teach_toggle(self):
        if self.btn_calib_teach.isChecked():
            if self.H_matrix is None: QMessageBox.warning(self,"","Compute homography first."); self.btn_calib_teach.setChecked(False); return
            if not self.robot: QMessageBox.warning(self,"","Connect robot first."); self.btn_calib_teach.setChecked(False); return
            self._calib_teach_mode=True; self.btn_calib_teach.setText("⏳ Click circle on camera...")
        else: self._calib_teach_mode=False; self.btn_calib_teach.setText("Teach Reference Point")
    def _calib_on_image_click(self, px, py):
        if self.H_matrix is None: return
        p=cv2.perspectiveTransform(np.array([[[px,py]]],dtype=np.float32),self.H_inv)
        pat_x,pat_y=p[0][0][0],p[0][0][1]
        if self._calib_teach_mode:
            if not self.robot: return
            try:
                _,pos=self.robot.get_position()
                if not pos: return
                ox=pos[0]-pat_x; oy=pos[1]-pat_y
                self.calib_off_x.setValue(round(ox,1)); self.calib_off_y.setValue(round(oy,1))
                self.calib_log.append(f"AUTO: Off X={ox:.1f}, Off Y={oy:.1f}")
            except Exception as e: self.calib_log.append(f"ERROR: {e}")
            self._calib_teach_mode=False; self.btn_calib_teach.setChecked(False); self.btn_calib_teach.setText("Teach Reference Point")
        else:
            wx=pat_x+self.calib_off_x.value(); wy=pat_y+self.calib_off_y.value()
            self.calib_log.append(f"({px:.0f},{py:.0f}) → Robot({wx:.1f},{wy:.1f}) mm")

    # ===== PAGE 3: VISION TRAINING =====
    def _page_vision(self):
        page=QWidget(); lay=QVBoxLayout(page); lay.setContentsMargins(16,16,16,16); lay.setSpacing(8)
        lay.addWidget(self._mk_title("Vision Training"))
        self.vis_status=QLabel("Ready."); self.vis_status.setObjectName("status"); lay.addWidget(self.vis_status)
        self.vis_progress=QProgressBar(); self.vis_progress.setValue(0); lay.addWidget(self.vis_progress)
        self.vis_viewer=BoundingBoxWidget(); lay.addWidget(self.vis_viewer,stretch=1)
        ctrl=QHBoxLayout(); ctrl.setSpacing(6)
        bc=QPushButton("Capture"); bc.setObjectName("action"); bc.clicked.connect(self._vis_capture)
        self.btn_vis_save=QPushButton("Save"); self.btn_vis_save.setObjectName("action"); self.btn_vis_save.setEnabled(False); self.btn_vis_save.clicked.connect(self._vis_save)
        bu=QPushButton("Undo Box"); bu.setObjectName("action"); bu.clicked.connect(self._vis_undo_box)
        self.btn_vis_train=QPushButton("Train"); self.btn_vis_train.setObjectName("action"); self.btn_vis_train.clicked.connect(self._vis_train)
        self.btn_vis_test=QPushButton("Test"); self.btn_vis_test.setObjectName("action"); self.btn_vis_test.setEnabled(os.path.isfile(self.model_path)); self.btn_vis_test.clicked.connect(self._vis_test)
        self.btn_vis_del_last=QPushButton("Del Last"); self.btn_vis_del_last.setObjectName("danger"); self.btn_vis_del_last.clicked.connect(self._vis_delete_last)
        self.btn_vis_del_all=QPushButton("Clear All"); self.btn_vis_del_all.setObjectName("danger"); self.btn_vis_del_all.clicked.connect(self._vis_delete_all)
        for b in (bc,self.btn_vis_save,bu,self.btn_vis_train,self.btn_vis_test,self.btn_vis_del_last,self.btn_vis_del_all): ctrl.addWidget(b)
        lay.addLayout(ctrl)
        pb_=QGroupBox("Parameters"); pg=QHBoxLayout(pb_); pg.setContentsMargins(10,20,10,8); pg.setSpacing(8)
        self.spin_epochs=QSpinBox(); self.spin_epochs.setRange(5,200); self.spin_epochs.setValue(30)
        self.spin_aug=QSpinBox(); self.spin_aug.setRange(5,100); self.spin_aug.setValue(25)
        self.spin_imgsz=QSpinBox(); self.spin_imgsz.setRange(320,1280); self.spin_imgsz.setSingleStep(32); self.spin_imgsz.setValue(640)
        self.spin_conf=QDoubleSpinBox(); self.spin_conf.setRange(0.05,0.95); self.spin_conf.setSingleStep(0.05); self.spin_conf.setValue(0.25)
        for l,w in [("Epochs:",self.spin_epochs),("Aug:",self.spin_aug),("Img:",self.spin_imgsz),("Conf:",self.spin_conf)]: pg.addWidget(QLabel(l)); pg.addWidget(w)
        lay.addWidget(pb_)
        self.vis_log=QTextEdit(); self.vis_log.setReadOnly(True); self.vis_log.setMaximumHeight(60); lay.addWidget(self.vis_log); return page
    def _vis_capture(self):
        cap=cv2.VideoCapture(self.cam_index_top); ret,f=cap.read(); cap.release()
        if ret: self.vis_viewer.set_image(f); self.btn_vis_save.setEnabled(True); self.vis_status.setText("Draw boxes.")
    def _vis_undo_box(self):
        if self.vis_viewer.boxes: self.vis_viewer.boxes.pop(); self.vis_viewer.update()
    def _vis_save(self):
        if self.vis_viewer.original_image_cv is None or not self.vis_viewer.boxes: return
        self.vision_dataset.append({"image":self.vis_viewer.original_image_cv.copy(),"boxes":list(self.vis_viewer.boxes)})
        self.vis_log.append(f"Saved #{len(self.vision_dataset)}"); self.dash_dataset_lbl.setText(f"Dataset: {len(self.vision_dataset)}")
        self.vis_viewer.clear_boxes(); self.btn_vis_save.setEnabled(False)
    def _vis_delete_last(self):
        if not self.vision_dataset: return
        self.vision_dataset.pop(); self.vis_log.append(f"Deleted. Dataset: {len(self.vision_dataset)}"); self.dash_dataset_lbl.setText(f"Dataset: {len(self.vision_dataset)}")
    def _vis_delete_all(self):
        if not self.vision_dataset: return
        if QMessageBox.question(self,"Clear",f"Delete all {len(self.vision_dataset)} samples?",QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        self.vision_dataset.clear(); self.vis_log.append("Cleared."); self.dash_dataset_lbl.setText("Dataset: 0")
    def _vis_train(self):
        if not self.vision_dataset: return
        self.btn_vis_train.setEnabled(False); self.vis_progress.setValue(0)
        self.trainer=YoloTrainerWorker(self.vision_dataset,self.project_dir,self.spin_epochs.value(),self.spin_aug.value(),self.spin_imgsz.value())
        self.trainer.status_update.connect(self.vis_status.setText); self.trainer.progress_update.connect(self.vis_progress.setValue)
        self.trainer.training_finished.connect(self._vis_train_done); self.trainer.start()
    def _vis_train_done(self, bp):
        self.model_path=bp; self.dash_model_lbl.setText("Model: Yes"); self.btn_vis_test.setEnabled(True); self.btn_vis_train.setEnabled(True)
        self.current_model=YOLO(self.model_path)
    def _vis_test(self):
        if not os.path.isfile(self.model_path): return
        cap=cv2.VideoCapture(self.cam_index_top); ret,f=cap.read(); cap.release()
        if not ret: return
        r=YOLO(self.model_path)(f,conf=self.spin_conf.value()); self.vis_viewer.set_image(r[0].plot()); self.vis_status.setText(f"{len(r[0].boxes)} detections.")

    # =========================================================================
    #  PAGE 4: PICK & PLACE
    # =========================================================================
    PP_STEPS=["Step 1/5 – Capture & select PICK part.",
        "Step 2/5 – Robot ABOVE pick. Jog XY. Save.",
        "Step 3/5 – Lower to PICK. Jog Z. Save (gripper CLOSES).",
        "Step 4/5 – Jog to PLACE approach. Save XY.",
        "Step 5/5 – Lower to PLACE. Jog Z. Save (gripper OPENS → home).",]
    PP_SHORT = ["Detect", "Above Pick", "Pick", "Above Place", "Place"]

    def _page_pick_place(self):
        page=QWidget(); lay=QVBoxLayout(page); lay.setContentsMargins(12,10,12,8); lay.setSpacing(6)
        hdr=QHBoxLayout(); hdr.addWidget(self._mk_title("Pick & Place")); hdr.addStretch(); lay.addLayout(hdr)
        self.pp_step_ind=StepIndicator(self.PP_SHORT); lay.addWidget(self.pp_step_ind)
        self.pp_instruction=QLabel("Press 'Start New Routine'."); self.pp_instruction.setObjectName("step_instruction"); self.pp_instruction.setWordWrap(True); lay.addWidget(self.pp_instruction)
        content=QHBoxLayout(); content.setSpacing(10)
        left=QVBoxLayout(); left.setSpacing(6)
        self.pp_viewer=QLabel("Camera"); self.pp_viewer.setAlignment(Qt.AlignCenter)
        self.pp_viewer.setStyleSheet(f"background:#11111b;border:2px solid {BORDER};border-radius:10px;"); self.pp_viewer.setMinimumHeight(180)
        left.addWidget(self.pp_viewer,stretch=1)
        self.pp_det_list=QTextEdit(); self.pp_det_list.setReadOnly(True); self.pp_det_list.setMaximumHeight(50); left.addWidget(self.pp_det_list)
        ds=QHBoxLayout(); ds.setSpacing(4); ds.addWidget(QLabel("Det:"))
        self.pp_det_idx=QSpinBox(); self.pp_det_idx.setRange(0,99); self.pp_det_idx.setFixedWidth(50); ds.addWidget(self.pp_det_idx)
        self.btn_pp_select=QPushButton("Select & Move"); self.btn_pp_select.setObjectName("action"); self.btn_pp_select.clicked.connect(self._pp_select_detection); self.btn_pp_select.setEnabled(False); ds.addWidget(self.btn_pp_select)
        left.addLayout(ds); content.addLayout(left,stretch=3)
        right=QVBoxLayout(); right.setSpacing(4)
        pos_row=QHBoxLayout(); pos_row.setSpacing(3)
        self.pp_x=QDoubleSpinBox(); self.pp_x.setRange(-1000,1000); self.pp_x.setDecimals(1)
        self.pp_y=QDoubleSpinBox(); self.pp_y.setRange(-1000,1000); self.pp_y.setDecimals(1)
        self.pp_z=QDoubleSpinBox(); self.pp_z.setRange(-100,500); self.pp_z.setDecimals(1); self.pp_z.setValue(200)
        for lbl,w in [("X",self.pp_x),("Y",self.pp_y),("Z",self.pp_z)]: pos_row.addWidget(QLabel(lbl)); pos_row.addWidget(w,1)
        right.addLayout(pos_row)
        self.pp_jog=AdvancedJogPanel("Jog")
        self.pp_jog.jog_step.connect(lambda a,d: self._jog_step_from_robot(a,d,self.pp_x,self.pp_y,self.pp_z))
        self.pp_jog.jog_cont_start.connect(lambda a,s: self._jog_cont_start_from_robot(a,s,self.pp_jog.vel_slider.value()))
        self.pp_jog.jog_cont_stop.connect(lambda: (self._jog_cont_stop_common(),self._sync_spinboxes_from_robot(self.pp_x,self.pp_y,self.pp_z)))
        right.addWidget(self.pp_jog)
        self.pp_gripper=GripperPanel("Gripper"); self.pp_gripper.gripper_requested.connect(self._gripper_open_close); right.addWidget(self.pp_gripper)
        ag=QGridLayout(); ag.setSpacing(4)
        self.btn_pp_start=QPushButton("Start New Routine"); self.btn_pp_start.setObjectName("action"); self.btn_pp_start.clicked.connect(self._pp_start_routine)
        self.btn_pp_capture=QPushButton("Capture & Detect"); self.btn_pp_capture.setObjectName("action"); self.btn_pp_capture.clicked.connect(self._pp_capture_detect); self.btn_pp_capture.setEnabled(False)
        self.btn_pp_recapture=QPushButton("Recapture"); self.btn_pp_recapture.setObjectName("action"); self.btn_pp_recapture.clicked.connect(self._pp_recapture); self.btn_pp_recapture.setEnabled(False)
        self.btn_pp_save_wp=QPushButton("▶ SAVE WAYPOINT"); self.btn_pp_save_wp.setObjectName("step_active"); self.btn_pp_save_wp.clicked.connect(self._pp_save_step); self.btn_pp_save_wp.setEnabled(False)
        self.btn_pp_add_cycle=QPushButton("Add Cycle"); self.btn_pp_add_cycle.setObjectName("action"); self.btn_pp_add_cycle.setEnabled(False); self.btn_pp_add_cycle.clicked.connect(self._pp_add_cycle)
        self.btn_pp_finish=QPushButton("Save Routine"); self.btn_pp_finish.setObjectName("action"); self.btn_pp_finish.setEnabled(False); self.btn_pp_finish.clicked.connect(self._pp_finish_routine)
        self.btn_pp_cancel=QPushButton("Cancel"); self.btn_pp_cancel.setObjectName("danger"); self.btn_pp_cancel.clicked.connect(self._pp_cancel_routine); self.btn_pp_cancel.setEnabled(False)
        ag.addWidget(self.btn_pp_start,0,0); ag.addWidget(self.btn_pp_cancel,0,1)
        ag.addWidget(self.btn_pp_capture,1,0); ag.addWidget(self.btn_pp_recapture,1,1)
        ag.addWidget(self.btn_pp_save_wp,2,0,1,2)
        ag.addWidget(self.btn_pp_add_cycle,3,0); ag.addWidget(self.btn_pp_finish,3,1)
        right.addLayout(ag); right.addStretch(); content.addLayout(right,stretch=2)
        lay.addLayout(content,stretch=1)
        self.pp_log=QTextEdit(); self.pp_log.setReadOnly(True); self.pp_log.setMaximumHeight(55); lay.addWidget(self.pp_log)
        self._pp_style_buttons(); return page

    def _pp_start_routine(self):
        dlg=ApproachZDialog(self,self.APPROACH_Z_DEFAULT)
        if dlg.exec()!=QDialog.Accepted: return
        self._guide_approach_z=dlg.get_z(); self._guide_step=0; self._guide_waypoints=[]; self._guide_detections=[]
        self.pp_log.clear(); self.pp_det_list.clear()
        self.btn_pp_start.setEnabled(False); self.btn_pp_capture.setEnabled(True)
        for b in [self.btn_pp_save_wp,self.btn_pp_select,self.btn_pp_add_cycle,self.btn_pp_finish,self.btn_pp_recapture]: b.setEnabled(False)
        self.btn_pp_cancel.setEnabled(True); self._pp_update_instruction()
        self.pp_log.append(f"Approach Z = {self._guide_approach_z:.1f}"); self._pp_style_buttons()
    def _pp_update_instruction(self):
        self.pp_instruction.setText(self.PP_STEPS[self._guide_step] if self._guide_step<len(self.PP_STEPS) else "Cycle done! Add another or save routine.")
    def _pp_capture_detect(self):
        if self.H_inv is None: QMessageBox.warning(self,"","Calibrate first."); return
        if self.current_model is None:
            if os.path.isfile(self.model_path): self.current_model=YOLO(self.model_path)
            else: QMessageBox.warning(self,"","Train YOLO first."); return
        cap=cv2.VideoCapture(self.cam_index_top); ret,frame=cap.read(); cap.release()
        if not ret: return
        results=self.current_model.predict(frame,conf=self.spin_conf.value())[0]
        ann=frame.copy(); self._guide_detections=[]; self.pp_det_list.clear()
        if len(results.boxes)==0:
            self.pp_det_list.append("No detections."); self._show_cv_on_label(ann,self.pp_viewer); self.btn_pp_recapture.setEnabled(True); self._pp_style_buttons(); return
        for i,box in enumerate(results.boxes):
            x1,y1,x2,y2=box.xyxy[0].tolist(); uc,vc=(x1+x2)/2,(y1+y2)/2
            pt=cv2.perspectiveTransform(np.array([[[uc,vc]]],dtype=np.float32),self.H_inv)
            rx=pt[0][0][0]+self.calib_off_x.value(); ry=pt[0][0][1]+self.calib_off_y.value()
            lbl=self.current_model.names[int(box.cls[0])]; conf=float(box.conf[0])
            self._guide_detections.append({"label":lbl,"rx":rx,"ry":ry,"conf":conf})
            cv2.circle(ann,(int(uc),int(vc)),5,(0,255,0),-1)
            cv2.putText(ann,f"[{i}] {lbl}",(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            self.pp_det_list.append(f"[{i}] {lbl}: X={rx:.1f}, Y={ry:.1f} ({conf:.2f})")
        self._show_cv_on_label(ann,self.pp_viewer); self.pp_det_idx.setRange(0,len(self._guide_detections)-1)
        self.btn_pp_select.setEnabled(True); self.btn_pp_capture.setEnabled(False); self.btn_pp_recapture.setEnabled(True); self._pp_style_buttons()
    def _pp_recapture(self):
        self._guide_detections=[]; self.pp_det_list.clear()
        self.btn_pp_select.setEnabled(False); self.btn_pp_recapture.setEnabled(False); self.btn_pp_capture.setEnabled(True); self._pp_style_buttons()
    def _pp_cancel_routine(self):
        if QMessageBox.question(self,"Cancel","Discard?",QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        self._guide_step=0; self._guide_waypoints=[]; self._guide_detections=[]; self.pp_det_list.clear()
        for b in [self.btn_pp_capture,self.btn_pp_recapture,self.btn_pp_save_wp,self.btn_pp_select,self.btn_pp_add_cycle,self.btn_pp_finish,self.btn_pp_cancel]: b.setEnabled(False)
        self.btn_pp_start.setEnabled(True); self.pp_instruction.setText("Cancelled."); self.pp_step_ind.reset(); self._pp_style_buttons()
    def _pp_select_detection(self):
        idx=self.pp_det_idx.value()
        if idx>=len(self._guide_detections): return
        det=self._guide_detections[idx]; az=self._guide_approach_z
        self.pp_x.setValue(det["rx"]); self.pp_y.setValue(det["ry"]); self.pp_z.setValue(az)
        self.pp_log.append(f"→ [{idx}] '{det['label']}'")
        self._robot_move_to(det["rx"],det["ry"],az,speed=self.MOVE_SPEED,move_type="moveJ")
        self.pp_log.append("Arrived above pick.")
        self.btn_pp_select.setEnabled(False); self.btn_pp_save_wp.setEnabled(True); self.btn_pp_recapture.setEnabled(False)
        if self._guide_step==0: self._guide_step=1
        self._pp_update_instruction(); self._pp_style_buttons()
    def _pp_save_step(self):
        x,y,z=self.pp_x.value(),self.pp_y.value(),self.pp_z.value(); az=self._guide_approach_z
        if self._guide_step==1:
            self._guide_waypoints.append({"label":"approach_pick","x":x,"y":y,"z":az,"gripper":"open","move_type":"moveJ"})
            self._guide_step=2; self.pp_z.setValue(az-100); self.pp_log.append(f"approach_pick Z={az:.1f}. Lower to pick.")
        elif self._guide_step==2:
            self._guide_waypoints.append({"label":"pick","x":x,"y":y,"z":z,"gripper":"close","move_type":"moveL"})
            self._gripper_open_close("close")
            self._guide_waypoints.append({"label":"ascend_after_pick","x":x,"y":y,"z":az,"gripper":"close","move_type":"moveL"})
            self._robot_move_to(x,y,az,speed=self.MOVE_SPEED_SLOW,move_type="moveL")
            self.pp_x.setValue(x); self.pp_y.setValue(y); self.pp_z.setValue(az)
            self._guide_step=3; self.btn_pp_save_wp.setEnabled(True)
            self.btn_pp_capture.setEnabled(False); self.btn_pp_recapture.setEnabled(False)
            self.pp_log.append(f"Picked Z={z:.1f}. Jog to PLACE approach, then Save.")
        elif self._guide_step==3:
            self._guide_waypoints.append({"label":"approach_place","x":x,"y":y,"z":az,"gripper":"close","move_type":"moveJ"})
            self._guide_step=4; self.pp_z.setValue(az-100); self.pp_log.append(f"approach_place Z={az:.1f}. Lower.")
        elif self._guide_step==4:
            self._guide_waypoints.append({"label":"place","x":x,"y":y,"z":z,"gripper":"open","move_type":"moveL"})
            self._gripper_open_close("open")
            self._guide_waypoints.append({"label":"ascend_after_place","x":x,"y":y,"z":az,"gripper":"open","move_type":"moveL"})
            self._robot_move_to(x,y,az,speed=self.MOVE_SPEED_SLOW,move_type="moveL")
            self._go_home_and_append(self._guide_waypoints)
            self._guide_step=5; self.btn_pp_save_wp.setEnabled(False)
            self.btn_pp_add_cycle.setEnabled(True); self.btn_pp_finish.setEnabled(True)
            self.pp_log.append(f"Placed Z={z:.1f}. Home. Cycle done!")
        self._pp_update_instruction(); self._pp_style_buttons()
    def _pp_add_cycle(self):
        self._guide_step=0; self.btn_pp_add_cycle.setEnabled(False); self.btn_pp_finish.setEnabled(False)
        self.btn_pp_capture.setEnabled(True); self._pp_update_instruction(); self._pp_style_buttons()
    def _pp_finish_routine(self):
        name,ok=QInputDialog.getText(self,"Name","Routine name:")
        if not ok or not name.strip(): return
        name=name.strip().replace(" ","_")
        r=Routine(name=name,routine_type="pick_place",waypoints=self._guide_waypoints,approach_z=self._guide_approach_z)
        self._save_routine(r); self.pp_log.append(f"'{name}' saved."); self.btn_pp_start.setEnabled(True)
        for b in [self.btn_pp_add_cycle,self.btn_pp_finish,self.btn_pp_capture,self.btn_pp_recapture,self.btn_pp_cancel]: b.setEnabled(False)
        self.pp_instruction.setText("Saved!"); self.dash_routines_lbl.setText(f"Routines: {len(self._list_routines())}")
        self.pp_step_ind.set_all_done(); self._pp_style_buttons()

    # =========================================================================
    #  PAGE 5: PALLETIZING  (config dialog on start)
    # =========================================================================
    PAL_STEPS=["Step 1/5 – Capture & select PICK part.",
        "Step 2/5 – Above pick. Jog XY. Save.",
        "Step 3/5 – Lower to pick. Jog Z. Save (gripper CLOSE).",
        "Step 4/5 – Jog to PALLET ORIGIN (1st cell). Save XY.",
        "Step 5/5 – Lower to place. Jog Z. Save (gripper OPEN → home).",]
    PAL_SHORT = ["Detect", "Above Pick", "Pick", "Pallet Origin", "Place"]

    def _page_palletize(self):
        page=QWidget(); lay=QVBoxLayout(page); lay.setContentsMargins(12,10,12,8); lay.setSpacing(6)
        hdr=QHBoxLayout(); hdr.addWidget(self._mk_title("Palletizing")); hdr.addStretch(); lay.addLayout(hdr)
        self.pal_step_ind=StepIndicator(self.PAL_SHORT); lay.addWidget(self.pal_step_ind)
        self.pal_instruction=QLabel("Press 'Start'."); self.pal_instruction.setObjectName("step_instruction"); self.pal_instruction.setWordWrap(True); lay.addWidget(self.pal_instruction)
        content=QHBoxLayout(); content.setSpacing(10)
        left=QVBoxLayout(); left.setSpacing(6)
        self.pal_viewer=QLabel("Camera"); self.pal_viewer.setAlignment(Qt.AlignCenter)
        self.pal_viewer.setStyleSheet(f"background:#11111b;border:2px solid {BORDER};border-radius:10px;"); self.pal_viewer.setMinimumHeight(160)
        left.addWidget(self.pal_viewer,stretch=1)
        self.pal_det_list=QTextEdit(); self.pal_det_list.setReadOnly(True); self.pal_det_list.setMaximumHeight(45); left.addWidget(self.pal_det_list)
        ds=QHBoxLayout(); ds.setSpacing(4); ds.addWidget(QLabel("Det:"))
        self.pal_det_idx=QSpinBox(); self.pal_det_idx.setRange(0,99); self.pal_det_idx.setFixedWidth(50); ds.addWidget(self.pal_det_idx)
        self.btn_pal_select=QPushButton("Select"); self.btn_pal_select.setObjectName("action"); self.btn_pal_select.clicked.connect(self._pal_select_detection); self.btn_pal_select.setEnabled(False); ds.addWidget(self.btn_pal_select)
        left.addLayout(ds); content.addLayout(left,stretch=3)
        right=QVBoxLayout(); right.setSpacing(4)
        # Config summary (read-only, set by dialog)
        self.pal_config_lbl=QLabel("No config set. Press Start."); self.pal_config_lbl.setObjectName("subtitle"); self.pal_config_lbl.setWordWrap(True)
        right.addWidget(self.pal_config_lbl)
        pos_row=QHBoxLayout(); pos_row.setSpacing(3)
        self.pal_x=QDoubleSpinBox(); self.pal_x.setRange(-1000,1000); self.pal_x.setDecimals(1)
        self.pal_y=QDoubleSpinBox(); self.pal_y.setRange(-1000,1000); self.pal_y.setDecimals(1)
        self.pal_z=QDoubleSpinBox(); self.pal_z.setRange(-100,500); self.pal_z.setDecimals(1); self.pal_z.setValue(200)
        for lbl,w in [("X",self.pal_x),("Y",self.pal_y),("Z",self.pal_z)]: pos_row.addWidget(QLabel(lbl)); pos_row.addWidget(w,1)
        right.addLayout(pos_row)
        self.pal_jog=AdvancedJogPanel("Jog")
        self.pal_jog.jog_step.connect(lambda a,d: self._jog_step_from_robot(a,d,self.pal_x,self.pal_y,self.pal_z))
        self.pal_jog.jog_cont_start.connect(lambda a,s: self._jog_cont_start_from_robot(a,s,self.pal_jog.vel_slider.value()))
        self.pal_jog.jog_cont_stop.connect(lambda: (self._jog_cont_stop_common(),self._sync_spinboxes_from_robot(self.pal_x,self.pal_y,self.pal_z)))
        right.addWidget(self.pal_jog)
        self.pal_gripper=GripperPanel("Gripper"); self.pal_gripper.gripper_requested.connect(self._gripper_open_close); right.addWidget(self.pal_gripper)
        ag=QGridLayout(); ag.setSpacing(4)
        self.btn_pal_start=QPushButton("Start"); self.btn_pal_start.setObjectName("action"); self.btn_pal_start.clicked.connect(self._pal_start)
        self.btn_pal_capture=QPushButton("Capture"); self.btn_pal_capture.setObjectName("action"); self.btn_pal_capture.clicked.connect(self._pal_capture); self.btn_pal_capture.setEnabled(False)
        self.btn_pal_recapture=QPushButton("Recapture"); self.btn_pal_recapture.setObjectName("action"); self.btn_pal_recapture.clicked.connect(self._pal_recapture); self.btn_pal_recapture.setEnabled(False)
        self.btn_pal_save_wp=QPushButton("▶ SAVE WP"); self.btn_pal_save_wp.setObjectName("step_active"); self.btn_pal_save_wp.clicked.connect(self._pal_save_step); self.btn_pal_save_wp.setEnabled(False)
        self.btn_pal_finish=QPushButton("Save Routine"); self.btn_pal_finish.setObjectName("action"); self.btn_pal_finish.clicked.connect(self._pal_finish); self.btn_pal_finish.setEnabled(False)
        self.btn_pal_cancel=QPushButton("Cancel"); self.btn_pal_cancel.setObjectName("danger"); self.btn_pal_cancel.clicked.connect(self._pal_cancel); self.btn_pal_cancel.setEnabled(False)
        ag.addWidget(self.btn_pal_start,0,0); ag.addWidget(self.btn_pal_cancel,0,1)
        ag.addWidget(self.btn_pal_capture,1,0); ag.addWidget(self.btn_pal_recapture,1,1)
        ag.addWidget(self.btn_pal_save_wp,2,0,1,2)
        ag.addWidget(self.btn_pal_finish,3,0,1,2)
        right.addLayout(ag); right.addStretch(); content.addLayout(right,stretch=2)
        lay.addLayout(content,stretch=1)
        self.pal_log=QTextEdit(); self.pal_log.setReadOnly(True); self.pal_log.setMaximumHeight(50); lay.addWidget(self.pal_log)
        # Store pallet config from dialog (used at finish)
        self._pal_dialog_config = {}
        self._pal_style_buttons(); return page

    def _pal_start(self):
        """Open config dialog forcing operator to enter all grid measurements."""
        defaults = {
            "approach_z": self.APPROACH_Z_DEFAULT,
            "obj_w": 50, "obj_d": 50, "obj_h": 20,
            "gap_x": 5, "gap_y": 5, "cols": 3, "rows": 3, "layers": 1,
        }
        # Pre-fill from previous config if exists
        if self._pal_dialog_config:
            defaults.update(self._pal_dialog_config)
        dlg = PalletConfigDialog(self, defaults)
        if dlg.exec() != QDialog.Accepted:
            return
        cfg = dlg.get_config()
        self._pal_dialog_config = cfg
        self._pal_approach_z = cfg["approach_z"]
        # Update summary label
        px = cfg["obj_w"] + cfg["gap_x"]; py = cfg["obj_d"] + cfg["gap_y"]
        total = cfg["cols"] * cfg["rows"] * cfg["layers"]
        self.pal_config_lbl.setText(
            f"W={cfg['obj_w']:.0f} L={cfg['obj_d']:.0f} H={cfg['obj_h']:.0f}  "
            f"Gap={cfg['gap_x']:.0f}/{cfg['gap_y']:.0f}  "
            f"{cfg['cols']}×{cfg['rows']}×{cfg['layers']}={total}  "
            f"Pitch={px:.0f}/{py:.0f}")
        # Reset guide state
        self._pal_guide_step=0; self._pal_waypoints=[]; self._guide_detections=[]
        self.pal_log.clear(); self.pal_det_list.clear()
        self.btn_pal_start.setEnabled(False); self.btn_pal_capture.setEnabled(True)
        for b in [self.btn_pal_save_wp,self.btn_pal_select,self.btn_pal_finish,self.btn_pal_recapture]: b.setEnabled(False)
        self.btn_pal_cancel.setEnabled(True); self._pal_update_instruction()
        self.pal_log.append(f"Approach Z = {self._pal_approach_z:.1f}")
        self._pal_style_buttons()
    def _pal_update_instruction(self):
        self.pal_instruction.setText(self.PAL_STEPS[self._pal_guide_step] if self._pal_guide_step<len(self.PAL_STEPS) else "Done! Save routine.")
    def _pal_capture(self):
        if self.H_inv is None: return
        if self.current_model is None:
            if os.path.isfile(self.model_path): self.current_model=YOLO(self.model_path)
            else: return
        cap=cv2.VideoCapture(self.cam_index_top); ret,frame=cap.read(); cap.release()
        if not ret: return
        results=self.current_model.predict(frame,conf=self.spin_conf.value())[0]
        ann=frame.copy(); self._guide_detections=[]; self.pal_det_list.clear()
        for i,box in enumerate(results.boxes):
            x1,y1,x2,y2=box.xyxy[0].tolist(); uc,vc=(x1+x2)/2,(y1+y2)/2
            pt=cv2.perspectiveTransform(np.array([[[uc,vc]]],dtype=np.float32),self.H_inv)
            rx=pt[0][0][0]+self.calib_off_x.value(); ry=pt[0][0][1]+self.calib_off_y.value()
            lbl=self.current_model.names[int(box.cls[0])]
            self._guide_detections.append({"label":lbl,"rx":rx,"ry":ry})
            cv2.circle(ann,(int(uc),int(vc)),5,(0,255,0),-1)
            cv2.putText(ann,f"[{i}] {lbl}",(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            self.pal_det_list.append(f"[{i}] {lbl}: X={rx:.1f}, Y={ry:.1f}")
        self._show_cv_on_label(ann,self.pal_viewer)
        if self._guide_detections:
            self.pal_det_idx.setRange(0,len(self._guide_detections)-1); self.btn_pal_select.setEnabled(True); self.btn_pal_capture.setEnabled(False)
        self.btn_pal_recapture.setEnabled(True); self._pal_style_buttons()
    def _pal_recapture(self):
        self._guide_detections=[]; self.pal_det_list.clear(); self.btn_pal_select.setEnabled(False); self.btn_pal_recapture.setEnabled(False); self.btn_pal_capture.setEnabled(True); self._pal_style_buttons()
    def _pal_cancel(self):
        if QMessageBox.question(self,"Cancel","Discard?",QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        self._pal_guide_step=0; self._pal_waypoints=[]; self._guide_detections=[]; self.pal_det_list.clear()
        for b in [self.btn_pal_capture,self.btn_pal_recapture,self.btn_pal_save_wp,self.btn_pal_select,self.btn_pal_finish,self.btn_pal_cancel]: b.setEnabled(False)
        self.btn_pal_start.setEnabled(True); self.pal_instruction.setText("Cancelled."); self.pal_step_ind.reset(); self._pal_style_buttons()
    def _pal_select_detection(self):
        idx=self.pal_det_idx.value()
        if idx>=len(self._guide_detections): return
        det=self._guide_detections[idx]; az=self._pal_approach_z
        self.pal_x.setValue(det["rx"]); self.pal_y.setValue(det["ry"]); self.pal_z.setValue(az)
        self._robot_move_to(det["rx"],det["ry"],az,speed=self.MOVE_SPEED,move_type="moveJ")
        self.btn_pal_select.setEnabled(False); self.btn_pal_save_wp.setEnabled(True); self.btn_pal_recapture.setEnabled(False)
        self._pal_guide_step=1; self._pal_update_instruction(); self._pal_style_buttons()
    def _pal_save_step(self):
        x,y,z=self.pal_x.value(),self.pal_y.value(),self.pal_z.value(); az=self._pal_approach_z
        if self._pal_guide_step==1:
            self._pal_waypoints.append({"label":"approach_pick","x":x,"y":y,"z":az,"gripper":"open","move_type":"moveJ"})
            self._pal_guide_step=2; self.pal_z.setValue(az-100); self.pal_log.append(f"approach_pick Z={az:.1f}. Lower.")
        elif self._pal_guide_step==2:
            self._pal_waypoints.append({"label":"pick","x":x,"y":y,"z":z,"gripper":"close","move_type":"moveL"})
            self._gripper_open_close("close")
            self._pal_waypoints.append({"label":"ascend_after_pick","x":x,"y":y,"z":az,"gripper":"close","move_type":"moveL"})
            self._robot_move_to(x,y,az,speed=self.MOVE_SPEED_SLOW,move_type="moveL")
            self.pal_x.setValue(x); self.pal_y.setValue(y); self.pal_z.setValue(az)
            self._pal_guide_step=3; self.btn_pal_save_wp.setEnabled(True)
            self.btn_pal_capture.setEnabled(False); self.btn_pal_recapture.setEnabled(False)
            self.pal_log.append("Picked. Jog to PALLET ORIGIN, then Save.")
        elif self._pal_guide_step==3:
            self._pal_waypoints.append({"label":"approach_place","x":x,"y":y,"z":az,"gripper":"close","move_type":"moveJ"})
            self._pal_guide_step=4; self.pal_z.setValue(az-100); self.pal_log.append(f"approach_place Z={az:.1f}. Lower.")
        elif self._pal_guide_step==4:
            self._pal_waypoints.append({"label":"place_primary","x":x,"y":y,"z":z,"gripper":"open","move_type":"moveL"})
            self._gripper_open_close("open")
            self._pal_waypoints.append({"label":"ascend_after_place","x":x,"y":y,"z":az,"gripper":"open","move_type":"moveL"})
            self._robot_move_to(x,y,az,speed=self.MOVE_SPEED_SLOW,move_type="moveL")
            self._go_home_and_append(self._pal_waypoints)
            self._pal_guide_step=5; self.btn_pal_save_wp.setEnabled(False); self.btn_pal_finish.setEnabled(True)
            cfg=self._pal_dialog_config; px=cfg["obj_w"]+cfg["gap_x"]; py=cfg["obj_d"]+cfg["gap_y"]
            self.pal_log.append(f"Place Z={z:.1f}. Grid: Pitch X={px:.1f}, Y={py:.1f}, ΔZ={cfg['obj_h']:.1f}")
        self._pal_update_instruction(); self._pal_style_buttons()
    def _pal_finish(self):
        name,ok=QInputDialog.getText(self,"Name","Routine name:")
        if not ok or not name.strip(): return
        name=name.strip().replace(" ","_")
        cfg=self._pal_dialog_config
        pc={"obj_w":cfg["obj_w"],"obj_d":cfg["obj_d"],"obj_h":cfg["obj_h"],
            "cols":cfg["cols"],"rows":cfg["rows"],"layers":cfg["layers"],
            "gap_x":cfg["gap_x"],"gap_y":cfg["gap_y"]}
        r=Routine(name=name,routine_type="palletizing",waypoints=self._pal_waypoints,pallet_config=pc,approach_z=self._pal_approach_z)
        self._save_routine(r); self.pal_log.append(f"'{name}' saved."); self.btn_pal_start.setEnabled(True)
        for b in [self.btn_pal_finish,self.btn_pal_recapture,self.btn_pal_cancel]: b.setEnabled(False)
        self.pal_instruction.setText("Saved!"); self.dash_routines_lbl.setText(f"Routines: {len(self._list_routines())}")
        self.pal_step_ind.set_all_done(); self._pal_style_buttons()

    # =========================================================================
    #  PAGE 6: PRODUCTION  (E-STOP aware loops)
    # =========================================================================
    def _page_production(self):
        page=QWidget(); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        cw=QWidget(); lay=QVBoxLayout(cw); lay.setContentsMargins(16,16,16,16); lay.setSpacing(8)
        lay.addWidget(self._mk_title("Run Production"))
        self.prod_status=QLabel("Idle."); self.prod_status.setObjectName("status"); lay.addWidget(self.prod_status)
        self.prod_cam1=QLabel("Top Camera"); self.prod_cam1.setAlignment(Qt.AlignCenter)
        self.prod_cam1.setStyleSheet(f"background:#11111b;border:2px solid {BORDER};border-radius:10px;font-weight:normal;"); self.prod_cam1.setMinimumHeight(160)
        lay.addWidget(self.prod_cam1,stretch=1)
        self.btn_prod_cam=QPushButton("Live Camera"); self.btn_prod_cam.setObjectName("action"); self.btn_prod_cam.clicked.connect(self._prod_toggle_cam)
        lay.addWidget(self.btn_prod_cam)
        mb=QGroupBox("Execute Routine"); ml=QVBoxLayout(mb); ml.setContentsMargins(10,20,10,8); ml.setSpacing(6)
        r1=QHBoxLayout(); r1.setSpacing(6)
        self.prod_type_filter=QComboBox(); self.prod_type_filter.addItems(["All","pick_place","palletizing"])
        self.prod_type_filter.currentTextChanged.connect(self._prod_refresh)
        r1.addWidget(QLabel("Type:")); r1.addWidget(self.prod_type_filter)
        self.prod_routine_combo=QComboBox(); r1.addWidget(QLabel("Routine:")); r1.addWidget(self.prod_routine_combo,1)
        br=QPushButton("↻"); br.setObjectName("action"); br.setFixedWidth(30); br.clicked.connect(self._prod_refresh); r1.addWidget(br)
        ml.addLayout(r1)
        r2=QHBoxLayout(); r2.setSpacing(6)
        self.prod_cycles=QSpinBox(); self.prod_cycles.setRange(1,9999); self.prod_cycles.setValue(1)
        r2.addWidget(QLabel("Cycles:")); r2.addWidget(self.prod_cycles)
        r2.addWidget(QLabel("Speed:")); self.prod_vel_slider=QSlider(Qt.Horizontal); self.prod_vel_slider.setRange(10,300); self.prod_vel_slider.setValue(100)
        self.prod_vel_lbl=QLabel("100"); self.prod_vel_lbl.setFixedWidth(24); self.prod_vel_slider.valueChanged.connect(lambda v: self.prod_vel_lbl.setText(str(v)))
        r2.addWidget(self.prod_vel_slider,1); r2.addWidget(self.prod_vel_lbl)
        ml.addLayout(r2)
        r3=QHBoxLayout(); r3.setSpacing(6)
        r3.addWidget(QLabel("Delay:"))
        self.prod_delay_slider=QSlider(Qt.Horizontal); self.prod_delay_slider.setRange(3,50); self.prod_delay_slider.setValue(10)
        self.prod_delay_lbl=QLabel("1.0s"); self.prod_delay_lbl.setFixedWidth(30); self.prod_delay_slider.valueChanged.connect(lambda v: self.prod_delay_lbl.setText(f"{v/10:.1f}s"))
        r3.addWidget(self.prod_delay_slider,1); r3.addWidget(self.prod_delay_lbl)
        brun=QPushButton("RUN"); brun.setObjectName("action"); brun.clicked.connect(self._prod_run); r3.addWidget(brun)
        bstop=QPushButton("⛔ E-STOP"); bstop.setObjectName("estop"); bstop.clicked.connect(self._prod_stop); r3.addWidget(bstop)
        ml.addLayout(r3); lay.addWidget(mb)
        mg=QGroupBox("Manage Routines"); mgl=QHBoxLayout(mg); mgl.setContentsMargins(10,20,10,8); mgl.setSpacing(6)
        self.prod_routine_list=QListWidget(); mgl.addWidget(self.prod_routine_list,1)
        mbtn=QVBoxLayout(); mbtn.setSpacing(4)
        self.btn_prod_rename=QPushButton("Rename"); self.btn_prod_rename.setObjectName("action"); self.btn_prod_rename.clicked.connect(self._prod_rename_routine)
        self.btn_prod_delete=QPushButton("Delete"); self.btn_prod_delete.setObjectName("danger"); self.btn_prod_delete.clicked.connect(self._prod_delete_routine)
        self.btn_prod_refresh_list=QPushButton("Refresh"); self.btn_prod_refresh_list.setObjectName("action"); self.btn_prod_refresh_list.clicked.connect(self._prod_refresh_list)
        mbtn.addWidget(self.btn_prod_rename); mbtn.addWidget(self.btn_prod_delete); mbtn.addWidget(self.btn_prod_refresh_list); mbtn.addStretch()
        mgl.addLayout(mbtn); lay.addWidget(mg)
        self.prod_log=QTextEdit(); self.prod_log.setReadOnly(True); lay.addWidget(self.prod_log,stretch=1)
        scroll.setWidget(cw); ml2=QVBoxLayout(page); ml2.setContentsMargins(0,0,0,0); ml2.addWidget(scroll)
        self._prod_refresh(); self._prod_refresh_list(); return page

    def _prod_toggle_cam(self):
        if self.prod_cam_timer.isActive():
            self.prod_cam_timer.stop()
            if self.prod_cap: self.prod_cap.release(); self.prod_cap=None
            self.btn_prod_cam.setText("Live Camera")
        else:
            self.prod_cap=cv2.VideoCapture(self.cam_index_top)
            if self.prod_cap.isOpened(): self.prod_cam_timer.start(30); self.btn_prod_cam.setText("Stop Camera")
            else: QMessageBox.warning(self,"Camera","Failed.")
    def _prod_update_cam(self):
        if not self.prod_cap: return
        ret,frame=self.prod_cap.read()
        if not ret: return
        self._show_cv_on_label(frame,self.prod_cam1)
    def _prod_refresh(self):
        self.prod_routine_combo.clear(); tf=self.prod_type_filter.currentText()
        for r in self._list_routines():
            if tf=="All" or r["routine_type"]==tf: self.prod_routine_combo.addItem(f"{r['name']} ({r['routine_type']})",r["name"])
    def _prod_refresh_list(self):
        self.prod_routine_list.clear()
        for r in self._list_routines(): self.prod_routine_list.addItem(f"{r['name']}  [{r['routine_type']}]")
    def _prod_rename_routine(self):
        item=self.prod_routine_list.currentItem()
        if not item: return
        old_name=item.text().split("  [")[0]
        new_name,ok=QInputDialog.getText(self,"Rename",f"New name for '{old_name}':",text=old_name)
        if not ok or not new_name.strip(): return
        new_name=new_name.strip().replace(" ","_")
        old_path=os.path.join(self.routines_dir,f"{old_name}.json")
        if not os.path.isfile(old_path): return
        with open(old_path) as f: data=json.load(f)
        data["name"]=new_name
        new_path=os.path.join(self.routines_dir,f"{new_name}.json")
        with open(new_path,"w") as f: json.dump(data,f,indent=2)
        if old_path!=new_path: os.remove(old_path)
        self._prod_refresh(); self._prod_refresh_list(); self.prod_log.append(f"Renamed '{old_name}' → '{new_name}'")
    def _prod_delete_routine(self):
        item=self.prod_routine_list.currentItem()
        if not item: return
        name=item.text().split("  [")[0]
        if QMessageBox.question(self,"Delete",f"Delete routine '{name}'?",QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        p=os.path.join(self.routines_dir,f"{name}.json")
        if os.path.isfile(p): os.remove(p)
        self._prod_refresh(); self._prod_refresh_list(); self.prod_log.append(f"Deleted '{name}'")
        self.dash_routines_lbl.setText(f"Routines: {len(self._list_routines())}")
    def _prod_run(self):
        if not self.prod_routine_combo.count(): QMessageBox.warning(self,"","No routines."); return
        rn=self.prod_routine_combo.currentData(); routine=self._load_routine(rn)
        if not routine: return
        self._estop_flag=False  # reset E-STOP before run
        cycles=self.prod_cycles.value(); speed=self.prod_vel_slider.value(); delay=self.prod_delay_slider.value()/10.0
        self.prod_status.setText(f"RUNNING: {routine.name} x {cycles}")
        self.prod_status.setStyleSheet(f"color:{SUCCESS};font-weight:bold;font-size:13px;")
        self.prod_log.append(f"=== {routine.name} ({routine.routine_type}) x {cycles} | speed={speed} delay={delay:.1f}s ===")
        self._ensure_robot_ready()
        if routine.routine_type=="palletizing" and routine.pallet_config:
            self._prod_run_pal(routine,cycles,speed,delay)
        else:
            self._prod_run_pp(routine,cycles,speed,delay)

    def _prod_run_pp(self, routine, cycles, speed, delay):
        for c in range(cycles):
            if self._estop_flag: break
            self.prod_log.append(f"\n--- Cycle {c+1}/{cycles} ---")
            for wp in routine.waypoints:
                if self._estop_flag: break
                self.prod_log.append(f"  -> {wp['label']}: ({wp['x']:.1f},{wp['y']:.1f},{wp['z']:.1f})")
                self._robot_move_to(wp["x"],wp["y"],wp["z"],speed=speed,move_type=wp.get("move_type","moveL"))
                if self._estop_flag: break
                g=wp.get("gripper")
                if g in ("open","close"):
                    self._gripper_open_close(g)
                    # Interruptible delay
                    self._interruptible_sleep(delay)
                QApplication.processEvents()
        if self._estop_flag:
            self.prod_log.append("\n⛔ ABORTED BY E-STOP"); self.prod_status.setText("E-STOP")
        else:
            self.prod_log.append("\nDone."); self.prod_status.setText("Idle.")

    def _prod_run_pal(self, routine, cycles, speed, delay):
        cfg=routine.pallet_config; cols,rows,layers=cfg["cols"],cfg["rows"],cfg["layers"]
        pitch_x = cfg["obj_w"] + cfg["gap_x"]
        pitch_y = cfg["obj_d"] + cfg["gap_y"]
        layer_dz = cfg["obj_h"]
        pw=aw=None
        for wp in routine.waypoints:
            if wp["label"]=="place_primary": pw=wp
            if wp["label"]=="approach_place": aw=wp
        if not pw or not aw: self.prod_log.append("ERROR: missing WPs."); return
        pick_wps=[]
        for wp in routine.waypoints:
            if wp["label"]=="approach_place": break
            pick_wps.append(wp)
        home_wp=None
        for wp in routine.waypoints:
            if wp["label"]=="return_home": home_wp=wp; break
        grid_size=cols*rows*layers
        self.prod_log.append(f"  Grid {cols}x{rows}x{layers}={grid_size}  PitchX=+{pitch_x:.1f} PitchY=-{pitch_y:.1f} ΔZ=+{layer_dz:.1f}")
        for cycle_num in range(cycles):
            if self._estop_flag: break
            self.prod_log.append(f"\n=== Grid fill {cycle_num+1}/{cycles} ===")
            cell=0
            for L in range(layers):
                if self._estop_flag: break
                for R in range(rows):
                    if self._estop_flag: break
                    for C in range(cols):
                        if self._estop_flag: break
                        self.prod_log.append(f"  Cell {cell+1}/{grid_size} [C{C} R{R} L{L}]")
                        for wp in pick_wps:
                            if self._estop_flag: break
                            self._robot_move_to(wp["x"],wp["y"],wp["z"],speed=speed,move_type=wp.get("move_type","moveL"))
                            g=wp.get("gripper")
                            if g in ("open","close"):
                                self._gripper_open_close(g); self._interruptible_sleep(delay)
                            QApplication.processEvents()
                        if self._estop_flag: break
                        ox=pw["x"]+C*pitch_x; oy=pw["y"]-R*pitch_y; oz=pw["z"]+L*layer_dz; az_=aw["z"]+L*layer_dz
                        self._robot_move_to(ox,oy,az_,speed=speed,move_type="moveJ")
                        if self._estop_flag: break
                        self._robot_move_to(ox,oy,oz,speed=max(speed//3,10),move_type="moveL")
                        if self._estop_flag: break
                        self._gripper_open_close("open"); self._interruptible_sleep(delay)
                        if self._estop_flag: break
                        self._robot_move_to(ox,oy,az_,speed=max(speed//3,10),move_type="moveL")
                        if home_wp and not self._estop_flag:
                            self._robot_move_to(home_wp["x"],home_wp["y"],home_wp["z"],speed=speed,move_type="moveJ")
                        cell+=1; QApplication.processEvents()
        if self._estop_flag:
            self.prod_log.append(f"\n⛔ ABORTED BY E-STOP after {cell} cells"); self.prod_status.setText("E-STOP")
        else:
            self.prod_log.append(f"\n{cell*cycles} total cells done."); self.prod_status.setText("Idle.")

    def _prod_stop(self):
        """Production E-STOP: halt robot AND break the loop."""
        self._global_estop()
        self.prod_status.setText("E-STOP")
        self.prod_log.append("⛔ E-STOP SENT")

    def _interruptible_sleep(self, seconds):
        """Sleep in small increments, checking E-STOP between each."""
        elapsed = 0.0
        step = 0.05
        while elapsed < seconds:
            if self._estop_flag: return
            time.sleep(step)
            elapsed += step
            QApplication.processEvents()

    # ===== ROBOT HELPERS =====
    def _ensure_robot_ready(self):
        if not self.robot or self._estop_flag: return
        try: self.robot.clean_error(); self.robot.clean_warn(); self.robot.set_mode(0); self.robot.set_state(0)
        except: pass

    def _robot_move_to(self, x, y, z, roll=-180, pitch=0, yaw=0, speed=100, move_type="moveL"):
        if not self.robot or self._estop_flag: return
        try:
            self.robot.set_state(0)
            mvacc=1000 if move_type=="moveJ" else 500
            code=self.robot.set_position(x=x,y=y,z=z,roll=roll,pitch=pitch,yaw=yaw,speed=speed,mvacc=mvacc,wait=True)
            if code!=0 and not self._estop_flag:
                print(f"[Motion] code={code}, recovering..."); self._ensure_robot_ready()
                self.robot.set_position(x=x,y=y,z=z,roll=roll,pitch=pitch,yaw=yaw,speed=speed,mvacc=mvacc,wait=True)
        except Exception as e: print(f"[Motion] {e}")

    def _robot_move_to_guarded(self, x, y, z, roll=-180, pitch=0, yaw=0, speed=100, move_type="moveL", wait=True):
        if self._motion_busy or not self.robot or self._estop_flag: return
        self._motion_busy=True
        try:
            self.robot.set_state(0); mvacc=1000 if move_type=="moveJ" else 500
            self.robot.set_position(x=x,y=y,z=z,roll=roll,pitch=pitch,yaw=yaw,speed=speed,mvacc=mvacc,wait=wait)
        except Exception as e: print(f"[Jog] {e}")
        finally: self._motion_busy=False

    def _gripper_enable(self, enable):
        if not self.robot: return
        try:
            if enable:
                self.robot.set_tgpio_digital(ionum=0,value=1); time.sleep(0.5)
                self.robot.open_lite6_gripper(); time.sleep(1.0); print("[Gripper] Enabled")
            else:
                self.robot.stop_lite6_gripper(); time.sleep(0.3)
                self.robot.set_tgpio_digital(ionum=0,value=0); time.sleep(0.3); print("[Gripper] OFF")
        except Exception as e: print(f"[Gripper Enable] {e}")
    def _gripper_open_close(self, action):
        if not self.robot: return
        try:
            if action=="open": code=self.robot.open_lite6_gripper()
            else: code=self.robot.close_lite6_gripper()
            print(f"[Gripper] {action} -> {code}"); time.sleep(1.0)
        except Exception as e: print(f"[Gripper {action}] {e}")
    def _gripper_full_action(self, action):
        if action=="enable": self._gripper_enable(True)
        elif action=="off": self._gripper_enable(False)
        elif action in ("open","close"): self._gripper_open_close(action)

    # ===== PERSISTENCE =====
    def _save_routine(self, routine):
        with open(os.path.join(self.routines_dir,f"{routine.name}.json"),"w") as f: json.dump(routine.as_dict(),f,indent=2)
    def _load_routine(self, name):
        p=os.path.join(self.routines_dir,f"{name}.json")
        if not os.path.isfile(p): return None
        with open(p) as f: return Routine.from_dict(json.load(f))
    def _list_routines(self):
        r=[]
        if not os.path.isdir(self.routines_dir): return r
        for fn in sorted(os.listdir(self.routines_dir)):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(self.routines_dir,fn)) as f: d=json.load(f); r.append({"name":d["name"],"routine_type":d["routine_type"]})
                except: pass
        return r

    # ===== UTIL =====
    def _show_cv_on_label(self, img, label):
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); h,w,c=rgb.shape
        label.setPixmap(QPixmap.fromImage(QImage(rgb.data,w,h,c*w,QImage.Format_RGB888).copy()).scaled(label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def _mk_title(self, t): l=QLabel(t); l.setObjectName("title"); return l
    def _mk_sub(self, t): l=QLabel(t); l.setObjectName("subtitle"); l.setWordWrap(True); return l


if __name__=="__main__":
    app=QApplication(sys.argv); w=RobotTeachApp(); w.show(); sys.exit(app.exec())