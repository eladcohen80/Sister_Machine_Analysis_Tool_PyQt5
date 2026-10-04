import os
import sys
import csv
import copy
import numpy as np
import tifffile as tiff
import cv2
from skimage.measure import regionprops

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsPolygonItem,
    QGraphicsLineItem, QFileDialog, QStatusBar, QToolBar, QAction,
    QSizePolicy, QFrame, QGraphicsPathItem, QShortcut, QMessageBox
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPen, QBrush, QColor, QPolygonF, QPainter,
    QKeySequence, QPainterPath, QCursor, QFont
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject, QTimer

import tkinter as tk
from tkinter import filedialog

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
PIXEL_TO_MICRON = 1 / 16
CONFIG = {
    "contour_thickness": 1,
    "two_point_anchor_radius": 2,
    "two_point_roi_scale": 2.2,
    "two_point_refine_expand_iters": 2,
    "two_point_refine_shrink_iters": 1,
    "two_point_longitudinal_pad_px": 0.0,
}

# ─────────────────────────────────────────────────────────────
#  LOGIC LAYER  (identical to original step2_manual_correction.py)
# ─────────────────────────────────────────────────────────────

def get_cell_metrics(mask):
    props = regionprops(mask.astype(int))
    if not props:
        return 0, 0, 1.0
    prop = props[0]
    length = prop.major_axis_length
    perimeter = prop.perimeter
    area = prop.area
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    return length, perimeter, circularity


def ensure_cell_metrics(cell, frame):
    mask = cell.get("mask")
    if mask is None:
        return cell
    if "area" not in cell or cell["area"] is None:
        cell["area"] = int(np.sum(mask))
    if "intensity" not in cell or cell["intensity"] is None:
        cell["intensity"] = float(np.mean(frame[mask > 0])) if np.any(mask) else 0.0
    return cell


def build_frame_result(frame_idx, frame, chosen):
    frame_res = {"Frame": frame_idx + 1, "Left": None, "Right": None}
    for i, cell in enumerate(chosen):
        side = "Left" if i == 0 else "Right"
        cell = ensure_cell_metrics(cell, frame)
        length_px, perimeter_px, circularity = get_cell_metrics(cell["mask"])
        frame_res[side] = {
            "Length_Microns": length_px * PIXEL_TO_MICRON,
            "Perimeter_Microns": perimeter_px * PIXEL_TO_MICRON,
            "Circularity": circularity,
            "Area_Pixels": cell["area"],
            "Mean_Intensity": cell["intensity"],
            "Center_X": cell["cx"],
            "Center_Y": cell["cy"],
        }
    return frame_res


def draw_cell_outline(annotated, mask, color):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return
    cv2.drawContours(annotated, contours, -1, color, CONFIG["contour_thickness"], lineType=cv2.LINE_8)


def render_frame_state(frame_idx, frame, chosen):
    annotated = cv2.cvtColor(
        cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    for i, cell in enumerate(chosen):
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        draw_cell_outline(annotated, cell["mask"], color)
    return annotated, build_frame_result(frame_idx, frame, chosen)


def estimate_cell_radius(reference_cells, fallback_length):
    radii = []
    for cell in reference_cells:
        mask = cell.get("mask")
        if mask is None:
            continue
        props = regionprops(mask.astype(int))
        if not props:
            continue
        minor_axis = float(props[0].minor_axis_length)
        if minor_axis > 0:
            radii.append(minor_axis / 2.0)
    if radii:
        radius = float(np.median(radii))
    else:
        radius = max(5.0, fallback_length * 0.22)
    return int(max(4.0, min(radius, max(10.0, fallback_length * 0.45))))


def build_capsule_mask(frame_shape, p1, p2, radius):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=max(2, 2 * int(radius)))
    cv2.circle(mask, (x1, y1), int(radius), 1, -1)
    cv2.circle(mask, (x2, y2), int(radius), 1, -1)
    return mask


def compute_edge_map_from_frame(frame_u8):
    blurred = cv2.GaussianBlur(frame_u8, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def edge_guided_refine(mask, edge_map, max_expand, max_shrink):
    refined = mask.astype(np.uint8).copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max(0, int(max_expand))):
        inner_boundary = refined - cv2.erode(refined, kernel, iterations=1)
        outer_boundary = cv2.dilate(refined, kernel, iterations=1) - refined
        if not np.any(outer_boundary):
            break
        inner_score = float(np.mean(edge_map[inner_boundary > 0])) if np.any(inner_boundary) else 0.0
        outer_score = float(np.mean(edge_map[outer_boundary > 0])) if np.any(outer_boundary) else 0.0
        if outer_score >= (inner_score + 1.5):
            refined = cv2.dilate(refined, kernel, iterations=1)
        else:
            break
    for _ in range(max(0, int(max_shrink))):
        eroded = cv2.erode(refined, kernel, iterations=1)
        if not np.any(eroded):
            break
        current_boundary = refined - eroded
        inner_candidate_boundary = eroded - cv2.erode(eroded, kernel, iterations=1)
        current_score = float(np.mean(edge_map[current_boundary > 0])) if np.any(current_boundary) else 0.0
        inner_score = float(np.mean(edge_map[inner_candidate_boundary > 0])) if np.any(inner_candidate_boundary) else 0.0
        if inner_score >= (current_score + 2.0):
            refined = eroded
        else:
            break
    return refined.astype(np.uint8)


def constrain_mask_between_points(mask, p1, p2, longitudinal_pad_px=0.0):
    constrained = mask.astype(np.uint8).copy()
    if not np.any(constrained):
        return constrained
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    vx, vy = x2 - x1, y2 - y1
    axis_len = float(np.hypot(vx, vy))
    if axis_len < 1e-6:
        return constrained
    ux, uy = vx / axis_len, vy / axis_len
    ys, xs = np.where(constrained > 0)
    proj = (xs - x1) * ux + (ys - y1) * uy
    pad = max(0.0, float(longitudinal_pad_px))
    keep = (proj >= -pad) & (proj <= axis_len + pad)
    clipped = np.zeros_like(constrained, dtype=np.uint8)
    clipped[ys[keep], xs[keep]] = 1
    return clipped


def enforce_rounded_endcaps(mask, p1, p2, radius):
    rounded = mask.astype(np.uint8).copy()
    if not np.any(rounded):
        return rounded
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    vx, vy = x2 - x1, y2 - y1
    axis_len = float(np.hypot(vx, vy))
    if axis_len < 1e-6:
        return rounded
    u = np.array([vx / axis_len, vy / axis_len], dtype=np.float32)
    r = max(2.0, float(radius))
    c1 = np.array([x1, y1], dtype=np.float32) + u * r
    c2 = np.array([x2, y2], dtype=np.float32) - u * r
    cap_mask = np.zeros_like(rounded, dtype=np.uint8)
    cv2.circle(cap_mask, (int(round(c1[0])), int(round(c1[1]))), int(round(r)), 1, -1)
    cv2.circle(cap_mask, (int(round(c2[0])), int(round(c2[1]))), int(round(r)), 1, -1)
    cap_mask = constrain_mask_between_points(cap_mask, p1, p2, longitudinal_pad_px=0.0)
    h, w = rounded.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    proj = (xx - x1) * u[0] + (yy - y1) * u[1]
    tip_zone = max(2.0, 1.25 * r)
    endpoint_zone = (proj <= tip_zone) | (proj >= (axis_len - tip_zone))
    rounded[endpoint_zone] = 0
    rounded[(cap_mask > 0) & endpoint_zone] = 1
    rounded = cv2.morphologyEx(
        rounded, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return rounded.astype(np.uint8)


def create_cell_from_two_points(frame, p1, p2, reference_cells):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 8.0:
        return None
    radius = estimate_cell_radius(reference_cells, length)
    base_mask = build_capsule_mask(frame.shape, (x1, y1), (x2, y2), radius)
    roi_radius = int(max(radius + 4, radius * float(CONFIG["two_point_roi_scale"])))
    roi_mask = build_capsule_mask(frame.shape, (x1, y1), (x2, y2), roi_radius)
    frame_u8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(frame_u8, (5, 5), 0)
    roi_values = blurred[roi_mask > 0]
    seed_values = blurred[base_mask > 0]
    if roi_values.size == 0 or seed_values.size == 0:
        return None
    otsu_thresh, _ = cv2.threshold(
        roi_values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    seed_low = float(np.percentile(seed_values, 20))
    threshold_value = int(max(0, min(255, 0.6 * seed_low + 0.4 * otsu_thresh)))
    binary = ((blurred >= threshold_value) & (roi_mask > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n_labels, labels = cv2.connectedComponents(binary)
    custom_mask = np.zeros_like(binary)
    best_overlap = 0
    for label_idx in range(1, n_labels):
        comp = (labels == label_idx).astype(np.uint8)
        overlap = int(np.sum(comp * base_mask))
        if overlap > best_overlap:
            best_overlap = overlap
            custom_mask = comp
    if not np.any(custom_mask):
        custom_mask = base_mask.copy()
    edge_map = compute_edge_map_from_frame(frame_u8)
    custom_mask = edge_guided_refine(
        custom_mask, edge_map,
        CONFIG["two_point_refine_expand_iters"],
        CONFIG["two_point_refine_shrink_iters"],
    )
    custom_mask = cv2.morphologyEx(custom_mask, cv2.MORPH_CLOSE, kernel)
    custom_mask = cv2.morphologyEx(custom_mask, cv2.MORPH_OPEN, kernel)
    custom_mask = constrain_mask_between_points(
        custom_mask, (x1, y1), (x2, y2), CONFIG["two_point_longitudinal_pad_px"]
    )
    custom_mask = enforce_rounded_endcaps(custom_mask, (x1, y1), (x2, y2), radius)
    custom_mask = cv2.morphologyEx(custom_mask, cv2.MORPH_CLOSE, kernel)
    if not np.any(custom_mask):
        return None
    M = cv2.moments(custom_mask)
    if M["m00"] == 0:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    return {"mask": custom_mask, "cx": cx, "cy": cy, "area": int(np.sum(custom_mask)), "type": "custom"}


def smooth_mask_by_contour(mask, frame, forbidden_mask=None, smooth_window=9, evolve_iters=6):
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return mask_u8
    ys, xs = np.where(mask_u8 > 0)
    y0, y1 = int(np.min(ys)), int(np.max(ys))
    x0, x1 = int(np.min(xs)), int(np.max(xs))
    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    margin = int(max(6, 0.35 * max(box_h, box_w)))
    y0r = max(0, y0 - margin)
    y1r = min(mask_u8.shape[0], y1 + margin + 1)
    x0r = max(0, x0 - margin)
    x1r = min(mask_u8.shape[1], x1 + margin + 1)
    roi_mask = mask_u8[y0r:y1r, x0r:x1r].copy()
    roi_frame = frame[y0r:y1r, x0r:x1r]
    roi_forbidden = None
    if forbidden_mask is not None:
        roi_forbidden = forbidden_mask[y0r:y1r, x0r:x1r].astype(np.uint8)
    roi_u8 = cv2.normalize(roi_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi_blur = cv2.GaussianBlur(roi_u8, (5, 5), 0)
    seed_vals = roi_blur[roi_mask > 0]
    if seed_vals.size < 10:
        return mask_u8
    ring = cv2.dilate(roi_mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    ring = (ring > 0).astype(np.uint8)
    ring[roi_mask > 0] = 0
    bg_vals = roi_blur[ring > 0]
    seed_low = float(np.percentile(seed_vals, 25))
    if bg_vals.size > 0:
        bg_median = float(np.median(bg_vals))
        intensity_thr = int(np.clip(0.55 * seed_low + 0.45 * bg_median, 0, 255))
    else:
        intensity_thr = int(np.clip(seed_low * 0.9, 0, 255))
    grad_x = cv2.Sobel(roi_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi_blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    edge_norm = cv2.normalize(edge, None, 0.0, 1.0, cv2.NORM_MINMAX)
    intensity_norm = roi_blur.astype(np.float32) / 255.0
    score = 0.62 * intensity_norm + 0.38 * edge_norm
    original_area = int(np.sum(roi_mask))
    max_drift_px = int(np.clip(2.0 + 0.035 * np.sqrt(max(1, original_area)), 3, 6))
    outside = (1 - roi_mask).astype(np.uint8)
    dist_from_original = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    allowed_drift = (dist_from_original <= float(max_drift_px)).astype(np.uint8)
    support = cv2.dilate(roi_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    support = (support > 0).astype(np.uint8)
    support = (support & allowed_drift).astype(np.uint8)
    if roi_forbidden is not None:
        support[roi_forbidden > 0] = 0
    current = roi_mask.copy().astype(np.uint8)
    k3 = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max(1, int(evolve_iters))):
        dil = cv2.dilate(current, k3, iterations=1)
        ero = cv2.erode(current, k3, iterations=1)
        outer = (dil > 0) & (current == 0) & (support > 0)
        inner = (current > 0) & (ero == 0)
        add_cond = outer & ((roi_blur >= intensity_thr) | (score >= 0.50))
        if roi_forbidden is not None:
            add_cond = add_cond & (roi_forbidden == 0)
        remove_cond = inner & ((roi_blur < intensity_thr - 8) & (score < 0.42))
        current[add_cond] = 1
        current[remove_cond] = 0
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, k3)
    contours, _ = cv2.findContours(current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        pts = contour[:, 0, :].astype(np.float32)
        n_pts = len(pts)
        if n_pts >= 8:
            win = int(max(3, smooth_window))
            if win % 2 == 0:
                win += 1
            win = min(win, n_pts - 1 if n_pts % 2 == 0 else n_pts)
            if win >= 3:
                pad = win // 2
                extended = np.vstack([pts[-pad:], pts, pts[:pad]])
                kernel_1d = np.ones(win, dtype=np.float32) / float(win)
                x_smooth = np.convolve(extended[:, 0], kernel_1d, mode="valid")
                y_smooth = np.convolve(extended[:, 1], kernel_1d, mode="valid")
                smooth_pts = np.stack([x_smooth, y_smooth], axis=1)
                smooth_pts[:, 0] = np.clip(smooth_pts[:, 0], 0, current.shape[1] - 1)
                smooth_pts[:, 1] = np.clip(smooth_pts[:, 1], 0, current.shape[0] - 1)
                poly = np.round(smooth_pts).astype(np.int32).reshape((-1, 1, 2))
                current = np.zeros_like(current, dtype=np.uint8)
                cv2.fillPoly(current, [poly], 1)
    if roi_forbidden is not None:
        current[roi_forbidden > 0] = 0
    current = (current > 0).astype(np.uint8)
    current = current & allowed_drift
    current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, k3)
    current = cv2.morphologyEx(current, cv2.MORPH_OPEN, k3)
    n_labels, labels = cv2.connectedComponents(current)
    if n_labels > 1:
        best_label = 1
        best_overlap = -1
        for label_idx in range(1, n_labels):
            comp = (labels == label_idx).astype(np.uint8)
            overlap = int(np.sum(comp * roi_mask))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label_idx
        current = (labels == best_label).astype(np.uint8)
    refined = np.zeros_like(mask_u8)
    refined[y0r:y1r, x0r:x1r] = current
    if not np.any(refined):
        return mask_u8
    return refined.astype(np.uint8)

# ─────────────────────────────────────────────────────────────
#  STATE MANAGER  (Undo / Redo per frame)
# ─────────────────────────────────────────────────────────────

class FrameStateManager:
    def __init__(self, backup_data: dict):
        self._backup = backup_data
        self._undo: dict[int, list] = {}
        self._redo: dict[int, list] = {}

    def _init(self, idx):
        if idx not in self._undo:
            self._undo[idx] = [copy.deepcopy(self._backup[idx]["chosen"])]
            self._redo[idx] = []

    def get_chosen(self, idx) -> list:
        self._init(idx)
        return copy.deepcopy(self._undo[idx][-1])

    def commit(self, idx, new_chosen: list):
        self._init(idx)
        self._undo[idx].append(copy.deepcopy(new_chosen))
        self._redo[idx].clear()
        self._backup[idx]["chosen"] = copy.deepcopy(new_chosen)

    def undo(self, idx):
        self._init(idx)
        if len(self._undo[idx]) <= 1:
            return None
        self._redo[idx].append(self._undo[idx].pop())
        state = copy.deepcopy(self._undo[idx][-1])
        self._backup[idx]["chosen"] = copy.deepcopy(state)
        return state

    def redo(self, idx):
        self._init(idx)
        if not self._redo[idx]:
            return None
        state = self._redo[idx].pop()
        self._undo[idx].append(copy.deepcopy(state))
        self._backup[idx]["chosen"] = copy.deepcopy(state)
        return copy.deepcopy(state)

    def can_undo(self, idx) -> bool:
        self._init(idx)
        return len(self._undo[idx]) > 1

    def can_redo(self, idx) -> bool:
        self._init(idx)
        return bool(self._redo[idx])


# ─────────────────────────────────────────────────────────────
#  CELL CANVAS  (QGraphicsView with all interaction modes)
# ─────────────────────────────────────────────────────────────

class CellCanvas(QGraphicsView):
    """
    Modes
    -----
    view       : zoom + pan only
    click_add  : click detected cell to add it to chosen
    two_point  : click top then bottom to auto-build a capsule cell
    polygon    : click anchor points, right-click / double-click to finish
    delete     : click nearest chosen cell to remove it
    smooth     : click nearest chosen cell to smooth its contour
    """

    sig_cell_clicked    = pyqtSignal(float, float)   # view coords → image coords
    sig_two_point_click = pyqtSignal(float, float)
    sig_polygon_done    = pyqtSignal(list)            # list of (x,y) in image coords
    sig_status          = pyqtSignal(str)

    # ── inner anchor point item ──────────────────────────────
    class _Anchor(QGraphicsEllipseItem):
        R = 2
        def __init__(self, x, y, canvas):
            super().__init__(-CellCanvas._Anchor.R, -CellCanvas._Anchor.R,
                             2*CellCanvas._Anchor.R, 2*CellCanvas._Anchor.R)
            self.setPos(x, y)
            self._canvas = canvas
            self.setFlags(
                QGraphicsEllipseItem.ItemIsMovable |
                QGraphicsEllipseItem.ItemSendsGeometryChanges
            )
            self.setBrush(QBrush(QColor(255, 200, 0)))
            self.setPen(QPen(QColor(255, 255, 255), 1))
            self.setZValue(10)

        def itemChange(self, change, value):
            if change == QGraphicsEllipseItem.ItemPositionHasChanged:
                self._canvas._update_poly_line()
            return super().itemChange(change, value)

    # ── init ────────────────────────────────────────────────
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background: #1a1a1a; border: none;")

        self._img_item   = None
        self._overlays   = []
        self._anchors    = []
        self._poly_item  = None
        self._two_pt_dot = None

        self.mode = "view"
        self._space = False

    # ── image display ────────────────────────────────────────
    def set_image(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        q = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        px = QPixmap.fromImage(q)
        if self._img_item is None:
            self._img_item = QGraphicsPixmapItem(px)
            self._img_item.setZValue(0)
            self._scene.addItem(self._img_item)
        else:
            self._img_item.setPixmap(px)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._clear_overlays()
        self._clear_polygon()
        self._clear_two_pt_dot()

    def fit_to_window(self):
        if self._img_item:
            self.fitInView(self._img_item, Qt.KeepAspectRatio)

    # ── cell outlines ────────────────────────────────────────
    def draw_chosen(self, chosen: list, two_pt_anchor=None):
        """Draw outlines for chosen cells. Left=green, Right=red."""
        self._clear_overlays()
        colors = [QColor(0, 255, 0), QColor(255, 60, 60)]
        for i, cell in enumerate(chosen):
            mask = cell.get("mask")
            if mask is None:
                continue
            color = colors[i % len(colors)]
            pen = QPen(color, 1.5)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                pts = c[:, 0, :]
                poly = QPolygonF([QPointF(float(p[0]), float(p[1])) for p in pts])
                item = self._scene.addPolygon(poly, pen, QBrush(Qt.transparent))
                item.setZValue(2)
                self._overlays.append(item)

        self._clear_two_pt_dot()
        if two_pt_anchor is not None:
            x, y = two_pt_anchor
            r = 2
            dot = self._scene.addEllipse(
                x - r, y - r, 2*r, 2*r,
                QPen(QColor(255, 255, 0), 1.5),
                QBrush(QColor(255, 220, 0, 180))
            )
            dot.setZValue(8)
            self._overlays.append(dot)
            self._two_pt_dot = dot

    def draw_detected_outlines(self, all_cells: list):
        """Draw faint outlines for all detected (not-chosen) cells."""
        pen = QPen(QColor(120, 120, 255, 140), 1, Qt.DotLine)
        for cell in all_cells:
            mask = cell.get("mask")
            if mask is None:
                continue
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                pts = c[:, 0, :]
                poly = QPolygonF([QPointF(float(p[0]), float(p[1])) for p in pts])
                item = self._scene.addPolygon(poly, pen, QBrush(Qt.transparent))
                item.setZValue(1)
                self._overlays.append(item)

    # ── helpers ──────────────────────────────────────────────
    def _clear_overlays(self):
        for it in self._overlays:
            self._scene.removeItem(it)
        self._overlays.clear()
        self._two_pt_dot = None

    def _clear_two_pt_dot(self):
        pass  # handled inside _clear_overlays / draw_chosen

    def _clear_polygon(self):
        for a in self._anchors:
            self._scene.removeItem(a)
        self._anchors.clear()
        if self._poly_item:
            self._scene.removeItem(self._poly_item)
            self._poly_item = None

    def _update_poly_line(self):
        if len(self._anchors) < 2:
            return
        pts = [a.pos() for a in self._anchors] + [self._anchors[0].pos()]
        poly = QPolygonF(pts)
        if self._poly_item is None:
            pen = QPen(QColor(0, 220, 0), 1.5, Qt.DashLine)
            self._poly_item = self._scene.addPolygon(poly, pen, QBrush(Qt.transparent))
            self._poly_item.setZValue(5)
        else:
            self._poly_item.setPolygon(poly)

    def _to_scene(self, qpos):
        return self.mapToScene(qpos)

    def _finalize_polygon(self):
        if len(self._anchors) < 3:
            self._clear_polygon()
            self.sig_status.emit("Polygon cancelled (need ≥ 3 points)")
            return
        pts = [(int(a.pos().x()), int(a.pos().y())) for a in self._anchors]
        self._clear_polygon()
        self.sig_polygon_done.emit(pts)

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "polygon":
            self._clear_polygon()
        cursors = {
            "view":       Qt.ArrowCursor,
            "click_add":  Qt.PointingHandCursor,
            "two_point":  Qt.CrossCursor,
            "polygon":    Qt.CrossCursor,
            "delete":     Qt.ForbiddenCursor,
            "smooth":     Qt.PointingHandCursor,
        }
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))

    # ── mouse events ─────────────────────────────────────────
    def mousePressEvent(self, ev):
        sp = self._to_scene(ev.pos())
        x, y = sp.x(), sp.y()

        if self._space and ev.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(ev)
            return

        if ev.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = ev.__class__(ev.type(), ev.pos(), Qt.LeftButton,
                                Qt.LeftButton, ev.modifiers())
            super().mousePressEvent(fake)
            return

        if self.mode == "polygon":
            if ev.button() == Qt.LeftButton:
                anchor = CellCanvas._Anchor(x, y, self)
                self._scene.addItem(anchor)
                self._anchors.append(anchor)
                self._update_poly_line()
            elif ev.button() == Qt.RightButton:
                self._finalize_polygon()

        elif self.mode == "two_point":
            if ev.button() == Qt.LeftButton:
                self.sig_two_point_click.emit(x, y)

        elif self.mode in ("click_add", "delete", "smooth"):
            if ev.button() == Qt.LeftButton:
                self.sig_cell_clicked.emit(x, y)

        else:
            super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if self.mode == "polygon" and ev.button() == Qt.LeftButton:
            self._finalize_polygon()
        else:
            super().mouseDoubleClickEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() in (Qt.LeftButton, Qt.MiddleButton):
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Space:
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        elif ev.key() == Qt.Key_Escape and self.mode == "polygon":
            self._clear_polygon()
            self.sig_status.emit("Polygon cancelled")
        else:
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Space:
            self._space = False
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(cursors := {
                "view":       Qt.ArrowCursor,
                "click_add":  Qt.PointingHandCursor,
                "two_point":  Qt.CrossCursor,
                "polygon":    Qt.CrossCursor,
                "delete":     Qt.ForbiddenCursor,
                "smooth":     Qt.PointingHandCursor,
            }.get(self.mode, Qt.ArrowCursor))
        else:
            super().keyReleaseEvent(ev)

# ─────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────

class CorrectionWindow(QMainWindow):
    SIDE_W = 260

    def __init__(self, images, backup_data, tif_filename, run_output_folder, data_file):
        super().__init__()
        self._images         = images
        self._backup         = backup_data
        self._tif_filename   = tif_filename
        self._out_folder     = run_output_folder
        self._data_file      = data_file
        self._mgr            = FrameStateManager(backup_data)
        self._frame_idx      = 0
        self._chosen         = self._mgr.get_chosen(0)
        self._brightness     = 1.0
        self._two_pt_anchor  = None
        self._frames_results = [None] * len(images)
        self._show_detected  = True

        self._build_ui()
        self._build_shortcuts()
        self._goto_frame(0, fit=True)

        self.setWindowTitle(f"Cell Correction — {tif_filename}")
        icon_path = os.path.join(os.path.dirname(__file__), "sister_machine_logo.ico")
        if os.path.exists(icon_path):
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))

    # ── UI construction ──────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_h = QHBoxLayout(central)
        root_h.setContentsMargins(0, 0, 0, 0)
        root_h.setSpacing(0)

        # ── canvas ──
        self._canvas = CellCanvas()
        self._canvas.sig_cell_clicked.connect(self._on_cell_click)
        self._canvas.sig_two_point_click.connect(self._on_two_point_click)
        self._canvas.sig_polygon_done.connect(self._on_polygon_done)
        self._canvas.sig_status.connect(self._status)
        root_h.addWidget(self._canvas, 1)

        # ── side panel ──
        side = QFrame()
        side.setFixedWidth(self.SIDE_W)
        side.setStyleSheet("background:#2b2b2b; color:white;")
        sv = QVBoxLayout(side)
        sv.setContentsMargins(10, 14, 10, 10)
        sv.setSpacing(6)
        root_h.addWidget(side)

        def lbl(text, color="#ffffff", size=13, bold=False):
            l = QLabel(text)
            weight = "bold" if bold else "normal"
            l.setStyleSheet(f"color:{color}; font-size:{size}px; font-weight:{weight};")
            l.setWordWrap(True)
            return l

        def sep():
            f = QFrame()
            f.setFrameShape(QFrame.HLine)
            f.setStyleSheet("color:#555;")
            return f

        # frame info
        self._lbl_frame = lbl("Frame 1 / 1", "#ffffff", 15, True)
        sv.addWidget(self._lbl_frame)
        sv.addWidget(sep())

        # frame slider
        sv.addWidget(lbl("Frame", "#aaaaaa", 11))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(1)
        self._slider.setMaximum(max(1, len(self._images)))
        self._slider.setValue(1)
        self._slider.setTickInterval(1)
        self._slider.valueChanged.connect(self._on_slider)
        sv.addWidget(self._slider)
        sv.addWidget(sep())

        # nav buttons
        nav = QHBoxLayout()
        self._btn_prev = self._make_btn("◀ Prev  [A]", self._prev_frame, "#3a7bd5")
        self._btn_next = self._make_btn("Next  [D] ▶", self._next_frame, "#3a7bd5")
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._btn_next)
        sv.addLayout(nav)
        sv.addWidget(sep())

        # brightness
        sv.addWidget(lbl("Brightness", "#aaaaaa", 11))
        bri_row = QHBoxLayout()
        bri_row.addWidget(self._make_btn("−", self._brightness_down, "#555", w=36))
        self._lbl_bri = lbl("1.0×", "#ffffff", 12)
        self._lbl_bri.setAlignment(Qt.AlignCenter)
        bri_row.addWidget(self._lbl_bri, 1)
        bri_row.addWidget(self._make_btn("+", self._brightness_up, "#555", w=36))
        sv.addLayout(bri_row)
        sv.addWidget(sep())

        # ADD SECTION
        sv.addWidget(lbl("── Add Cell ──", "#44cc88", 12, True))

        # click-add (detected cells)
        self._btn_click = self._make_btn("🖱 Click Detected Cell", self._mode_click_add, "#226644")
        self._btn_click.setToolTip("Click on any detected cell (blue outline) to add it to chosen")
        sv.addWidget(self._btn_click)

        # two-point
        self._btn_two = self._make_btn("⊕ Two-Point Capsule  [T]", self._mode_two_point, "#226644")
        self._btn_two.setToolTip("Click top endpoint, then bottom endpoint.\nAuto-builds a capsule mask.")
        sv.addWidget(self._btn_two)

        # polygon
        self._btn_poly = self._make_btn("✏ Draw Polygon  [P]", self._mode_polygon, "#226644")
        self._btn_poly.setToolTip(
            "Click to place anchor points.\nDouble-click or Right-click to close.\n"
            "You can drag anchors before closing."
        )
        sv.addWidget(self._btn_poly)

        sv.addWidget(sep())

        # EDIT SECTION
        sv.addWidget(lbl("── Edit Cell ──", "#ee9944", 12, True))

        self._btn_smooth = self._make_btn("〰 Smooth Mask  [S]", self._mode_smooth, "#664422")
        self._btn_smooth.setToolTip("Click a chosen cell to smooth its contour (edge-guided)")
        sv.addWidget(self._btn_smooth)

        self._btn_del = self._make_btn("✖ Delete Cell  [X]", self._mode_delete, "#882222")
        self._btn_del.setToolTip("Click a chosen cell to remove it")
        sv.addWidget(self._btn_del)

        sv.addWidget(sep())

        # undo / redo
        ur = QHBoxLayout()
        self._btn_undo = self._make_btn("↩ Undo [Z]", self._do_undo, "#555")
        self._btn_redo = self._make_btn("↪ Redo", self._do_redo, "#555")
        ur.addWidget(self._btn_undo)
        ur.addWidget(self._btn_redo)
        sv.addLayout(ur)
        sv.addWidget(sep())

        # toggle detected outlines
        self._btn_det = self._make_btn("👁 Detected Cells: ON", self._toggle_detected, "#444")
        sv.addWidget(self._btn_det)

        sv.addWidget(sep())

        # chosen cells info
        self._lbl_chosen = lbl("Chosen: 0 cell(s)", "#aaffaa", 11)
        sv.addWidget(self._lbl_chosen)
        self._lbl_hint = lbl("", "#ffcc44", 11)
        sv.addWidget(self._lbl_hint)

        sv.addStretch()

        # save & close
        self._make_btn("💾 Save & Close  [Esc]", self._save_and_close, "#555533", parent_layout=sv, h=36)

        # status bar
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet("color:#aaa; font-size:11px;")
        self.setStatusBar(self._status_bar)

        # geometry
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1400, screen.width() - 40)
        h = min(820, screen.height() - 60)
        self.resize(w, h)
        self.move(
            screen.x() + (screen.width() - w) // 2,
            screen.y() + (screen.height() - h) // 2
        )

    def _make_btn(self, text, slot, color="#3a3a3a", w=None, h=30, parent_layout=None):
        b = QPushButton(text)
        b.setFixedHeight(h)
        if w:
            b.setFixedWidth(w)
        b.setStyleSheet(
            f"QPushButton {{background:{color}; color:white; border-radius:4px; font-size:12px;}}"
            f"QPushButton:hover {{background:{color}dd;}}"
            f"QPushButton:pressed {{background:{color}99;}}"
        )
        b.clicked.connect(slot)
        if parent_layout:
            parent_layout.addWidget(b)
        return b

    def _build_shortcuts(self):
        QShortcut(QKeySequence("D"),     self, self._next_frame)
        QShortcut(QKeySequence("Space"), self, self._next_frame)
        QShortcut(QKeySequence("A"),     self, self._prev_frame)
        QShortcut(QKeySequence("Z"),     self, self._do_undo)
        QShortcut(QKeySequence("T"),     self, self._mode_two_point)
        QShortcut(QKeySequence("P"),     self, self._mode_polygon)
        QShortcut(QKeySequence("S"),     self, self._mode_smooth)
        QShortcut(QKeySequence("X"),     self, self._mode_delete)
        QShortcut(QKeySequence("Escape"), self, self._save_and_close)
        QShortcut(QKeySequence("+"),     self, self._brightness_up)
        QShortcut(QKeySequence("-"),     self, self._brightness_down)

    # ── navigation ───────────────────────────────────────────
    def _on_slider(self, val):
        new_idx = val - 1
        if new_idx != self._frame_idx:
            self._goto_frame(new_idx)

    def _prev_frame(self):
        if self._frame_idx > 0:
            self._goto_frame(self._frame_idx - 1)

    def _next_frame(self):
        if self._frame_idx < len(self._images) - 1:
            self._goto_frame(self._frame_idx + 1)

    def _goto_frame(self, idx, fit=False):
        # commit current state
        self._mgr.commit(self._frame_idx, self._chosen)

        self._frame_idx = idx
        self._chosen    = self._mgr.get_chosen(idx)
        self._two_pt_anchor = None

        # reset mode to view
        self._set_mode("view")

        self._slider.blockSignals(True)
        self._slider.setValue(idx + 1)
        self._slider.blockSignals(False)

        self._refresh_canvas(fit=fit)
        self._update_info()
        self._status(f"Frame {idx+1}/{len(self._images)}")

    # ── rendering ────────────────────────────────────────────
    def _current_frame_u8(self):
        frame = self._images[self._frame_idx]
        f8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self._brightness != 1.0:
            f8 = cv2.convertScaleAbs(f8, alpha=self._brightness, beta=0)
        return cv2.cvtColor(f8, cv2.COLOR_GRAY2RGB)

    def _refresh_canvas(self, fit=False):
        rgb = self._current_frame_u8()
        self._canvas.set_image(rgb)

        self._canvas.draw_chosen(self._chosen, self._two_pt_anchor)
        if self._show_detected:
            all_cells = self._backup[self._frame_idx].get("all", [])
            chosen_set = {id(c) for c in self._chosen}
            not_chosen = [c for c in all_cells if id(c) not in chosen_set]
            self._canvas.draw_detected_outlines(not_chosen)

        if fit:
            QTimer.singleShot(50, self._canvas.fit_to_window)

    def _update_info(self):
        self._lbl_frame.setText(f"Frame {self._frame_idx+1} / {len(self._images)}")
        n = len(self._chosen)
        sides = ["Left", "Right"]
        parts = [f"{sides[i]}" for i in range(min(n, 2))]
        self._lbl_chosen.setText(f"Chosen: {n} cell(s)  {', '.join(parts)}")

    def _status(self, msg: str):
        self._status_bar.showMessage(msg)

    # ── mode management ──────────────────────────────────────
    _MODE_BUTTONS = {}

    def _set_mode(self, mode: str, hint: str = ""):
        self._canvas.set_mode(mode)
        self._lbl_hint.setText(hint)
        # reset two_pt_anchor when leaving two_point
        if mode != "two_point":
            self._two_pt_anchor = None
            self._canvas.draw_chosen(self._chosen, None)
            self._redraw_detected()
        self._update_button_states(mode)

    def _update_button_states(self, active_mode: str):
        btn_map = {
            "click_add": self._btn_click,
            "two_point":  self._btn_two,
            "polygon":    self._btn_poly,
            "smooth":     self._btn_smooth,
            "delete":     self._btn_del,
        }
        for m, b in btn_map.items():
            highlight = "border: 2px solid #ffcc00;" if m == active_mode else ""
            b.setStyleSheet(
                b.styleSheet().split("border:")[0].rstrip(";") + f"; {highlight}"
            )

    def _mode_click_add(self):
        self._set_mode("click_add", "Click a detected cell (blue outline) to add it")

    def _mode_two_point(self):
        if self._canvas.mode == "two_point":
            self._set_mode("view")
            self._status("Two-point mode OFF")
        else:
            self._two_pt_anchor = None
            self._set_mode("two_point", "Click the TOP endpoint of the new cell")
            self._status("Two-point mode: click top endpoint")

    def _mode_polygon(self):
        self._set_mode("polygon", "Click to place vertices. Right-click or double-click to finish.")
        self._status("Polygon mode: place vertices, right-click to close")

    def _mode_smooth(self):
        self._set_mode("smooth", "Click a chosen cell to smooth its mask")

    def _mode_delete(self):
        self._set_mode("delete", "Click a chosen cell to remove it")

    # ── brightness ───────────────────────────────────────────
    def _brightness_up(self):
        self._brightness = min(self._brightness + 0.2, 3.0)
        self._lbl_bri.setText(f"{self._brightness:.1f}×")
        self._refresh_canvas()

    def _brightness_down(self):
        self._brightness = max(self._brightness - 0.2, 0.4)
        self._lbl_bri.setText(f"{self._brightness:.1f}×")
        self._refresh_canvas()

    # ── toggle detected outlines ─────────────────────────────
    def _redraw_detected(self):
        """Re-draw blue detected outlines after a direct draw_chosen call."""
        if self._show_detected:
            all_cells = self._backup[self._frame_idx].get("all", [])
            chosen_set = {id(c) for c in self._chosen}
            not_chosen = [c for c in all_cells if id(c) not in chosen_set]
            self._canvas.draw_detected_outlines(not_chosen)

    def _toggle_detected(self):
        self._show_detected = not self._show_detected
        state = "ON" if self._show_detected else "OFF"
        self._btn_det.setText(f"👁 Detected Cells: {state}")
        self._refresh_canvas()

    # ── undo / redo ──────────────────────────────────────────
    def _do_undo(self):
        state = self._mgr.undo(self._frame_idx)
        if state is not None:
            self._chosen = state
            self._refresh_canvas()
            self._update_info()
            self._status("Undo")
        else:
            self._status("Nothing to undo")

    def _do_redo(self):
        state = self._mgr.redo(self._frame_idx)
        if state is not None:
            self._chosen = state
            self._refresh_canvas()
            self._update_info()
            self._status("Redo")
        else:
            self._status("Nothing to redo")

    # ─────────────────────────────────────────────────────────
    #  CELL INTERACTION HANDLERS
    # ─────────────────────────────────────────────────────────

    def _nearest_chosen(self, x, y, max_dist=80):
        best, best_d = None, float("inf")
        for i, cell in enumerate(self._chosen):
            d = np.hypot(cell["cx"] - x, cell["cy"] - y)
            if d < best_d:
                best_d = d
                best = i
        if best is not None and best_d <= max_dist:
            return best, best_d
        return None, float("inf")

    def _nearest_detected(self, x, y, max_dist=80):
        all_cells = self._backup[self._frame_idx].get("all", [])
        best, best_d = None, float("inf")
        for cell in all_cells:
            # also check if pixel is inside the mask
            mask = cell.get("mask")
            if mask is not None:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                    return cell, 0.0
            d = np.hypot(cell["cx"] - x, cell["cy"] - y)
            if d < best_d:
                best_d = d
                best = cell
        if best is not None and best_d <= max_dist:
            return best, best_d
        return None, float("inf")

    def _on_cell_click(self, x, y):
        mode = self._canvas.mode
        frame = self._images[self._frame_idx]

        # ── DELETE ──────────────────────────────────────────
        if mode == "delete":
            idx, dist = self._nearest_chosen(x, y)
            if idx is not None:
                self._mgr.commit(self._frame_idx, self._chosen)
                del self._chosen[idx]
                self._mgr.commit(self._frame_idx, self._chosen)
                self._refresh_canvas()
                self._update_info()
                self._status(f"Cell deleted (dist={dist:.1f})")
            else:
                self._status("No nearby chosen cell to delete")
            self._set_mode("view")
            return

        # ── SMOOTH ──────────────────────────────────────────
        if mode == "smooth":
            idx, dist = self._nearest_chosen(x, y)
            if idx is None:
                self._status("No nearby chosen cell to smooth")
                self._set_mode("view")
                return
            target_cell = self._chosen[idx]
            target_mask_u8 = target_cell["mask"].astype(np.uint8)
            forbidden = np.zeros(frame.shape[:2], dtype=np.uint8)
            for oi, oc in enumerate(self._chosen):
                if oi != idx:
                    forbidden = np.maximum(forbidden, oc["mask"].astype(np.uint8))
            all_cells = self._backup[self._frame_idx].get("all", [])
            for det in all_cells:
                dm = det.get("mask")
                if dm is None:
                    continue
                dm_u8 = dm.astype(np.uint8)
                inter = int(np.sum((dm_u8 > 0) & (target_mask_u8 > 0)))
                if inter > 0:
                    union = int(np.sum((dm_u8 > 0) | (target_mask_u8 > 0)))
                    if union > 0 and inter / union > 0.35:
                        continue
                forbidden = np.maximum(forbidden, dm_u8)
            forbidden = cv2.dilate(forbidden, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
            forbidden[target_mask_u8 > 0] = 0
            smoothed = smooth_mask_by_contour(target_cell["mask"], frame, forbidden_mask=forbidden)
            if np.any(smoothed):
                self._mgr.commit(self._frame_idx, self._chosen)
                target_cell["mask"] = smoothed
                M = cv2.moments(smoothed)
                if M["m00"] != 0:
                    target_cell["cx"] = int(M["m10"] / M["m00"])
                    target_cell["cy"] = int(M["m01"] / M["m00"])
                ensure_cell_metrics(target_cell, frame)
                self._mgr.commit(self._frame_idx, self._chosen)
                self._refresh_canvas()
                self._update_info()
                self._status("Mask smoothed")
            else:
                self._status("Smoothing failed: empty result")
            self._set_mode("view")
            return

        # ── CLICK_ADD ────────────────────────────────────────
        if mode == "click_add":
            # First check: did user click directly inside a chosen cell? → select its mask
            for cell in self._chosen:
                mask = cell.get("mask")
                if mask is not None:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                        self._status("Cell already in chosen list")
                        self._set_mode("view")
                        return

            # Then: find nearest detected cell
            det_cell, dist = self._nearest_detected(x, y)
            if det_cell is not None:
                already = any(
                    c["cx"] == det_cell["cx"] and c["cy"] == det_cell["cy"]
                    for c in self._chosen
                )
                if not already:
                    self._mgr.commit(self._frame_idx, self._chosen)
                    new_cell = copy.deepcopy(det_cell)
                    ensure_cell_metrics(new_cell, frame)
                    self._chosen.append(new_cell)
                    self._chosen.sort(key=lambda c: c["cx"])
                    self._mgr.commit(self._frame_idx, self._chosen)
                    self._refresh_canvas()
                    self._update_info()
                    self._status(f"Detected cell added (dist={dist:.1f})")
                else:
                    self._status("Cell already in chosen list")
            else:
                self._status("No nearby detected cell found")
            self._set_mode("view")
            return

    # ── TWO-POINT ────────────────────────────────────────────
    def _on_two_point_click(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        frame = self._images[self._frame_idx]

        # ── If clicked on a chosen cell with existing mask → select/highlight it
        for cell in self._chosen:
            mask = cell.get("mask")
            if mask is not None:
                if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
                    self._status(f"This cell is already in the chosen list")
                    return  # stay in two-point mode to allow next click

        if self._two_pt_anchor is None:
            self._two_pt_anchor = (ix, iy)
            self._canvas.draw_chosen(self._chosen, self._two_pt_anchor)
            self._redraw_detected()
            self._status("Top endpoint set — now click the BOTTOM endpoint")
            self._lbl_hint.setText("Now click the BOTTOM endpoint of the cell")
        else:
            p1 = self._two_pt_anchor
            p2 = (ix, iy)
            self._two_pt_anchor = None
            self._set_mode("view")

            ref = list(self._chosen) + list(self._backup[self._frame_idx].get("all", []))
            new_cell = create_cell_from_two_points(frame, p1, p2, ref)
            if new_cell is not None:
                self._mgr.commit(self._frame_idx, self._chosen)
                ensure_cell_metrics(new_cell, frame)
                self._chosen.append(new_cell)
                self._chosen.sort(key=lambda c: c["cx"])
                self._mgr.commit(self._frame_idx, self._chosen)
                self._refresh_canvas()
                self._update_info()
                self._status("New cell added via two-point capsule")
            else:
                self._status("Two-point failed: points too close or invalid")

    # ── POLYGON ──────────────────────────────────────────────
    def _on_polygon_done(self, pts):
        if len(pts) < 3:
            self._status("Polygon cancelled: need at least 3 points")
            self._set_mode("view")
            return

        frame = self._images[self._frame_idx]
        custom_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(custom_mask, [pts_arr], 1)

        M = cv2.moments(custom_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = pts[0]

        new_cell = {
            "mask": custom_mask,
            "cx": cx, "cy": cy,
            "area": int(np.sum(custom_mask)),
            "intensity": float(np.mean(frame[custom_mask > 0])) if np.any(custom_mask) else 0.0,
            "type": "custom",
        }
        self._mgr.commit(self._frame_idx, self._chosen)
        self._chosen.append(new_cell)
        self._chosen.sort(key=lambda c: c["cx"])
        self._mgr.commit(self._frame_idx, self._chosen)
        self._refresh_canvas()
        self._update_info()
        self._status("Polygon cell added")
        self._set_mode("view")

    # ── save & close ─────────────────────────────────────────
    def _save_and_close(self):
        self._mgr.commit(self._frame_idx, self._chosen)
        self._status("Saving…")
        images   = self._images
        backup   = self._backup

        # recompute all frame results
        frames_results  = []
        frames_outputs  = []
        for fi, frame in enumerate(images):
            chosen = backup[fi]["chosen"]
            rendered, result = render_frame_state(fi, frame, chosen)
            frames_results.append(result)
            frames_outputs.append(rendered)

        base_name = self._tif_filename.replace(".tif", "").replace(".TIF", "")
        np.save(os.path.join(self._out_folder, self._data_file), backup)

        h, w = images[0].shape[:2]
        valid = [img if img is not None else np.zeros((h, w, 3), dtype=np.uint8) for img in frames_outputs]
        tiff.imwrite(
            os.path.join(self._out_folder, f"{base_name}_corrected_analyzed.tif"),
            np.array(valid), photometric="rgb"
        )

        with open(os.path.join(self._out_folder, f"{base_name}_FINAL_corrected_data.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["Frame","Length Left","Area Left","Intensity Left",
                         "Length Right","Area Right","Intensity Right"])
            for d in frames_results:
                if d is not None:
                    left  = d["Left"]  or {}
                    right = d["Right"] or {}
                    wr.writerow([
                        d["Frame"],
                        left.get("Length_Microns", 0),  left.get("Area_Pixels", 0),  left.get("Mean_Intensity", 0),
                        right.get("Length_Microns", 0), right.get("Area_Pixels", 0), right.get("Mean_Intensity", 0),
                    ])

        print(f"Saved to: {self._out_folder}")
        self._status("Saved ✓")
        self.close()

# ─────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT  (called from launcher / step2)
# ─────────────────────────────────────────────────────────────

def run_correction_qt(images, backup_data, tif_filename, run_output_folder, data_file):
    """
    Drop-in replacement for the tkinter-based run_correction().
    Parameters match the original step2_manual_correction.py API.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    win = CorrectionWindow(images, backup_data, tif_filename, run_output_folder, data_file)
    win.show()
    app.exec_()
    return backup_data


# ─────────────────────────────────────────────────────────────
#  LAUNCHER ENTRY POINT  (called from launcher with no args)
# ─────────────────────────────────────────────────────────────

def run_correction_qt_standalone():
    """
    Opens file-dialogs (TIF folder + NPY file), loads data,
    then launches the Qt correction window.
    Called from launcher.py with no arguments.
    After each file is saved & closed, automatically prompts for the next file.
    Cancel the NPY dialog to exit.
    """
    # 1. Select original TIF folder ONCE
    root = tk.Tk()
    root.withdraw()
    original_tif_folder = filedialog.askdirectory(title="Select Original TIF Folder")
    root.destroy()
    if not original_tif_folder:
        print("No folder selected. Cancelled.")
        return

    while True:
        # 2. Select next segmentation NPY file (Cancel = done)
        root2 = tk.Tk()
        root2.withdraw()
        data_file_path = filedialog.askopenfilename(
            title="Select segmentation data file to edit  —  Cancel to exit",
            filetypes=[("NPY files", "*_segmentation_data.npy"), ("All NPY files", "*.npy")]
        )
        root2.destroy()

        if not data_file_path:
            print("No file selected. Closing.")
            break

        base_folder = os.path.dirname(data_file_path)
        normalized_parts = [
            part.lower()
            for part in os.path.normpath(base_folder).split(os.sep)
            if part
        ]
        if "corrected_results" in normalized_parts:
            run_output_folder = base_folder
        else:
            run_output_folder = os.path.join(base_folder, "corrected_results")
            os.makedirs(run_output_folder, exist_ok=True)

        data_file = os.path.basename(data_file_path)
        tif_filename = data_file.replace("_segmentation_data.npy", "")
        orig_path = os.path.join(original_tif_folder, tif_filename)

        if not os.path.exists(orig_path):
            from tkinter import messagebox
            root3 = tk.Tk()
            root3.withdraw()
            messagebox.showwarning(
                "File Not Found",
                f"Could not find the matching TIF file:\n{tif_filename}\n"
                f"in folder:\n{original_tif_folder}",
                parent=root3
            )
            root3.destroy()
            continue

        # 3. Load TIF
        with tiff.TiffFile(orig_path) as tif_file:
            images_arr = tif_file.asarray()
        if images_arr.ndim == 2:
            images_arr = np.expand_dims(images_arr, axis=0)
        images = [images_arr[i] for i in range(len(images_arr))]

        # 4. Load NPY backup
        backup_data = np.load(data_file_path, allow_pickle=True).item()

        # 5. Launch Qt window
        # 6. Launch window — after Save & Close it returns here automatically
        run_correction_qt(images, backup_data, tif_filename, run_output_folder, data_file)
        # loop back → ask for next NPY file


# ─────────────────────────────────────────────────────────────
#  STANDALONE LAUNCH  (python step2_manual_correction_qt.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    tif_path = filedialog.askopenfilename(
        title="Select TIF file",
        filetypes=[("TIFF files", "*.tif *.TIF"), ("All files", "*.*")]
    )
    if not tif_path:
        print("Cancelled")
        sys.exit(0)

    try:
        stack = tiff.imread(tif_path)
    except Exception as e:
        print(f"Could not read TIF: {e}")
        sys.exit(1)

    if stack.ndim == 2:
        stack = stack[np.newaxis]
    elif stack.ndim == 3 and stack.shape[2] <= 4:
        # single colour image
        stack = stack[np.newaxis]
    elif stack.ndim == 3:
        # (frames, H, W) already
        pass

    # minimal backup_data structure
    backup = {}
    for i in range(len(stack)):
        frame = stack[i].squeeze()
        backup[i] = {"chosen": [], "all": []}

    out_folder = os.path.dirname(tif_path) or "."
    tif_filename = os.path.basename(tif_path)
    data_file = tif_filename.replace(".tif", "_backup.npy").replace(".TIF", "_backup.npy")
    images = [stack[i].squeeze() for i in range(len(stack))]

    run_correction_qt(images, backup, tif_filename, out_folder, data_file)
