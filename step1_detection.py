import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import tifffile as tiff
import cv2
import torch
import csv
from skimage.measure import regionprops

try:
    from cellpose import models as cellpose_models
    CELLPOSE_AVAILABLE = True
except Exception:
    cellpose_models = None
    CELLPOSE_AVAILABLE = False



# --- הגדרות ---
PIXEL_TO_MICRON = 1 / 16
CONFIG = {
    "gamma": 0.6,
    "min_cell_area": 150,
    "cellprob_threshold": 0.0,
    "flow_threshold": 0.4,
    "contour_thickness": 1,
    "boundary_max_expand_px": 5,
    "boundary_max_shrink_px": 3,
    "boundary_edge_gain": 1.06,
    "boundary_edge_delta": 2.5,
    "boundary_min_edge_strength": 12.0,
    "boundary_edge_percentile": 75,
    "boundary_min_support_fraction": 0.22,
    "boundary_shrink_gain": 1.24,
    "boundary_shrink_delta": 5.0,
    "boundary_shrink_max_support_fraction": 0.20,
    "boundary_length_pad_px": 1.0,
    "boundary_endcap_fraction": 0.14,
    "boundary_tip_protection_fraction": 0.06,
    "boundary_min_width_change_px": 1.0,
    "boundary_round_kernel": 5,
    "boundary_profile_window_px": 5,
    "temporal_width_control_enabled": True,
    "temporal_width_control_side": "Both",
    "temporal_width_max_ratio_per_frame": 1.18,
    "temporal_width_max_delta_px": 6.0,
    "temporal_width_adjust_max_iters": 6,
    "continuation_max_gap_px": 25,
    "segmentation_backend": "cellpose",
}

def create_segmentation_model():
    """Create segmentation model, enforcing Cellpose-only execution."""
    use_gpu = torch.cuda.is_available()
    backend = str(CONFIG.get("segmentation_backend", "cellpose")).strip().lower()

    if backend != "cellpose":
        raise RuntimeError(
            "segmentation_backend must be 'cellpose'."
        )

    if not CELLPOSE_AVAILABLE:
        raise RuntimeError(
            "Cellpose is not installed or failed to import. "
            "Install dependencies and retry."
        )

    model = cellpose_models.CellposeModel(gpu=use_gpu, model_type="cyto")

    print("Using Cellpose backend (cyto)")
    return model, "cellpose"

def run_segmentation_eval(model, backend, enhanced_frame):
    if backend != "cellpose":
        raise RuntimeError("Invalid backend: only cellpose is allowed.")
    
    if enhanced_frame.ndim == 3 and enhanced_frame.shape[-1] >= 1:
        enhanced_frame = enhanced_frame[:, :, 0]
    
    # הסרת מימדים ריקים (למשל אם המבנה הוא 1xHxW)
    enhanced_frame = np.squeeze(enhanced_frame)

    eval_kwargs = {
        "diameter": None,
        "channels": [0, 0],
        "cellprob_threshold": CONFIG["cellprob_threshold"],
        "flow_threshold": CONFIG["flow_threshold"],
    }

    masks, _, _ = model.eval(enhanced_frame, **eval_kwargs)
    return masks

def adjust_gamma(image, gamma=None):
    gamma = gamma if gamma is not None else CONFIG["gamma"]
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def get_cell_metrics(mask):
    """מחשבת את כל המטריקות הנדרשות עבור התא מתוך המסכה שלו."""
    props = regionprops(mask.astype(int))
    if not props:
        return 0, 0, 1.0
        
    prop = props[0]
    length = prop.major_axis_length
    perimeter = prop.perimeter
    area = prop.area
    
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0.0
        
    return length, perimeter, circularity

def compute_edge_map(frame):
    if frame.dtype != np.uint8:
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        normalized = frame.copy()

    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)

def get_edge_score(edge_map, region_mask):
    if not np.any(region_mask):
        return 0.0
    return float(np.mean(edge_map[region_mask > 0]))

def get_edge_percentile_score(edge_map, region_mask, percentile):
    if not np.any(region_mask):
        return 0.0
    return float(np.percentile(edge_map[region_mask > 0], percentile))

def get_mask_axis_profile(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return None

    coords = np.column_stack((ys, xs)).astype(np.float32)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    major_axis_norm = np.linalg.norm(major_axis)
    if major_axis_norm == 0:
        return None

    major_axis = major_axis / major_axis_norm
    projections = centered @ major_axis
    return center, major_axis, float(np.min(projections)), float(np.max(projections))

def get_mask_width_px(mask):
    axis_profile = get_mask_axis_profile(mask)
    if axis_profile is None:
        return 0.0

    center, major_axis, _, _ = axis_profile
    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)
    minor_norm = np.linalg.norm(minor_axis)
    if minor_norm == 0:
        return 0.0
    minor_axis = minor_axis / minor_norm

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    coords = np.column_stack((ys, xs)).astype(np.float32)
    minor_proj = (coords - center) @ minor_axis
    return float(np.max(minor_proj) - np.min(minor_proj))

def get_mask_endcap_region(mask, axis_profile, endcap_fraction):
    if axis_profile is None or not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)

    center, major_axis, min_proj, max_proj = axis_profile
    length = max_proj - min_proj
    if length <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    cap_len = max(1.0, float(length) * max(0.0, min(0.45, endcap_fraction)))
    ys, xs = np.where(mask > 0)
    coords = np.column_stack((ys, xs)).astype(np.float32)
    projections = (coords - center) @ major_axis

    near_min = projections <= (min_proj + cap_len)
    near_max = projections >= (max_proj - cap_len)
    keep = near_min | near_max

    cap = np.zeros_like(mask, dtype=np.uint8)
    cap[ys[keep], xs[keep]] = 1
    return cap

def constrain_region_away_from_endcaps(region_mask, axis_profile, endcap_fraction):
    if axis_profile is None or not np.any(region_mask):
        return region_mask.astype(np.uint8)

    center, major_axis, min_proj, max_proj = axis_profile
    length = max_proj - min_proj
    if length <= 0:
        return region_mask.astype(np.uint8)

    cap_len = max(1.0, float(length) * max(0.0, min(0.45, endcap_fraction)))
    ys, xs = np.where(region_mask > 0)
    coords = np.column_stack((ys, xs)).astype(np.float32)
    projections = (coords - center) @ major_axis
    keep = (projections >= min_proj + cap_len) & (projections <= max_proj - cap_len)

    constrained = np.zeros_like(region_mask, dtype=np.uint8)
    constrained[ys[keep], xs[keep]] = 1
    return constrained

def preserve_mask_endcaps(original_mask, corrected_mask):
    axis_profile = get_mask_axis_profile(original_mask)
    endcap_fraction = float(CONFIG["boundary_endcap_fraction"])
    endcap_region = get_mask_endcap_region(original_mask, axis_profile, endcap_fraction)
    if not np.any(endcap_region):
        return corrected_mask.astype(np.uint8)

    merged = corrected_mask.astype(np.uint8).copy()
    merged[endcap_region > 0] = original_mask[endcap_region > 0]
    return merged

def smooth_1d_profile(values, window_size):
    if window_size <= 1 or len(values) == 0:
        return values.astype(np.float32)

    pad = window_size // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(padded, kernel, mode="valid")

def close_1d_profile(values, window_size):
    if window_size <= 1 or len(values) == 0:
        return values.astype(np.float32)

    pad = window_size // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    dilated = np.empty(len(values), dtype=np.float32)
    for i in range(len(values)):
        dilated[i] = np.max(padded[i:i + window_size])

    padded_dilated = np.pad(dilated, (pad, pad), mode="edge")
    closed = np.empty(len(values), dtype=np.float32)
    for i in range(len(values)):
        closed[i] = np.min(padded_dilated[i:i + window_size])

    return closed

def regularize_mask_profile(original_mask, corrected_mask):
    if not np.any(corrected_mask):
        return corrected_mask.astype(np.uint8)

    axis_profile = get_mask_axis_profile(original_mask)
    if axis_profile is None:
        axis_profile = get_mask_axis_profile(corrected_mask)
    if axis_profile is None:
        return corrected_mask.astype(np.uint8)

    center, major_axis, min_proj, max_proj = axis_profile
    length = max_proj - min_proj
    if length <= 0:
        return corrected_mask.astype(np.uint8)

    minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=np.float32)
    minor_norm = np.linalg.norm(minor_axis)
    if minor_norm == 0:
        return corrected_mask.astype(np.uint8)
    minor_axis = minor_axis / minor_norm

    ys, xs = np.where(corrected_mask > 0)
    coords = np.column_stack((ys, xs)).astype(np.float32)
    projections = (coords - center) @ major_axis
    minor_projections = (coords - center) @ minor_axis

    start_bin = int(np.floor(min_proj))
    end_bin = int(np.ceil(max_proj))
    bin_edges = np.arange(start_bin, end_bin + 2, dtype=np.float32)
    if len(bin_edges) < 2:
        return corrected_mask.astype(np.uint8)

    half_widths = np.zeros(len(bin_edges) - 1, dtype=np.float32)
    for idx in range(len(half_widths)):
        in_bin = (projections >= bin_edges[idx]) & (projections < bin_edges[idx + 1])
        if np.any(in_bin):
            half_widths[idx] = float(np.max(np.abs(minor_projections[in_bin])))

    window_size = max(1, int(CONFIG["boundary_profile_window_px"]))
    if window_size % 2 == 0:
        window_size += 1

    regularized_half_widths = close_1d_profile(half_widths, window_size)
    regularized_half_widths = smooth_1d_profile(regularized_half_widths, window_size)
    regularized_half_widths = np.maximum(regularized_half_widths, half_widths)

    height, width = corrected_mask.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    grid_coords = np.column_stack((yy.ravel(), xx.ravel()))
    grid_proj = (grid_coords - center) @ major_axis
    grid_minor = np.abs((grid_coords - center) @ minor_axis)

    valid_longitudinal = (grid_proj >= min_proj) & (grid_proj <= max_proj)
    bin_indices = np.clip(np.floor(grid_proj - start_bin).astype(int), 0, len(regularized_half_widths) - 1)
    allowed_half_width = regularized_half_widths[bin_indices]
    rebuilt = valid_longitudinal & (grid_minor <= allowed_half_width)

    return rebuilt.reshape((height, width)).astype(np.uint8)

def round_corrected_mask(mask):
    kernel_size = max(1, int(CONFIG["boundary_round_kernel"]))
    if kernel_size <= 1 or not np.any(mask):
        return mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    rounded = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    rounded = cv2.morphologyEx(rounded, cv2.MORPH_OPEN, kernel)
    return rounded.astype(np.uint8)

def constrain_region_to_cell_width(region_mask, axis_profile, pad_px):
    if axis_profile is None or not np.any(region_mask):
        return region_mask.astype(np.uint8)

    center, major_axis, min_proj, max_proj = axis_profile
    ys, xs = np.where(region_mask > 0)
    coords = np.column_stack((ys, xs)).astype(np.float32)
    projections = (coords - center) @ major_axis
    keep = (projections >= min_proj - pad_px) & (projections <= max_proj + pad_px)

    constrained = np.zeros_like(region_mask, dtype=np.uint8)
    constrained[ys[keep], xs[keep]] = 1
    return constrained

def adjust_mask_width_towards_target(mask, target_width_px):
    original_mask = mask.astype(np.uint8)
    mask_uint8 = original_mask.copy()
    if not np.any(mask_uint8):
        return mask_uint8

    kernel = np.ones((3, 3), dtype=np.uint8)
    length_pad_px = float(CONFIG["boundary_length_pad_px"])
    max_iters = max(1, int(CONFIG["temporal_width_adjust_max_iters"]))
    tip_protection_fraction = float(CONFIG["boundary_tip_protection_fraction"])

    for _ in range(max_iters):
        current_width = get_mask_width_px(mask_uint8)
        if current_width <= 0:
            break

        if abs(current_width - target_width_px) <= 0.5:
            break

        axis_profile = get_mask_axis_profile(mask_uint8)
        if axis_profile is None:
            break

        if current_width < target_width_px:
            outer_boundary = cv2.dilate(mask_uint8, kernel, iterations=1) - mask_uint8
            outer_boundary = constrain_region_to_cell_width(outer_boundary, axis_profile, length_pad_px)
            outer_boundary = constrain_region_away_from_endcaps(outer_boundary, axis_profile, tip_protection_fraction)
            if not np.any(outer_boundary):
                break
            mask_uint8 = np.maximum(mask_uint8, outer_boundary).astype(np.uint8)
        else:
            eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
            if not np.any(eroded_mask):
                break
            boundary = mask_uint8 - eroded_mask
            boundary = constrain_region_to_cell_width(boundary, axis_profile, -length_pad_px)
            boundary = constrain_region_away_from_endcaps(boundary, axis_profile, tip_protection_fraction)
            if not np.any(boundary):
                break
            shrunk = mask_uint8.copy()
            shrunk[boundary > 0] = 0
            if not np.any(shrunk):
                break
            mask_uint8 = shrunk

    mask_uint8 = round_corrected_mask(mask_uint8)
    return preserve_mask_endcaps(original_mask, mask_uint8)

def apply_temporal_width_control(chosen_cells, frame, prev_width_by_side):
    if not CONFIG.get("temporal_width_control_enabled", False):
        return chosen_cells, prev_width_by_side

    side_filter = str(CONFIG.get("temporal_width_control_side", "Both")).lower()
    ratio_limit = max(1.0, float(CONFIG["temporal_width_max_ratio_per_frame"]))
    delta_limit_px = max(0.0, float(CONFIG["temporal_width_max_delta_px"]))

    updated_prev = dict(prev_width_by_side)
    adjusted_cells = []

    for i, cell in enumerate(chosen_cells):
        side = "Left" if i == 0 else "Right"
        apply_for_side = side_filter in ("both", side.lower())

        cell_local = dict(cell)
        current_width = get_mask_width_px(cell_local["mask"])

        if apply_for_side and side in updated_prev and current_width > 0:
            prev_width = float(updated_prev[side])
            allowed_delta = max(delta_limit_px, prev_width * (ratio_limit - 1.0))
            min_width = max(1.0, prev_width - allowed_delta)
            max_width = prev_width + allowed_delta
            target_width = min(max(current_width, min_width), max_width)

            if abs(target_width - current_width) > 0.5:
                adjusted_mask = adjust_mask_width_towards_target(cell_local["mask"], target_width)
                if np.any(adjusted_mask):
                    ys, xs = np.where(adjusted_mask > 0)
                    cell_local["mask"] = adjusted_mask
                    cell_local["y_min"] = int(ys.min())
                    cell_local["y_max"] = int(ys.max())
                    cell_local["cx"] = int(xs.mean())
                    cell_local["cy"] = int(ys.mean())
                    cell_local["area"] = int(np.sum(adjusted_mask))
                    cell_local["intensity"] = float(np.mean(frame[adjusted_mask > 0]))
                    current_width = get_mask_width_px(adjusted_mask)

        if current_width > 0:
            updated_prev[side] = float(current_width)

        adjusted_cells.append(cell_local)

    return adjusted_cells, updated_prev

def refine_cell_mask(mask, edge_map):
    original_mask = mask.astype(np.uint8)
    mask_uint8 = original_mask.copy()
    if not np.any(mask_uint8):
        return mask_uint8

    original_width = get_mask_width_px(original_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    max_expand = max(0, int(CONFIG["boundary_max_expand_px"]))
    min_gain = float(CONFIG["boundary_edge_gain"])
    min_delta = float(CONFIG["boundary_edge_delta"])
    min_edge_strength = float(CONFIG["boundary_min_edge_strength"])
    edge_percentile = float(CONFIG["boundary_edge_percentile"])
    min_support_fraction = float(CONFIG["boundary_min_support_fraction"])
    max_shrink = max(0, int(CONFIG["boundary_max_shrink_px"]))
    shrink_gain = float(CONFIG["boundary_shrink_gain"])
    shrink_delta = float(CONFIG["boundary_shrink_delta"])
    max_current_support_fraction = float(CONFIG["boundary_shrink_max_support_fraction"])
    length_pad_px = float(CONFIG["boundary_length_pad_px"])
    tip_protection_fraction = float(CONFIG["boundary_tip_protection_fraction"])

    for _ in range(max_expand):
        axis_profile = get_mask_axis_profile(mask_uint8)
        inner_boundary = mask_uint8 - cv2.erode(mask_uint8, kernel, iterations=1)
        outer_boundary = cv2.dilate(mask_uint8, kernel, iterations=1) - mask_uint8
        outer_boundary = constrain_region_to_cell_width(outer_boundary, axis_profile, length_pad_px)
        outer_boundary = constrain_region_away_from_endcaps(outer_boundary, axis_profile, tip_protection_fraction)

        if not np.any(outer_boundary):
            break

        current_score = get_edge_score(edge_map, inner_boundary)
        outer_score = get_edge_score(edge_map, outer_boundary)
        current_peak_score = get_edge_percentile_score(edge_map, inner_boundary, edge_percentile)
        outer_peak_score = get_edge_percentile_score(edge_map, outer_boundary, edge_percentile)
        outer_values = edge_map[outer_boundary > 0]
        support_fraction = float(
            np.mean(outer_values >= max(min_edge_strength, current_peak_score + min_delta))
        ) if outer_values.size else 0.0

        should_expand = (
            outer_peak_score >= min_edge_strength and
            outer_peak_score >= current_peak_score * min_gain and
            outer_peak_score >= current_peak_score + min_delta and
            (outer_score >= current_score or support_fraction >= min_support_fraction)
        )

        if not should_expand:
            break

        mask_uint8 = np.maximum(mask_uint8, outer_boundary).astype(np.uint8)

    for _ in range(max_shrink):
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
        if not np.any(eroded_mask):
            break

        axis_profile = get_mask_axis_profile(mask_uint8)
        current_boundary = mask_uint8 - eroded_mask
        current_boundary = constrain_region_to_cell_width(current_boundary, axis_profile, -length_pad_px)
        current_boundary = constrain_region_away_from_endcaps(current_boundary, axis_profile, tip_protection_fraction)
        if not np.any(current_boundary):
            break

        width_only_shrunk_mask = mask_uint8.copy()
        width_only_shrunk_mask[current_boundary > 0] = 0
        if not np.any(width_only_shrunk_mask):
            break

        inner_candidate_boundary = width_only_shrunk_mask - cv2.erode(width_only_shrunk_mask, kernel, iterations=1)
        inner_candidate_boundary = constrain_region_to_cell_width(inner_candidate_boundary, axis_profile, -length_pad_px)
        inner_candidate_boundary = constrain_region_away_from_endcaps(inner_candidate_boundary, axis_profile, tip_protection_fraction)

        if not np.any(inner_candidate_boundary):
            break

        current_score = get_edge_score(edge_map, current_boundary)
        inner_score = get_edge_score(edge_map, inner_candidate_boundary)
        current_peak_score = get_edge_percentile_score(edge_map, current_boundary, edge_percentile)
        inner_peak_score = get_edge_percentile_score(edge_map, inner_candidate_boundary, edge_percentile)
        current_values = edge_map[current_boundary > 0]
        current_support_fraction = float(
            np.mean(current_values >= max(min_edge_strength, current_peak_score))
        ) if current_values.size else 0.0

        should_shrink = (
            inner_peak_score >= min_edge_strength and
            inner_peak_score >= current_peak_score * shrink_gain and
            inner_peak_score >= current_peak_score + shrink_delta and
            inner_score >= current_score and
            current_support_fraction <= max_current_support_fraction
        )

        if not should_shrink:
            break

        mask_uint8 = width_only_shrunk_mask

    mask_uint8 = round_corrected_mask(mask_uint8)
    mask_uint8 = preserve_mask_endcaps(original_mask, mask_uint8)

    corrected_width = get_mask_width_px(mask_uint8)
    min_width_change = float(CONFIG["boundary_min_width_change_px"])
    if original_width > 0 and abs(corrected_width - original_width) < min_width_change:
        return original_mask

    return mask_uint8

def extract_cells(masks, frame):
    cells = []
    edge_map = compute_edge_map(frame)
    for i in range(1, masks.max() + 1):
        mask = refine_cell_mask(masks == i, edge_map)
        area = int(np.sum(mask))
        if area < CONFIG["min_cell_area"]: continue
        ys, xs = np.where(mask > 0)
        cells.append({
            "id": i,
            "y_min": int(ys.min()), "y_max": int(ys.max()),
            "cx": int(xs.mean()), "cy": int(ys.mean()),
            "area": area, "intensity": float(np.mean(frame[mask > 0])),
            "mask": mask,
        })
    return cells

def pick_two_top_cells_logic(all_cells):
    """
    מנגנון גלובלי: מחלק את כל התאים הנוכחיים בפריים לשמאל וימין,
    ובוחר בצורה אבסולוטית את התא הגבוה ביותר בכל צד (cy מינימלי).
    """
    if not all_cells:
        return []
    if len(all_cells) < 2:
        return sorted(all_cells, key=lambda c: c["cx"])
        
    # מציאת נקודת האמצע האופקית הריאליסטית של התאים בפריים הנוכחי
    all_x = [c["cx"] for c in all_cells]
    mid_x = (min(all_x) + max(all_x)) / 2.0
    
    left_side = [c for c in all_cells if c["cx"] <= mid_x]
    right_side = [c for c in all_cells if c["cx"] > mid_x]
    
    if not left_side or not right_side:
        # פולבק קיצוני אם הכל בצד אחד: לוקחים את שני הגבוהים ביותר
        top_two = sorted(all_cells, key=lambda c: c["cy"])[:2]
        return sorted(top_two, key=lambda c: c["cx"])
        
    best_left = min(left_side, key=lambda c: c["cy"])
    best_right = min(right_side, key=lambda c: c["cy"])
    
    return [best_left, best_right]




def pick_closest_cells_logic(all_cells, prev_chosen):
    """
    מעקב מקומי מחמיר (Gatekeeper): בודק תנאי יציבות, רדיוס, תזוזת מרכוז וכמות בחירות.
    אם משהו משתבש או חורג מהנורמה, מחזיר רשימה ריקה [] ומאלץ אתחול גלובלי.
    """
    if not all_cells or len(prev_chosen) < 2:
        return []
        
    prev_left = prev_chosen[0]
    prev_right = prev_chosen[1]
    
    # ------------------------------------------------------------------
    # תנאי 1: בדיקת רדיוס מקומי סביר (מניעת בריחה של תאים)
    # ------------------------------------------------------------------
    MAX_RADIUS = 70  # רדיוס תנועה מקסימלי הגיוני בין פריימים עוקבים
    
    left_candidates = []
    right_candidates = []
    
    for cell in all_cells:
        dist_l = np.sqrt((cell["cx"] - prev_left["cx"])**2 + (cell["cy"] - prev_left["cy"])**2)
        if dist_l <= MAX_RADIUS:
            left_candidates.append(cell)
            
        dist_r = np.sqrt((cell["cx"] - prev_right["cx"])**2 + (cell["cy"] - prev_right["cy"])**2)
        if dist_r <= MAX_RADIUS:
            right_candidates.append(cell)
            
    # אם אחת מהזרועות לא מצאה מועמד ברדיוס שלה -> בעיית רדיוס! נפסל ומאתחלים
    if not left_candidates or not right_candidates:
        return []
        
    # שליפת המועמדים הקרובים והגבוהים ביותר ברדיוס
    best_left = min(left_candidates, key=lambda c: c["cy"])
    
    available_right = [c for c in right_candidates if c.get("label") != best_left.get("label")]
    if not available_right:
        return []  # התנגשות (שני החיפושים מצאו את אותו תא) -> נפסל ומאתחלים
        
    best_right = min(available_right, key=lambda c: c["cy"])
    
    # ------------------------------------------------------------------
    # תנאי 2: בדיקת תזוזת מרכוז קיצונית (Shift / Drift Detection)
    # ------------------------------------------------------------------
    # נבדוק אם שני התאים זזו בבת אחת באותו כיוון חריף מדי לעומת הפריים הקודם
    shift_left = np.sqrt((best_left["cx"] - prev_left["cx"])**2 + (best_left["cy"] - prev_left["cy"])**2)
    shift_right = np.sqrt((best_right["cx"] - prev_right["cx"])**2 + (best_right["cy"] - prev_right["cy"])**2)
    
    MAX_ALLOWED_SHIFT = 50  # אם יש קפיצת תמונה גדולה מ-50 פיקסלים בפריים בודד
    if shift_left > MAX_ALLOWED_SHIFT or shift_right > MAX_ALLOWED_SHIFT:
        return []  # זוהתה תזוזת מרכוז חריפה! נפסל ומאתחלים גלובלית
        
    # ------------------------------------------------------------------
    # תנאי 3: וידוא מבני סופי (שמאל נשאר משמאל, ימין מימין)
    # ------------------------------------------------------------------
    if best_left["cx"] >= best_right["cx"]:
        return []  # האינדקסים התהפכו או חצו אחד את השני -> נפסל ומאתחלים
        
    # המצב מושלם ועבר את כל שלבי ה-Gatekeeper, אפשר להמשיך לעקוב חלק
    return [best_left, best_right]



def draw_cell_outline(annotated, mask, color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    cv2.drawContours(annotated, contours, -1, color, CONFIG["contour_thickness"], lineType=cv2.LINE_8)

def build_output_folder_from_input(input_folder):
    normalized_input = os.path.normpath(input_folder)
    parent_dir = os.path.dirname(normalized_input)
    input_name = os.path.basename(normalized_input)

    if not input_name:
        return normalized_input + "_results"

    return os.path.join(parent_dir, f"{input_name}_results")

def process_folder_automation(input_folder, output_folder, callback=None):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    model, backend = create_segmentation_model()

    # רשימת הקבצים לחישוב התקדמות
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]
    total_files = len(files)

    for file_idx, filename in enumerate(files):
        print(f"Processing {filename}...")
        
        with tiff.TiffFile(os.path.join(input_folder, filename)) as tif:
            images = tif.asarray()
        if len(images.shape) == 2: images = np.expand_dims(images, axis=0)

        all_frames_data = []
        output_frames = []
        backup_data = {}
        prev_chosen = []
        prev_width_by_side = {}
        
        total_frames = len(images) # סה"כ פריימים בקובץ הנוכחי
        
        for frame_idx, frame in enumerate(images):
            # --- דיווח לממשק (הוסף זאת לפני הלוגיקה של העיבוד) ---
            if callback:
                status_msg = f"File {file_idx + 1}/{total_files} ({filename}) | Frame {frame_idx + 1}/{total_frames}"
                callback(status_msg)
            
            enhanced = adjust_gamma(frame, gamma=CONFIG["gamma"])
            masks = run_segmentation_eval(model, backend, enhanced)
            
            all_detected_cells = extract_cells(masks, frame)
            
            if frame_idx == 0:
                chosen = pick_two_top_cells_logic(all_detected_cells)
            else:
                chosen = pick_closest_cells_logic(all_detected_cells, prev_chosen)
                if not chosen or len(chosen) < 2:
                    chosen = pick_two_top_cells_logic(all_detected_cells)

            chosen, prev_width_by_side = apply_temporal_width_control(chosen, frame, prev_width_by_side)
            
            prev_chosen = chosen
            backup_data[frame_idx] = {"chosen": chosen, "all": all_detected_cells}
            
            # --- ציור ועיבוד (ללא שינוי) ---
            frame_res = {"Frame": frame_idx + 1, "Left": None, "Right": None}
            
            # תיקון OpenCV: בדיקת ערוצים לפני המרה
            norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if len(norm_frame.shape) == 2:
                annotated = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)
            else:
                annotated = norm_frame

            for i, cell in enumerate(chosen):
                side = "Left" if i == 0 else "Right"
                color = (0, 255, 0) if side == "Left" else (0, 0, 255)
                draw_cell_outline(annotated, cell["mask"], color)
                
                length_px, perimeter_px, circularity = get_cell_metrics(cell["mask"])
                frame_res[side] = {
                    "Length_Microns": length_px * PIXEL_TO_MICRON,
                    "Perimeter_Microns": perimeter_px * PIXEL_TO_MICRON,
                    "Circularity": circularity,
                    "Area_Pixels": cell.get("area", 0),
                    "Mean_Intensity": cell.get("intensity", 0),
                    "Center_X": cell["cx"],
                    "Center_Y": cell["cy"]
                }
            
            all_frames_data.append(frame_res)
            output_frames.append(annotated)

        # שמירת תוצאות (ללא שינוי)
        tiff.imwrite(os.path.join(output_folder, f"{filename}_analyzed.tif"), np.array(output_frames), photometric="rgb")
        np.save(os.path.join(output_folder, f"{filename}_segmentation_data.npy"), backup_data, allow_pickle=True)
        
        with open(os.path.join(output_folder, f"{filename}_raw_data.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Length Left", "Area Left", "Intensity Left", "Length Right", "Area Right", "Intensity Right"])
            for d in all_frames_data:
                left = d["Left"] or {}
                right = d["Right"] or {}
                w.writerow([
                    d["Frame"],
                    left.get("Length_Microns", 0),
                    left.get("Area_Pixels", 0),
                    left.get("Mean_Intensity", 0),
                    right.get("Length_Microns", 0),
                    right.get("Area_Pixels", 0),
                    right.get("Mean_Intensity", 0),
                ])
                        
        print(f"Processing has ended for: {filename}")

if __name__ == "__main__":
    from tkinter import filedialog

    input_dir = filedialog.askdirectory(title="Input Folder")
    output_dir = build_output_folder_from_input(input_dir) if input_dir else None

    if input_dir and output_dir:
        process_folder_automation(input_dir, output_dir)