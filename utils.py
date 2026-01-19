from __future__ import annotations

import logging
import math
import re
from typing import Optional, Tuple

import cv2
import numpy as np


def parse_tile_coords(filename: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"^(\d+),(\d+)(?:_.*)?\.(jpg|jpeg)$", filename, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def make_content_mask(img: np.ndarray, bg_threshold: int) -> np.ndarray:
    mask = np.all(img > bg_threshold, axis=2)
    return (~mask).astype(np.uint8)


def crop_to_mask(img: np.ndarray, mask: np.ndarray, margin: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img, mask
    x_min = max(xs.min() - margin, 0)
    x_max = min(xs.max() + margin + 1, img.shape[1])
    y_min = max(ys.min() - margin, 0)
    y_max = min(ys.max() + margin + 1, img.shape[0])
    return img[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def compute_bg_threshold(img: np.ndarray, base_threshold: int) -> int:
    height, width = img.shape[:2]
    band_h = max(1, int(height * 0.08))
    band_w = max(1, int(width * 0.08))
    top = img[:band_h, :, :]
    bottom = img[-band_h:, :, :]
    left = img[:, :band_w, :]
    right = img[:, -band_w:, :]
    border = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    gray = border.mean(axis=1)
    percentile = float(np.percentile(gray, 90))
    adaptive = min(base_threshold, int(percentile))
    return int(np.clip(adaptive, 200, 255))


def _crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def trim_white_borders(
    img: np.ndarray,
    mask: np.ndarray,
    min_content_ratio: float = 0.01,
    margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    row_min_content = max(1, int(width * min_content_ratio))
    col_min_content = max(1, int(height * min_content_ratio))

    rows = np.where(row_counts >= row_min_content)[0]
    cols = np.where(col_counts >= col_min_content)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img, mask

    y_min = max(rows.min() - margin, 0)
    y_max = min(rows.max() + margin + 1, height)
    x_min = max(cols.min() - margin, 0)
    x_max = min(cols.max() + margin + 1, width)
    return img[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


def estimate_rotation_angle(mask: np.ndarray, max_angle: float) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angle = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = float(rect[2])
        if angle < -45:
            angle += 90

    if abs(angle) < 0.1:
        edges = cv2.Canny(mask * 255, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)
        if lines is not None:
            angles = []
            weights = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2 and y1 == y2:
                    continue
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 40:
                    continue
                line_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if line_angle < -45:
                    line_angle += 90
                if line_angle > 45:
                    line_angle -= 90
                angles.append(line_angle)
                weights.append(length)
            if angles:
                angle = float(np.average(angles, weights=weights))

    if abs(angle) > max_angle:
        logging.info("Angle %.2f exceeds max-angle %.2f; using 0", angle, max_angle)
        return 0.0
    return angle


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def prepare_strip(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    return mag


def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        height = min(a.shape[0], b.shape[0])
        width = min(a.shape[1], b.shape[1])
        if height < 4 or width < 4:
            return -1.0
        a = a[:height, :width]
        b = b[:height, :width]
    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-5 or b_std < 1e-5:
        return -1.0
    norm = ((a - a_mean) / a_std) * ((b - b_mean) / b_std)
    return float(norm.mean())


def phase_shift(base: np.ndarray, other: np.ndarray) -> Tuple[float, float]:
    base, other = _crop_to_common(base, other)
    base_f = base.astype(np.float32)
    other_f = other.astype(np.float32)
    shift, _ = cv2.phaseCorrelate(base_f, other_f)
    return shift[0], shift[1]


def match_with_shift_local(base: np.ndarray, other: np.ndarray, max_shift: int, center: Tuple[int, int]) -> Tuple[int, int, float]:
    best_dx = 0
    best_dy = 0
    best_score = -1.0
    h, w = base.shape
    center_dx, center_dy = center
    for dy in range(center_dy - max_shift, center_dy + max_shift + 1):
        for dx in range(center_dx - max_shift, center_dx + max_shift + 1):
            x0 = max(0, dx)
            y0 = max(0, dy)
            x1 = min(w, w + dx)
            y1 = min(h, h + dy)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            base_crop = base[y0:y1, x0:x1]
            other_crop = other[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            score = ncc_score(base_crop, other_crop)
            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy
    return best_dx, best_dy, best_score


def match_with_shift_multiscale(base: np.ndarray, other: np.ndarray, max_shift: int) -> Tuple[int, int, float]:
    base, other = _crop_to_common(base, other)
    h, w = base.shape
    small_w = max(1, w // 2)
    small_h = max(1, h // 2)
    small_base = cv2.resize(base, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_other = cv2.resize(other, (small_w, small_h), interpolation=cv2.INTER_AREA)
    coarse_dx, coarse_dy = phase_shift(small_base, small_other)
    coarse = (int(round(coarse_dx * 2)), int(round(coarse_dy * 2)))
    coarse = (int(np.clip(coarse[0], -max_shift, max_shift)), int(np.clip(coarse[1], -max_shift, max_shift)))
    return match_with_shift_local(base, other, max_shift, coarse)

def match_with_shift(base: np.ndarray, other: np.ndarray, max_shift: int) -> Tuple[int, int, float]:
    return match_with_shift_multiscale(base, other, max_shift)


def match_strips_left_right(
    left: np.ndarray,
    current: np.ndarray,
    overlap_max: int,
    bg_threshold: int,
) -> Tuple[int, int, float]:
    strip_left = left[:, -overlap_max:]
    strip_right = current[:, :overlap_max]
    base = prepare_strip(strip_left)
    other = prepare_strip(strip_right)
    mask_left = clean_mask(make_content_mask(strip_left, bg_threshold))
    mask_right = clean_mask(make_content_mask(strip_right, bg_threshold))
    mask_left, mask_right = _crop_to_common(mask_left, mask_right)
    base, other = _crop_to_common(base, other)
    base = base * mask_left
    other = other * mask_right
    dx, dy, score = match_with_shift(base, other, overlap_max)
    return dx, dy, score


def match_strips_top_bottom(
    top: np.ndarray,
    current: np.ndarray,
    overlap_max: int,
    bg_threshold: int,
) -> Tuple[int, int, float]:
    strip_top = top[-overlap_max:, :]
    strip_bottom = current[:overlap_max, :]
    base = prepare_strip(strip_top)
    other = prepare_strip(strip_bottom)
    mask_top = clean_mask(make_content_mask(strip_top, bg_threshold))
    mask_bottom = clean_mask(make_content_mask(strip_bottom, bg_threshold))
    mask_top, mask_bottom = _crop_to_common(mask_top, mask_bottom)
    base, other = _crop_to_common(base, other)
    base = base * mask_top
    other = other * mask_bottom
    dx, dy, score = match_with_shift(base, other, overlap_max)
    return dx, dy, score
