from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import cv2
import numpy as np


def parse_tile_coords(filename: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"^(\d+),(\d+)(?:_.*)?\.jpg$", filename, re.IGNORECASE)
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


def estimate_rotation_angle(mask: np.ndarray, max_angle: float) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = float(angle)
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
    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std()
    b_std = b.std()
    if a_std < 1e-5 or b_std < 1e-5:
        return -1.0
    norm = ((a - a_mean) / a_std) * ((b - b_mean) / b_std)
    return float(norm.mean())


def match_with_shift(base: np.ndarray, other: np.ndarray, max_shift: int) -> Tuple[int, int, float]:
    best_dx = 0
    best_dy = 0
    best_score = -1.0
    h, w = base.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
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


def match_strips_left_right(left: np.ndarray, current: np.ndarray, overlap_max: int) -> Tuple[int, int, float]:
    strip_left = left[:, -overlap_max:]
    strip_right = current[:, :overlap_max]
    base = prepare_strip(strip_left)
    other = prepare_strip(strip_right)
    dx, dy, score = match_with_shift(base, other, overlap_max)
    return dx, dy, score


def match_strips_top_bottom(top: np.ndarray, current: np.ndarray, overlap_max: int) -> Tuple[int, int, float]:
    strip_top = top[-overlap_max:, :]
    strip_bottom = current[:overlap_max, :]
    base = prepare_strip(strip_top)
    other = prepare_strip(strip_bottom)
    dx, dy, score = match_with_shift(base, other, overlap_max)
    return dx, dy, score
