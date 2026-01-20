from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from models import StitchParams, Tile
from utils import (
    clean_mask,
    compute_bg_threshold,
    estimate_rotation_angle,
    make_content_mask,
    match_strips_left_right,
    match_strips_top_bottom,
    parse_tile_coords,
    rotate_image,
    trim_outer_white,
)


class StitchCancelled(Exception):
    pass


def _check_cancel(cancel_flag) -> None:
    if cancel_flag is not None and cancel_flag.is_set():
        raise StitchCancelled("Cancelled by user")


def preprocess_tile(path: Path, params: StitchParams, on_log: Optional[Callable[[str], None]]) -> Tile:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read {path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    adaptive_threshold = compute_bg_threshold(img, params.bg_threshold)
    cropped, cropped_mask = trim_outer_white(img, adaptive_threshold, margin=params.crop_padding_px)
    if params.debug and params.debug_dir:
        cv2.imwrite(str(params.debug_dir / f"{path.stem}_mask_before.png"), cropped_mask * 255)
        cv2.imwrite(str(params.debug_dir / f"{path.stem}_crop_before.png"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    angle = estimate_rotation_angle(cropped_mask, params.max_angle)
    rotated = rotate_image(cropped, angle)
    rotated, rotated_mask = trim_outer_white(rotated, adaptive_threshold, margin=params.crop_padding_px)
    if params.debug and params.debug_dir:
        cv2.imwrite(str(params.debug_dir / f"{path.stem}_mask_after.png"), rotated_mask * 255)
        cv2.imwrite(str(params.debug_dir / f"{path.stem}_crop_after.png"), cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
    h, w = rotated.shape[:2]
    coords = parse_tile_coords(path.name)
    if coords is None:
        raise ValueError(f"Invalid tile filename {path.name}")
    if on_log:
        on_log(f"Processed tile {path.name}: angle={angle:.2f}, size={w}x{h}, bg={adaptive_threshold}")
    return Tile(
        x=coords[0],
        y=coords[1],
        path=path,
        img=rotated,
        w=w,
        h=h,
        angle=angle,
        bg_threshold=adaptive_threshold,
    )


def build_rows(tiles: List[Tile]) -> List[List[Tile]]:
    rows_map: Dict[int, List[Tile]] = {}
    for tile in tiles:
        rows_map.setdefault(tile.x, []).append(tile)
    rows = [rows_map[key] for key in sorted(rows_map.keys())]
    for row in rows:
        row.sort(key=lambda t: t.y)
    return rows


def initial_layout(rows: List[List[Tile]]) -> None:
    y_offset = 0
    for row in rows:
        x_offset = 0
        row_height = max(tile.h for tile in row)
        for tile in row:
            tile.offset_x = x_offset
            tile.offset_y = y_offset
            x_offset += tile.w
        y_offset += row_height


def refine_layout(rows: List[List[Tile]], params: StitchParams) -> None:
    for row_idx, row in enumerate(rows):
        for col_idx, tile in enumerate(row):
            base_x = tile.offset_x
            base_y = tile.offset_y
            proposals = []
            if col_idx > 0:
                left = row[col_idx - 1]
                bg_threshold = min(left.bg_threshold, tile.bg_threshold)
                dx, dy, score = match_strips_left_right(left.img, tile.img, params.overlap_max, bg_threshold)
                if params.debug and params.debug_dir:
                    debug_path = params.debug_dir / f"match_left_{tile.x}_{tile.y}.json"
                    debug_path.write_text(json.dumps({"dx": dx, "dy": dy, "score": score}))
                if score < 0.2:
                    proposals.append((left.offset_x + left.w, left.offset_y, 0.1))
                else:
                    proposals.append((left.offset_x + left.w + dx, left.offset_y + dy, score))
            if row_idx > 0 and col_idx < len(rows[row_idx - 1]):
                top = rows[row_idx - 1][col_idx]
                bg_threshold = min(top.bg_threshold, tile.bg_threshold)
                dx, dy, score = match_strips_top_bottom(top.img, tile.img, params.overlap_max, bg_threshold)
                if params.debug and params.debug_dir:
                    debug_path = params.debug_dir / f"match_top_{tile.x}_{tile.y}.json"
                    debug_path.write_text(json.dumps({"dx": dx, "dy": dy, "score": score}))
                if score < 0.2:
                    proposals.append((top.offset_x, top.offset_y + top.h, 0.1))
                else:
                    proposals.append((top.offset_x + dx, top.offset_y + top.h + dy, score))
            if proposals:
                total = sum(p[2] for p in proposals)
                if total <= 0:
                    tile.offset_x = base_x
                    tile.offset_y = base_y
                else:
                    tile.offset_x = int(sum(p[0] * p[2] for p in proposals) / total)
                    tile.offset_y = int(sum(p[1] * p[2] for p in proposals) / total)
            else:
                tile.offset_x = base_x
                tile.offset_y = base_y


def _crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def seam_error(strip_a: np.ndarray, strip_b: np.ndarray) -> float:
    strip_a, strip_b = _crop_to_common(strip_a, strip_b)
    if strip_a.size == 0 or strip_b.size == 0:
        return float("inf")
    gray_a = cv2.cvtColor(strip_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(strip_b, cv2.COLOR_RGB2GRAY)
    diff = np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32))
    return float(diff.mean())


def compute_seam_metrics(left: Tile, right: Tile, seam_band_px: int) -> Dict[str, float]:
    band = min(seam_band_px, left.w, right.w)
    strip_left = left.img[:, -band:]
    strip_right = right.img[:, :band]
    err = seam_error(strip_left, strip_right)
    score = match_strips_left_right(left.img, right.img, min(band, 15), min(left.bg_threshold, right.bg_threshold))[2]
    overlap = max(0, (left.offset_x + left.w) - right.offset_x)
    gap = max(0, right.offset_x - (left.offset_x + left.w))
    return {"gap_px": float(gap), "overlap_px": float(overlap), "seam_error": err, "score": float(score)}


def compute_seam_metrics_vertical(top: Tile, bottom: Tile, seam_band_px: int) -> Dict[str, float]:
    band = min(seam_band_px, top.h, bottom.h)
    strip_top = top.img[-band:, :]
    strip_bottom = bottom.img[:band, :]
    err = seam_error(strip_top, strip_bottom)
    score = match_strips_top_bottom(top.img, bottom.img, min(band, 15), min(top.bg_threshold, bottom.bg_threshold))[2]
    overlap = max(0, (top.offset_y + top.h) - bottom.offset_y)
    gap = max(0, bottom.offset_y - (top.offset_y + top.h))
    return {"gap_px": float(gap), "overlap_px": float(overlap), "seam_error": err, "score": float(score)}


def compute_seam_metrics_map(rows: List[List[Tile]], seam_band_px: int) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    for row_idx, row in enumerate(rows):
        for col_idx, tile in enumerate(row):
            if col_idx > 0:
                left = row[col_idx - 1]
                data = compute_seam_metrics(left, tile, seam_band_px)
                data.update({"direction": "left", "tile": f"{tile.x},{tile.y}", "neighbor": f"{left.x},{left.y}"})
                metrics.append(data)
            if row_idx > 0 and col_idx < len(rows[row_idx - 1]):
                top = rows[row_idx - 1][col_idx]
                data = compute_seam_metrics_vertical(top, tile, seam_band_px)
                data.update({"direction": "top", "tile": f"{tile.x},{tile.y}", "neighbor": f"{top.x},{top.y}"})
                metrics.append(data)
    return metrics


def estimate_affine_to_neighbor(left: Tile, current: Tile, params: StitchParams) -> Tuple[int, int, float, float, float]:
    bg_threshold = min(left.bg_threshold, current.bg_threshold)
    dx, dy, score = match_strips_left_right(left.img, current.img, params.overlap_max, bg_threshold)
    base_error = seam_error(left.img[:, -params.seam_band_px :], current.img[:, : params.seam_band_px])
    best_scale = 1.0
    best_error = base_error
    scale_range = params.max_scale_percent / 100.0
    for scale in np.linspace(1.0 - scale_range, 1.0 + scale_range, 5):
        if abs(scale - 1.0) < 1e-4:
            continue
        scaled = cv2.resize(current.img, None, fx=scale, fy=1.0, interpolation=cv2.INTER_LINEAR)
        dx_s, dy_s, score_s = match_strips_left_right(left.img, scaled, params.overlap_max, bg_threshold)
        err_s = seam_error(left.img[:, -params.seam_band_px :], scaled[:, : params.seam_band_px])
        if err_s < best_error and score_s >= score:
            best_error = err_s
            best_scale = scale
            dx, dy, score = dx_s, dy_s, score_s
    return dx, dy, best_scale, 1.0, score


def estimate_affine_to_neighbor_vertical(top: Tile, current: Tile, params: StitchParams) -> Tuple[int, int, float, float, float]:
    bg_threshold = min(top.bg_threshold, current.bg_threshold)
    dx, dy, score = match_strips_top_bottom(top.img, current.img, params.overlap_max, bg_threshold)
    base_error = seam_error(top.img[-params.seam_band_px :, :], current.img[: params.seam_band_px, :])
    best_scale = 1.0
    best_error = base_error
    scale_range = params.max_scale_percent / 100.0
    for scale in np.linspace(1.0 - scale_range, 1.0 + scale_range, 5):
        if abs(scale - 1.0) < 1e-4:
            continue
        scaled = cv2.resize(current.img, None, fx=1.0, fy=scale, interpolation=cv2.INTER_LINEAR)
        dx_s, dy_s, score_s = match_strips_top_bottom(top.img, scaled, params.overlap_max, bg_threshold)
        err_s = seam_error(top.img[-params.seam_band_px :, :], scaled[: params.seam_band_px, :])
        if err_s < best_error and score_s >= score:
            best_error = err_s
            best_scale = scale
            dx, dy, score = dx_s, dy_s, score_s
    return dx, dy, 1.0, best_scale, score


def apply_edge_warp_vertical_seam(tile: Tile, shift_map: np.ndarray, seam_band_px: int) -> None:
    band = min(seam_band_px, tile.w)
    h = tile.img.shape[0]
    band_img = tile.img[:, -band:]
    grid_x, grid_y = np.meshgrid(np.arange(band), np.arange(h))
    map_x = (grid_x + shift_map[:, :band]).astype(np.float32)
    map_y = grid_y.astype(np.float32)
    warped = cv2.remap(band_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tile.img[:, -band:] = warped


def apply_edge_warp_horizontal_seam(tile: Tile, shift_map: np.ndarray, seam_band_px: int) -> None:
    band = min(seam_band_px, tile.h)
    w = tile.img.shape[1]
    band_img = tile.img[-band:, :]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(band))
    map_x = grid_x.astype(np.float32)
    map_y = (grid_y + shift_map[:band, :]).astype(np.float32)
    warped = cv2.remap(band_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tile.img[-band:, :] = warped


def _apply_edge_warp_if_improves(left: Optional[Tile], top: Optional[Tile], tile: Tile, params: StitchParams) -> None:
    if left:
        before = seam_error(left.img[:, -params.seam_band_px :], tile.img[:, : params.seam_band_px])
        dx, _, _ = match_strips_left_right(left.img, tile.img, params.overlap_max, min(left.bg_threshold, tile.bg_threshold))
        shift = int(np.clip(-dx, -params.max_edge_warp_px, params.max_edge_warp_px))
        if shift != 0:
            backup = left.img[:, -params.seam_band_px :].copy()
            shift_map = np.full((left.h, params.seam_band_px), shift, dtype=np.float32)
            apply_edge_warp_vertical_seam(left, shift_map, params.seam_band_px)
            after = seam_error(left.img[:, -params.seam_band_px :], tile.img[:, : params.seam_band_px])
            if after >= before:
                left.img[:, -params.seam_band_px :] = backup
    if top:
        before = seam_error(top.img[-params.seam_band_px :, :], tile.img[: params.seam_band_px, :])
        _, dy, _ = match_strips_top_bottom(top.img, tile.img, params.overlap_max, min(top.bg_threshold, tile.bg_threshold))
        shift = int(np.clip(-dy, -params.max_edge_warp_px, params.max_edge_warp_px))
        if shift != 0:
            backup = top.img[-params.seam_band_px :, :].copy()
            shift_map = np.full((params.seam_band_px, top.w), shift, dtype=np.float32)
            apply_edge_warp_horizontal_seam(top, shift_map, params.seam_band_px)
            after = seam_error(top.img[-params.seam_band_px :, :], tile.img[: params.seam_band_px, :])
            if after >= before:
                top.img[-params.seam_band_px :, :] = backup


def apply_scale(tile: Tile, scale_x: float, scale_y: float) -> None:
    if abs(scale_x - 1.0) < 1e-4 and abs(scale_y - 1.0) < 1e-4:
        return
    new_w = max(1, int(round(tile.w * scale_x)))
    new_h = max(1, int(round(tile.h * scale_y)))
    tile.img = cv2.resize(tile.img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    tile.w = new_w
    tile.h = new_h


def refine_layout_iterative(
    rows: List[List[Tile]],
    params: StitchParams,
    on_log: Optional[Callable[[str], None]],
) -> None:
    for iteration in range(params.refine_iterations):
        if on_log:
            on_log(f"Restoration refine iteration {iteration + 1}/{params.refine_iterations}")
        for row_idx, row in enumerate(rows):
            for col_idx, tile in enumerate(row):
                proposals = []
                if col_idx > 0:
                    left = row[col_idx - 1]
                    dx, dy, scale_x, scale_y, score = estimate_affine_to_neighbor(left, tile, params)
                    proposals.append(
                        {
                            "x": left.offset_x + left.w + dx,
                            "y": left.offset_y + dy,
                            "scale_x": scale_x,
                            "scale_y": scale_y,
                            "score": max(score, 0.05),
                            "left": left,
                            "top": None,
                        }
                    )
                if row_idx > 0 and col_idx < len(rows[row_idx - 1]):
                    top = rows[row_idx - 1][col_idx]
                    dx, dy, scale_x, scale_y, score = estimate_affine_to_neighbor_vertical(top, tile, params)
                    proposals.append(
                        {
                            "x": top.offset_x + dx,
                            "y": top.offset_y + top.h + dy,
                            "scale_x": scale_x,
                            "scale_y": scale_y,
                            "score": max(score, 0.05),
                            "left": None,
                            "top": top,
                        }
                    )
                if proposals:
                    total = sum(p["score"] for p in proposals)
                    tile.offset_x = int(sum(p["x"] * p["score"] for p in proposals) / total)
                    tile.offset_y = int(sum(p["y"] * p["score"] for p in proposals) / total)
                    scale_x = sum(p["scale_x"] * p["score"] for p in proposals) / total
                    scale_y = sum(p["scale_y"] * p["score"] for p in proposals) / total
                    tile.scale_x = scale_x
                    tile.scale_y = scale_y
                    apply_scale(tile, tile.scale_x, tile.scale_y)
                    left_ref = next((p["left"] for p in proposals if p["left"] is not None), None)
                    top_ref = next((p["top"] for p in proposals if p["top"] is not None), None)
                    _apply_edge_warp_if_improves(left_ref, top_ref, tile, params)
        enforce_zero_gaps(rows, params, on_log)


def enforce_zero_gaps(rows: List[List[Tile]], params: StitchParams, on_log: Optional[Callable[[str], None]]) -> None:
    for row_idx, row in enumerate(rows):
        for col_idx, tile in enumerate(row):
            if col_idx > 0:
                left = row[col_idx - 1]
                gap = tile.offset_x - (left.offset_x + left.w)
                if gap > 0:
                    tile.offset_x -= gap
                    if on_log:
                        on_log(f"Gap closed (left) {tile.x},{tile.y}: {gap}px")
            if row_idx > 0 and col_idx < len(rows[row_idx - 1]):
                top = rows[row_idx - 1][col_idx]
                gap = tile.offset_y - (top.offset_y + top.h)
                if gap > 0:
                    tile.offset_y -= gap
                    if on_log:
                        on_log(f"Gap closed (top) {tile.x},{tile.y}: {gap}px")


def build_seam_band_mask(tiles: List[Tile], shape: Tuple[int, int], seam_band_px: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for tile in tiles:
        x0 = max(tile.offset_x, 0)
        y0 = max(tile.offset_y, 0)
        x1 = min(tile.offset_x + tile.w, width)
        y1 = min(tile.offset_y + tile.h, height)
        band = seam_band_px
        cv2.rectangle(mask, (x0, y0), (x1, min(y0 + band, y1)), 1, -1)
        cv2.rectangle(mask, (x0, max(y1 - band, y0)), (x1, y1), 1, -1)
        cv2.rectangle(mask, (x0, y0), (min(x0 + band, x1), y1), 1, -1)
        cv2.rectangle(mask, (max(x1 - band, x0), y0), (x1, y1), 1, -1)
    return mask


def build_gap_mask(canvas: np.ndarray, filled_mask: np.ndarray, tiles: List[Tile], params: StitchParams) -> np.ndarray:
    seam_band = build_seam_band_mask(tiles, canvas.shape[:2], params.seam_band_px)
    gap_mask = (filled_mask == 0) & (seam_band > 0)
    return gap_mask.astype(np.uint8)


def edge_extend_fill_gaps(canvas: np.ndarray, gap_mask: np.ndarray, max_px: int = 2) -> np.ndarray:
    filled = canvas.copy()
    mask = gap_mask.copy().astype(bool)
    for _ in range(max_px):
        if not mask.any():
            break
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dy, dx = shift
            src = np.roll(filled, shift=(dy, dx), axis=(0, 1))
            src_mask = np.roll(~mask, shift=(dy, dx), axis=(0, 1))
            fill_here = mask & src_mask
            filled[fill_here] = src[fill_here]
            mask[fill_here] = False
    return filled


def seam_fill_inpaint_local(
    canvas: np.ndarray,
    gap_mask: np.ndarray,
    params: StitchParams,
    debug_dir: Optional[Path],
) -> np.ndarray:
    if gap_mask is None or gap_mask.sum() == 0:
        return canvas
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(gap_mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(gap_mask, dtype=np.uint8)
    for label in range(1, label_count):
        x, y, w, h, area = stats[label]
        if area == 0:
            continue
        if max(w, h) <= params.seam_fill_max_px:
            filtered[labels == label] = 1
    gap_mask = filtered
    if gap_mask.sum() == 0:
        return canvas
    hsv = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    paper_mask = (val > max(200, params.bg_threshold - 10)) & (sat < 60)
    inpaint_mask = (gap_mask > 0) & paper_mask
    if not inpaint_mask.any():
        return canvas
    mask_u8 = inpaint_mask.astype(np.uint8) * 255
    if debug_dir:
        cv2.imwrite(str(debug_dir / "gap_mask_before.png"), gap_mask * 255)
        cv2.imwrite(str(debug_dir / "seam_fill_mask.png"), mask_u8)
    filled = cv2.inpaint(canvas, mask_u8, params.inpaint_radius, cv2.INPAINT_TELEA)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "seam_fill_result.png"), cv2.cvtColor(filled, cv2.COLOR_RGB2BGR))
    return filled


def compose_canvas(tiles: List[Tile], params: StitchParams, return_mask: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    min_x = min(tile.offset_x for tile in tiles)
    min_y = min(tile.offset_y for tile in tiles)
    for tile in tiles:
        tile.offset_x -= min_x
        tile.offset_y -= min_y
    max_x = max(tile.offset_x + tile.w for tile in tiles)
    max_y = max(tile.offset_y + tile.h for tile in tiles)
    canvas = np.full((max_y, max_x, 3), 255, dtype=np.uint8)
    filled = np.zeros((max_y, max_x), dtype=np.uint8)

    for tile in tiles:
        x0 = tile.offset_x
        y0 = tile.offset_y
        x1 = x0 + tile.w
        y1 = y0 + tile.h
        region = canvas[y0:y1, x0:x1]
        region_filled = filled[y0:y1, x0:x1]
        tile_mask = make_content_mask(tile.img, tile.bg_threshold)
        tile_mask = clean_mask(tile_mask)
        existing_mask = make_content_mask(region, tile.bg_threshold)
        existing_mask = clean_mask(existing_mask)
        overlap_mask = region_filled > 0
        if overlap_mask.any():
            ys, xs = np.where(overlap_mask)
            min_ox, max_ox = xs.min(), xs.max()
            min_oy, max_oy = ys.min(), ys.max()
            alpha = np.ones((tile.h, tile.w), dtype=np.float32)
            overlap_width = max_ox - min_ox + 1
            overlap_height = max_oy - min_oy + 1
            if overlap_width <= overlap_height:
                width = min(max_ox + 1, params.overlap_max)
                width = int(np.clip(width, 5, 15))
                ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
                alpha[:, :width] = np.minimum(alpha[:, :width], ramp)
            else:
                height = min(max_oy + 1, params.overlap_max)
                height = int(np.clip(height, 5, 15))
                ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
                alpha[:height, :] = np.minimum(alpha[:height, :], ramp)
            alpha = alpha[..., None]
            blended = (alpha * tile.img.astype(np.float32) + (1 - alpha) * region.astype(np.float32))
            overlap_content = (tile_mask > 0) & (existing_mask > 0)
            region[overlap_content] = np.clip(blended, 0, 255).astype(np.uint8)[overlap_content]
            only_new = (tile_mask > 0) & (~overlap_content)
            region[only_new] = tile.img[only_new]
        else:
            region[tile_mask > 0] = tile.img[tile_mask > 0]
        filled[y0:y1, x0:x1] = 1
    if return_mask:
        return canvas, filled
    return canvas, None


def build_preview(tiles: List[Tile], preview_max_size: int, with_labels: bool = False) -> np.ndarray:
    min_x = min(tile.offset_x for tile in tiles)
    min_y = min(tile.offset_y for tile in tiles)
    max_x = max(tile.offset_x + tile.w for tile in tiles)
    max_y = max(tile.offset_y + tile.h for tile in tiles)
    width = max_x - min_x
    height = max_y - min_y
    scale = min(preview_max_size / max(width, height), 1.0)
    preview = np.full((int(height * scale) + 2, int(width * scale) + 2, 3), 255, dtype=np.uint8)

    for tile in tiles:
        x0 = int((tile.offset_x - min_x) * scale)
        y0 = int((tile.offset_y - min_y) * scale)
        x1 = int((tile.offset_x - min_x + tile.w) * scale)
        y1 = int((tile.offset_y - min_y + tile.h) * scale)
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 120, 255), 1)
        if with_labels:
            cv2.putText(
                preview,
                f"{tile.x},{tile.y}",
                (x0 + 4, max(12, y0 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return preview


def save_tiff(img: np.ndarray, output_path: Path, compression: str) -> None:
    pillow_img = Image.fromarray(img)
    compress_map = {
        "none": "raw",
        "lzw": "tiff_lzw",
        "deflate": "tiff_adobe_deflate",
    }
    comp = compress_map[compression]
    bytes_per_pixel = 3
    total_bytes = img.shape[0] * img.shape[1] * bytes_per_pixel
    bigtiff = total_bytes > (4 * 1024**3)
    pillow_img.save(
        output_path,
        format="TIFF",
        compression=comp,
        bigtiff=bigtiff,
    )


def collect_tiles(
    paths: List[Path],
    params: StitchParams,
    on_log: Optional[Callable[[str], None]],
    on_progress: Optional[Callable[[int], None]],
    cancel_flag,
) -> List[Tile]:
    tiles: List[Tile] = []
    max_workers = min(4, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for path in paths:
            coords = parse_tile_coords(path.name)
            if coords is None:
                if on_log:
                    on_log(f"Skipping invalid tile name: {path.name}")
                continue
            future_map[executor.submit(preprocess_tile, path, params, on_log)] = path
        total = max(len(future_map), 1)
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            _check_cancel(cancel_flag)
            tiles.append(future.result())
            completed += 1
            if on_progress:
                on_progress(int((completed / total) * 50))
    return tiles


def write_debug_positions(tiles: List[Tile], debug_dir: Optional[Path]) -> None:
    if not debug_dir:
        return
    positions = {
        f"{tile.x},{tile.y}": {
            "x": tile.offset_x,
            "y": tile.offset_y,
            "w": tile.w,
            "h": tile.h,
            "angle": tile.angle,
            "scale_x": tile.scale_x,
            "scale_y": tile.scale_y,
        }
        for tile in tiles
    }
    (debug_dir / "positions.json").write_text(json.dumps(positions, indent=2))


def save_debug_package(
    tiles: List[Tile],
    canvas: np.ndarray,
    metrics_before: List[Dict[str, float]],
    metrics_after: List[Dict[str, float]],
    gap_mask_before: Optional[np.ndarray],
    gap_mask_after: Optional[np.ndarray],
    params: StitchParams,
    output_file: Path,
) -> None:
    if not params.debug or params.debug_dir is None:
        return
    debug_dir = params.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    preview = build_preview(tiles, params.preview_max_size, with_labels=True)
    cv2.imwrite(str(debug_dir / "preview_layout.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    heatmap = np.zeros(canvas.shape[:2], dtype=np.float32)
    tile_map = {f"{tile.x},{tile.y}": tile for tile in tiles}
    for data in metrics_after:
        tile = tile_map.get(data["tile"])
        if tile is None:
            continue
        if data["direction"] == "left":
            x = max(tile.offset_x, 0)
            y0 = max(tile.offset_y, 0)
            y1 = min(tile.offset_y + tile.h, heatmap.shape[0])
            heatmap[y0:y1, max(0, x - 1) : min(heatmap.shape[1], x + 1)] = max(data["seam_error"], 1.0)
        else:
            y = max(tile.offset_y, 0)
            x0 = max(tile.offset_x, 0)
            x1 = min(tile.offset_x + tile.w, heatmap.shape[1])
            heatmap[max(0, y - 1) : min(heatmap.shape[0], y + 1), x0:x1] = max(data["seam_error"], 1.0)
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(debug_dir / "seam_heatmap.png"), heatmap_color)

    seams_dir = debug_dir / "seams"
    seams_dir.mkdir(exist_ok=True)
    worst = sorted(metrics_after, key=lambda d: d["seam_error"], reverse=True)[:5]
    for idx, data in enumerate(worst, start=1):
        tile = tile_map.get(data["tile"])
        neighbor = tile_map.get(data["neighbor"])
        if tile is None or neighbor is None:
            continue
        if data["direction"] == "left":
            band = min(params.seam_band_px, tile.w, neighbor.w)
            strip_left = neighbor.img[:, -band:]
            strip_right = tile.img[:, :band]
            diff = cv2.absdiff(strip_left, strip_right)
            cv2.imwrite(str(seams_dir / f"{idx:02d}_left_strip.png"), cv2.cvtColor(strip_left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(seams_dir / f"{idx:02d}_right_strip.png"), cv2.cvtColor(strip_right, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(seams_dir / f"{idx:02d}_diff.png"), cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
        else:
            band = min(params.seam_band_px, tile.h, neighbor.h)
            strip_top = neighbor.img[-band:, :]
            strip_bottom = tile.img[:band, :]
            diff = cv2.absdiff(strip_top, strip_bottom)
            cv2.imwrite(str(seams_dir / f"{idx:02d}_top_strip.png"), cv2.cvtColor(strip_top, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(seams_dir / f"{idx:02d}_bottom_strip.png"), cv2.cvtColor(strip_bottom, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(seams_dir / f"{idx:02d}_diff.png"), cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))

    if gap_mask_before is not None:
        cv2.imwrite(str(debug_dir / "gap_mask_before.png"), gap_mask_before * 255)
    if gap_mask_after is not None:
        cv2.imwrite(str(debug_dir / "gap_mask_after.png"), gap_mask_after * 255)

    report = {
        "tiles": [
            {
                "id": f"{tile.x},{tile.y}",
                "offset_x": tile.offset_x,
                "offset_y": tile.offset_y,
                "scale_x": tile.scale_x,
                "scale_y": tile.scale_y,
                "angle": tile.angle,
            }
            for tile in tiles
        ],
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "iterations": params.refine_iterations if params.restoration_mode else 0,
    }
    (debug_dir / "debug_report.json").write_text(json.dumps(report, indent=2))
    preview_png = debug_dir / f"{output_file.stem}_preview.png"
    cv2.imwrite(str(preview_png), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    try:
        import shutil

        shutil.copy2(output_file, debug_dir / output_file.name)
    except Exception:
        logging.exception("Failed to copy output into debug package")


def prepare_tiles(
    paths: List[Path],
    params: StitchParams,
    on_progress: Optional[Callable[[int], None]],
    on_log: Optional[Callable[[str], None]],
    cancel_flag,
) -> Tuple[List[Tile], List[Dict[str, float]], List[Dict[str, float]]]:
    start = time.time()
    if on_progress:
        on_progress(0)
    tiles = collect_tiles(paths, params, on_log, on_progress, cancel_flag)
    if not tiles:
        raise ValueError("No valid tiles found")

    rows = build_rows(tiles)
    initial_layout(rows)
    refine_layout(rows, params)
    metrics_before = compute_seam_metrics_map(rows, params.seam_band_px)
    if params.restoration_mode:
        refine_layout_iterative(rows, params, on_log)
    metrics_after = compute_seam_metrics_map(rows, params.seam_band_px)
    if on_progress:
        on_progress(70)

    ordered_tiles = [tile for row in rows for tile in row]
    if on_progress:
        on_progress(75)
    if on_log:
        on_log(f"Prepared {len(ordered_tiles)} tiles in {time.time() - start:.2f}s")
    return ordered_tiles, metrics_before, metrics_after


def stitch_tiles(
    paths: List[str],
    output_path: str,
    params: StitchParams,
    on_progress: Optional[Callable[[int], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel_flag=None,
) -> str:
    return run_stitching(paths, output_path, params, on_progress, on_log, cancel_flag)


def run_stitching(
    paths: List[str],
    output_path: str,
    params: StitchParams,
    on_progress: Optional[Callable[[int], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel_flag=None,
    on_preview: Optional[Callable[[np.ndarray], None]] = None,
) -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start = time.time()

    output_clean = output_path.strip().strip('"')
    output_file = Path(output_clean)
    if output_file.suffix.lower() not in {".tif", ".tiff"}:
        output_file = output_file.with_suffix(".tif")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_dir():
        raise ValueError(f"Output path is a directory: {output_file}")

    if params.debug:
        if params.debug_dir is None:
            params.debug_dir = output_file.parent / "debug"
        params.debug_dir.mkdir(exist_ok=True, parents=True)

    tile_paths = [Path(p) for p in paths]
    tiles, metrics_before, metrics_after = prepare_tiles(tile_paths, params, on_progress, on_log, cancel_flag)
    _check_cancel(cancel_flag)

    if on_preview:
        preview = build_preview(tiles, params.preview_max_size)
        on_preview(preview)

    if on_progress:
        on_progress(80)
    canvas, filled_mask = compose_canvas(tiles, params, return_mask=True)
    gap_mask_before = None
    gap_mask_after = None
    if filled_mask is None:
        filled_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    gap_mask_before = build_gap_mask(canvas, filled_mask, tiles, params)
    if params.restoration_mode and gap_mask_before.any():
        canvas = edge_extend_fill_gaps(canvas, gap_mask_before, max_px=2)
    gap_mask_after = build_gap_mask(canvas, filled_mask, tiles, params)
    if params.seam_fill_enabled and params.restoration_mode and gap_mask_after.any():
        canvas = seam_fill_inpaint_local(canvas, gap_mask_after, params, params.debug_dir)
    if on_progress:
        on_progress(90)

    write_debug_positions(tiles, params.debug_dir if params.debug else None)
    if on_progress:
        on_progress(95)
    save_tiff(canvas, output_file, params.compression)
    save_debug_package(
        tiles,
        canvas,
        metrics_before,
        metrics_after,
        gap_mask_before if params.debug else None,
        gap_mask_after if params.debug else None,
        params,
        output_file,
    )

    if on_progress:
        on_progress(100)
    if on_log:
        on_log(
            f"Saved {output_file} ({canvas.shape[1]}x{canvas.shape[0]}), tiles={len(tiles)}, time={time.time() - start:.2f}s"
        )
    return str(output_file)
