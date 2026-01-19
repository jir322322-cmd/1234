from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from models import StitchParams, Tile
from utils import (
    crop_to_mask,
    estimate_rotation_angle,
    make_content_mask,
    match_strips_left_right,
    match_strips_top_bottom,
    parse_tile_coords,
    rotate_image,
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
    mask = make_content_mask(img, params.bg_threshold)
    cropped, cropped_mask = crop_to_mask(img, mask)
    if params.debug and params.debug_dir:
        cv2.imwrite(str(params.debug_dir / f"{path.stem}_crop_before.png"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    angle = estimate_rotation_angle(cropped_mask, params.max_angle)
    rotated = rotate_image(cropped, angle)
    rotated_mask = make_content_mask(rotated, params.bg_threshold)
    rotated, rotated_mask = crop_to_mask(rotated, rotated_mask)
    if params.debug and params.debug_dir:
        cv2.imwrite(str(params.debug_dir / f"{path.stem}_crop_after.png"), cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
    h, w = rotated.shape[:2]
    coords = parse_tile_coords(path.name)
    if coords is None:
        raise ValueError(f"Invalid tile filename {path.name}")
    if on_log:
        on_log(f"Processed tile {path.name}: angle={angle:.2f}, size={w}x{h}")
    return Tile(x=coords[0], y=coords[1], path=path, img=rotated, w=w, h=h, angle=angle)


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
                dx, dy, score = match_strips_left_right(left.img, tile.img, params.overlap_max)
                if params.debug and params.debug_dir:
                    debug_path = params.debug_dir / f"match_left_{tile.x}_{tile.y}.json"
                    debug_path.write_text(json.dumps({"dx": dx, "dy": dy, "score": score}))
                if score < 0.2:
                    proposals.append((left.offset_x + left.w, left.offset_y, 0.1))
                else:
                    proposals.append((left.offset_x + left.w + dx, left.offset_y + dy, score))
            if row_idx > 0:
                top = rows[row_idx - 1][col_idx]
                dx, dy, score = match_strips_top_bottom(top.img, tile.img, params.overlap_max)
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


def compose_canvas(tiles: List[Tile]) -> np.ndarray:
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
        overlap_mask = region_filled > 0
        if overlap_mask.any():
            ys, xs = np.where(overlap_mask)
            min_ox, max_ox = xs.min(), xs.max()
            min_oy, max_oy = ys.min(), ys.max()
            alpha = np.ones((tile.h, tile.w), dtype=np.float32)
            if min_ox == 0:
                width = max_ox + 1
                ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
                alpha[:, :width] = np.minimum(alpha[:, :width], ramp)
            if min_oy == 0:
                height = max_oy + 1
                ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
                alpha[:height, :] = np.minimum(alpha[:height, :], ramp)
            alpha = alpha[..., None]
            blended = (alpha * tile.img.astype(np.float32) + (1 - alpha) * region.astype(np.float32))
            region[:] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            region[:] = tile.img
        filled[y0:y1, x0:x1] = 1
    return canvas


def build_preview(tiles: List[Tile], preview_max_size: int) -> np.ndarray:
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


def collect_tiles(paths: Iterable[Path], params: StitchParams, on_log: Optional[Callable[[str], None]], cancel_flag) -> List[Tile]:
    tiles: List[Tile] = []
    for path in paths:
        _check_cancel(cancel_flag)
        coords = parse_tile_coords(path.name)
        if coords is None:
            if on_log:
                on_log(f"Skipping invalid tile name: {path.name}")
            continue
        tiles.append(preprocess_tile(path, params, on_log))
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
        }
        for tile in tiles
    }
    (debug_dir / "positions.json").write_text(json.dumps(positions, indent=2))


def prepare_tiles(
    paths: Iterable[Path],
    params: StitchParams,
    on_progress: Optional[Callable[[int], None]],
    on_log: Optional[Callable[[str], None]],
    cancel_flag,
) -> List[Tile]:
    start = time.time()
    tiles = collect_tiles(paths, params, on_log, cancel_flag)
    if not tiles:
        raise ValueError("No valid tiles found")

    rows = build_rows(tiles)
    initial_layout(rows)
    refine_layout(rows, params)

    ordered_tiles = [tile for row in rows for tile in row]
    if on_progress:
        on_progress(60)
    if on_log:
        on_log(f"Prepared {len(ordered_tiles)} tiles in {time.time() - start:.2f}s")
    return ordered_tiles


def stitch_tiles(
    paths: List[str],
    output_path: str,
    params: StitchParams,
    on_progress: Optional[Callable[[int], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel_flag=None,
) -> None:
    run_stitching(paths, output_path, params, on_progress, on_log, cancel_flag)


def run_stitching(
    paths: List[str],
    output_path: str,
    params: StitchParams,
    on_progress: Optional[Callable[[int], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel_flag=None,
    on_preview: Optional[Callable[[np.ndarray], None]] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start = time.time()

    if params.debug:
        if params.debug_dir is None:
            params.debug_dir = Path(output_path).parent / "debug"
        params.debug_dir.mkdir(exist_ok=True, parents=True)

    tile_paths = [Path(p) for p in paths]
    tiles = prepare_tiles(tile_paths, params, on_progress, on_log, cancel_flag)
    _check_cancel(cancel_flag)

    if on_preview:
        preview = build_preview(tiles, params.preview_max_size)
        on_preview(preview)

    canvas = compose_canvas(tiles)
    if on_progress:
        on_progress(90)

    write_debug_positions(tiles, params.debug_dir if params.debug else None)
    save_tiff(canvas, Path(output_path), params.compression)

    if on_progress:
        on_progress(100)
    if on_log:
        on_log(
            f"Saved {output_path} ({canvas.shape[1]}x{canvas.shape[0]}), tiles={len(tiles)}, time={time.time() - start:.2f}s"
        )
