from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Tile:
    x: int
    y: int
    path: Path
    img: np.ndarray
    w: int
    h: int
    angle: float
    bg_threshold: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    angle_delta: float = 0.0
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class StitchParams:
    bg_threshold: int = 245
    max_angle: float = 7.0
    overlap_max: int = 15
    compression: str = "deflate"
    debug: bool = False
    restoration_mode: bool = False
    crop_padding_px: int = 4
    refine_iterations: int = 3
    seam_band_px: int = 30
    max_scale_percent: float = 0.5
    max_edge_warp_px: int = 2
    seam_fill_enabled: bool = False
    seam_fill_max_px: int = 15
    inpaint_radius: int = 3
    inpaint_method: str = "telea"
    preview_max_size: int = 800
    debug_dir: Optional[Path] = None
