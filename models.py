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
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class StitchParams:
    bg_threshold: int = 245
    max_angle: float = 7.0
    overlap_max: int = 15
    compression: str = "deflate"
    debug: bool = False
    preview_max_size: int = 800
    debug_dir: Optional[Path] = None
