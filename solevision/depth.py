"""Step 3: 高度场（毫米级）。

provider:
- classical: 由沟槽/花纹块亮度差估计浮雕深度，再用类别先验标定（零依赖、可复现）
- depth_anything: Depth Anything V2 相对深度 + 类别先验标定（更真实的形状）

两者语义一致: height(x,y) = 该点相对"沟槽谷底"的高度（mm）；
花纹块顶面 ≈ tread block height h_prior_max，沟谷 ≈ 0。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .schema import TREAD_DEPTH_PRIOR_MM


@dataclass
class HeightResult:
    height_mm: np.ndarray          # float32, 单位 mm，outline 外为 0
    mask: np.ndarray               # uint8 0/1
    provider: str
    uncertainty_mm: Tuple[float, float]   # (mean, max)


def _prior(shoe_type: str) -> Tuple[float, float]:
    return TREAD_DEPTH_PRIOR_MM.get(shoe_type, TREAD_DEPTH_PRIOR_MM["unknown"])


def _smooth(h: np.ndarray, mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    h16 = np.clip(h, 0, 50).astype(np.float32)
    blurred = cv2.GaussianBlur(h16, (ksize, ksize), 0)
    return blurred * mask


# ---------------- classical ----------------

def height_classical(bgr: np.ndarray, blocks: np.ndarray, grooves: np.ndarray,
                     shoe_type: str = "unknown") -> HeightResult:
    h_min, h_max = _prior(shoe_type)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inner = (blocks | grooves).astype(np.uint8)

    if grooves.sum() > 0.02 * inner.sum():
        bright = gray / 255.0        # 越亮越高（沟槽暗→低；块面亮→高）
    else:                            # 近乎全平（磨平底/平底）
        bright = np.full_like(gray, 0.9)

    vals = bright[inner > 0]
    if vals.size:
        lo, hi = np.percentile(vals, 10), np.percentile(vals, 95)
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((bright - lo) / max(hi - lo, 1e-3), 0, 1)

    height = norm * h_max
    # 花纹块顶面统一到 h_max（它们才是实际接触面）
    height[blocks > 0] = h_max
    height = _smooth(height, inner)
    height[inner == 0] = 0
    unc = (0.35 * (h_max - h_min), (h_max - h_min))
    return HeightResult(height.astype(np.float32), inner, "classical", unc)


# ---------------- Depth Anything V2 ----------------

class DepthAnythingProvider:
    """加载第三方 Depth-Anything-V2 仓库（third_party 克隆）+ 本地权重。"""

    def __init__(self, repo_dir: str, checkpoint: str, device: Optional[str] = None):
        import sys
        import torch
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402（仓库无 __init__.py）

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = DepthAnythingV2(encoder="vits", features=64,
                                out_channels=[48, 96, 192, 384])
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        self.model = model
        self.device = device

    def relative_depth(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.model.infer_image(rgb)   # HxW float, 值越大越近


def height_depth_anything(provider: DepthAnythingProvider,
                          bgr: np.ndarray, blocks: np.ndarray, grooves: np.ndarray,
                          shoe_type: str = "unknown") -> HeightResult:
    h_min, h_max = _prior(shoe_type)
    rel = provider.relative_depth(bgr).astype(np.float32)
    inner = (blocks | grooves).astype(np.uint8)

    vals = rel[inner > 0]
    if vals.size:
        lo, hi = np.percentile(vals, 5), np.percentile(vals, 97)
    else:
        lo, hi = float(rel.min()), float(rel.max())
    norm = np.clip((rel - lo) / max(hi - lo, 1e-6), 0, 1)

    height = norm * h_max
    height[blocks > 0] = np.maximum(height[blocks > 0], h_max * 0.9)
    height = _smooth(height, inner)
    height[inner == 0] = 0
    unc = (0.25 * (h_max - h_min), 0.8 * (h_max - h_min))
    return HeightResult(height.astype(np.float32), inner, "depth_anything_v2_vits", unc)
