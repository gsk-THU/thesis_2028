"""Step 1a: 拍摄质量门控——模糊度 / 曝光 / 可分割性。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class QualityReport:
    status: str            # pass / warn / fail
    blur_var: float
    overexposed_ratio: float
    underexposed_ratio: float
    contrast: float
    notes: List[str]

    @property
    def ok_to_process(self) -> bool:
        return self.status != "fail"


def check_quality(bgr: np.ndarray,
                  blur_fail: float = 15.0,
                  blur_warn: float = 50.0,
                  expose_fail: float = 0.35,
                  expose_warn: float = 0.12) -> QualityReport:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    notes: List[str] = []

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    frac = hist / hist.sum()
    over = float(frac[246:].sum())
    under = float(frac[:6].sum())
    contrast = float(gray.std())

    if blur_var < blur_fail:
        notes.append(f"图像过模糊 (Laplacian var={blur_var:.1f}<{blur_fail})，深度/花纹测量不可靠")
    elif blur_var < blur_warn:
        notes.append(f"图像略模糊 (var={blur_var:.1f})，建议补拍")

    if over > expose_fail or under > expose_fail:
        notes.append(f"严重欠/过曝 (over={over:.0%}, under={under:.0%})，无法处理")
    elif over > expose_warn or under > expose_warn:
        notes.append(f"存在欠/过曝区域 (over={over:.0%}, under={under:.0%})，反光区测量置信度下调")

    if contrast < 18:
        notes.append(f"对比度低 (std={contrast:.1f})，分割可能不稳")

    status = "pass"
    if blur_var < blur_fail or over > expose_fail or under > expose_fail:
        status = "fail"
    elif notes:
        status = "warn"
    return QualityReport(status, blur_var, over, under, contrast, notes)
