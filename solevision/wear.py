"""Step 5b: 磨损评估（光泽 / 纹理熵 / 平坦度）。"""
from __future__ import annotations

import cv2
import numpy as np


def analyze_wear(bgr: np.ndarray, blocks: np.ndarray, height_mm: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sel = blocks > 0
    if sel.sum() < 200:
        gloss = texture = flat = float("nan")
    else:
        v = hsv[..., 2].astype(np.float32)
        s = hsv[..., 1].astype(np.float32)
        gloss = float(((v > 242) & (s < 40))[sel].mean())          # 镜面高光占比

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        local_std = _local_std(gray)
        texture = float(local_std[sel].mean())                     # 局部纹理能量

        # 平坦度: 块区域中低图像纹理占比（磨平→镜面光滑→局部标准差趋 0）
        flat = float((local_std[sel] < 3.0).mean())

    # bald 区: 大面积高光+低纹理块
    bald_ratio = 0.0
    if sel.sum() >= 200 and gloss == gloss:
        bald_mask = ((hsv[..., 2] > 242) & (hsv[..., 1] < 40)) | ((gray == gray) & (local_std < 2.0) & sel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bald_mask.astype(np.uint8))
        if n > 1:
            biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            big_area = stats[biggest, cv2.CC_STAT_AREA]
            if big_area > 0.25 * sel.sum():
                bald_ratio = round(big_area / float(sel.sum()), 3)

    if gloss != gloss:
        state = "unknown"
    elif gloss > 0.20 and flat > 0.75:
        state = "bald"
    elif gloss > 0.10 or texture < 4.0:
        state = "worn"
    elif texture < 7.0:
        state = "light"
    else:
        state = "new"
    return {
        "state": state,
        "gloss_score": round(gloss, 3),
        "texture_index": round(texture, 2),
        "flatness_index": round(flat, 3),
        "bald_ratio": bald_ratio,
    }


def _local_std(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    mean = cv2.boxFilter(gray, -1, (ksize, ksize))
    sq = cv2.boxFilter(gray ** 2, -1, (ksize, ksize))
    return np.sqrt(np.maximum(sq - mean ** 2, 0))
