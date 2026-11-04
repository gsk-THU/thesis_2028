"""Step 2: 外底分割（轮廓 + 花纹块/沟槽）。

双通道设计：
- classical: Otsu/自适应阈值 + 形态学（可复现、无依赖）
- sam: Segment Anything (vit_b) 框提示轮廓（更稳的外形先验）
轮廓融合后，在矫正图上做花纹块 vs 沟槽的二分割（确定性 CV）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import keep_components, largest_component


# ---------------- 轮廓 ----------------

def outline_classical(bgr: np.ndarray, min_area_frac: float = 0.03) -> np.ndarray:
    """经典轮廓分割：以四角背景中位数为参考，取与背景亮度差显著的大块区域。

    对彩色外底照片与凝胶印痕（暗花纹、白背景）均适用；抗黑角/暗晕污染。
    返回 uint8 0/1。
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    k = max(int(min(H, W) * 0.08), 8)
    corners = np.concatenate([
        gray[:k, :k].ravel(), gray[:k, -k:].ravel(),
        gray[-k:, :k].ravel(), gray[-k:, -k:].ravel()])
    bg_med = float(np.median(corners))
    delta = max(35.0, 0.18 * bg_med)
    if bg_med > 127:
        cand = (gray < bg_med - delta)
    else:
        cand = (gray > bg_med + delta)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = largest_component(mask)
    if mask.sum() < min_area_frac * H * W:
        mask = largest_component(_canny_filled(gray))
    if mask.sum() < min_area_frac * H * W:
        return np.zeros((H, W), np.uint8)
    mask = _fill_holes(mask)
    mask = largest_component(mask)
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """填充 mask 内部孔洞（保留外边界）。"""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=1).astype(np.uint8)
    flood = padded.copy()
    ffmask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 0)
    holes = (flood > 0)
    holes[1:-1, 1:-1] &= (mask == 0)
    out = mask.copy()
    out[holes[1:-1, 1:-1]] = 1
    return out


def _canny_filled(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(gray)
    if cnts:
        best = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [best], -1, 1, -1)
    return out


class SamOutline:
    """SAM vit_b 框提示轮廓分割器。权重缺失时不可用。"""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        self.torch = torch
        self.model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.model.to(device).eval()
        self.predictor = SamPredictor(self.model)
        self.device = device

    def outline(self, bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        box = np.array(box_xyxy, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(box=box[None, :], multimask_output=True)
        i = int(np.argmax(scores))
        return masks[i].astype(np.uint8)


def outline_fuse(classical: np.ndarray, sam: Optional[np.ndarray]) -> np.ndarray:
    """融合策略:
    - 经典结果失败（<0.5×SAM）→ 信 SAM
    - 否则取 SAM ∩ dilate(classical, 9px)：允许 SAM 修正边界（≤9px），
      同时防止其把邻近背景并入（合成/真实照片均验证）
    """
    if sam is None or sam.sum() == 0:
        return largest_component(classical)
    cls_area, sam_area = classical.sum(), sam.sum()
    if cls_area < 0.5 * sam_area:
        return largest_component(sam.astype(np.uint8))
    inter = np.logical_and(sam, classical).sum()
    iou = inter / (np.logical_or(sam, classical).sum() + 1e-6)
    if iou > 0.6:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        allowed = cv2.dilate(classical, kernel) > 0
        merged = largest_component(np.logical_and(sam, allowed).astype(np.uint8))
        if 0.75 * cls_area <= merged.sum() <= 1.25 * cls_area:
            return merged
    return largest_component(classical)


# ---------------- 花纹块 / 沟槽 ----------------

def _tread_split_polarity(bgr: np.ndarray, outline_mask: np.ndarray,
                          groove_dark: bool, min_groove_area_px: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inner = outline_mask.astype(np.uint8)
    g = gray.astype(np.uint8)
    if not groove_dark:              # 沟槽偏亮（浅色底/凝胶印痕）→ 灰度取反统一成"沟槽暗"
        g = 255 - g
    # 1) mask 内全局 Otsu（光照均匀时的首选），沟槽占比落在合理区间才采纳
    thr, _ = cv2.threshold(g[inner > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grooves = ((g < thr) & (inner > 0)).astype(np.uint8)
    frac = grooves.sum() / max(inner.sum(), 1)
    if not (0.06 <= frac <= 0.55):
        # 2) 光照不均 → 高斯自适应阈值兜底
        t = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, blockSize=31, C=5)
        grooves = (t * inner).astype(np.uint8)
    grooves = keep_components(grooves, min_groove_area_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grooves = cv2.morphologyEx(grooves, cv2.MORPH_OPEN, kernel)
    return (grooves * inner).astype(np.uint8)


def _polarity_score(bgr: np.ndarray, blocks: np.ndarray, grooves: np.ndarray,
                    outline_mask: np.ndarray) -> float:
    """极性合理性评分: 沟槽比块面更暗(物理先验,决定性) + 接触比合理 + 周期显著。"""
    from .pattern import _principal_axis_periodicity
    area = max(int(outline_mask.sum()), 1)
    if grooves.sum() < 0.02 * area:
        return -1.0                          # 没有可见沟槽 → 极性必然错误
    contact = blocks.sum() / area
    if not (0.25 <= contact <= 0.92):
        return -1.0
    # 物理先验: 沟槽应比块顶面暗（真实鞋底照片）；凝胶印痕中花纹块为暗色印迹、
    # 空隙接近背景白 → 同样满足"块间空隙更亮"，以灰度符号一致性表达
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gm = float(np.median(gray[grooves > 0])) if grooves.any() else 0.0
    bm = float(np.median(gray[blocks > 0])) if blocks.any() else 0.0
    sign = np.tanh((bm - gm) / 15.0)         # 沟槽更暗 → 正;更亮 → 负
    _, score, _ = _principal_axis_periodicity(blocks, 1.0)
    sane = 1.0 - abs(contact - 0.62) / 0.62
    return float(sign) * (0.45 + 0.25 * max(sane, 0.0) + 0.3 * score)


def tread_split(bgr: np.ndarray, outline_mask: np.ndarray,
                groove_dark: str = "auto",
                min_groove_area_px: int = 40) -> tuple:
    """在外底内部划分花纹块（凸起）与沟槽（凹陷/缝隙）。

    groove_dark: 'auto' 双极性试探自动选择 / True(沟槽暗) / False(沟槽亮)。
    返回 (blocks, grooves, groove_dark_actual)。
    """
    if groove_dark == "auto":
        best = None
        for dark in (True, False):
            g = _tread_split_polarity(bgr, outline_mask, dark, min_groove_area_px)
            b = (outline_mask & (1 - g)).astype(np.uint8)
            s = _polarity_score(bgr, b, g, outline_mask)
            if best is None or s > best[0]:
                best = (s, dark, b, g)
        return best[2], best[3], best[1]
    g = _tread_split_polarity(bgr, outline_mask, bool(groove_dark), min_groove_area_px)
    b = (outline_mask & (1 - g)).astype(np.uint8)
    return b, g, bool(groove_dark)
