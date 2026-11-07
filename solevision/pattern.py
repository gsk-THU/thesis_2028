"""Step 4: 花纹图结构抽取（确定性 CV）。

输出: 花纹块节点表、邻接图、周期/方向、接触面积比、沟槽几何、花纹类型。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schema import TreadElement


def extract_elements(blocks: np.ndarray, height_mm: np.ndarray, pixel_mm: float,
                     min_area_mm2: float = 3.0,
                     restrict_mask: Optional[np.ndarray] = None) -> List[TreadElement]:
    if pixel_mm == pixel_mm:  # 已标定（mm/px）
        min_area_px = max(int(min_area_mm2 / max(pixel_mm, 1e-6) ** 2), 8)
    else:                     # 未标定：退化为像素面积阈值，mm 字段将为 NaN（序列化时剔除）
        min_area_px = 40
    src = blocks if restrict_mask is None else (blocks & restrict_mask).astype(np.uint8)
    # 轻微腐蚀断开花纹块间的细连接（如边缘包边），避免整底连成单元素
    eroded = cv2.erode(src, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    if eroded.sum() < 0.3 * src.sum():      # 腐蚀过度（细密花纹）→ 退回原 mask
        eroded = src
    n, labels, stats, cents = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    out: List[TreadElement] = []
    idx = 0
    for i in range(1, n):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < min_area_px:
            continue
        x, y = cents[i]
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        peri = cv2.arcLength(cnts[0], True)
        (_, _), (w, h), angle = cv2.minAreaRect(cnts[0])
        roundness = float(4 * np.pi * area_px / (peri ** 2 + 1e-6))
        cx_mm, cy_mm = x * pixel_mm, y * pixel_mm
        hx, hy = int(round(x)), int(round(y))
        h = float(height_mm[hy, hx]) if (0 <= hy < height_mm.shape[0] and 0 <= hx < height_mm.shape[1]) else 0.0
        idx += 1
        out.append(TreadElement(id=idx, type="lug",
                                centroid_xy_mm=[round(cx_mm, 2), round(cy_mm, 2)],
                                area_mm2=round(area_px * pixel_mm ** 2, 2),
                                height_mm=round(h, 2),
                                angle_deg=round(float(angle), 1),
                                roundness=round(roundness, 3)))
    return out


def build_graph(elements: List[TreadElement], max_edge_mm: float = 60.0) -> List[List[float]]:
    """质心 Delaunay 邻接图，边过长时剪枝。返回 [i, j, dist_mm]。"""
    if len(elements) < 3:
        return []
    from scipy.spatial import Delaunay
    pts = np.array([e.centroid_xy_mm for e in elements], dtype=np.float64)
    edges = set()
    try:
        tri = Delaunay(pts)
        for simplex in tri.simplices:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                i, j = int(simplex[a]), int(simplex[b])
                if i > j:
                    i, j = j, i
                d = float(np.hypot(*(pts[i] - pts[j])))
                if d <= max_edge_mm:
                    edges.add((i, j, round(d, 2)))
    except Exception:
        pass
    return [[i, j, d] for i, j, d in sorted(edges)]


def _fft2_peaks(blocks: np.ndarray, pixel_mm: float,
                period_lo_mm: float = 4.0, period_hi_mm: float = 25.0,
                max_peaks: int = 4) -> list:
    """带通预滤波 + FFT2 多峰提取。

    返回按能量降序的峰列表: [{'period_mm', 'energy', 'direction_deg'}, ...]。
    预滤波去除鞋底外形包络的低频能量，使花纹基频成为主峰。
    """
    m = (blocks > 0).astype(np.float32)
    h, w = m.shape
    if m.sum() < 200:
        return []
    if pixel_mm == pixel_mm:  # 已标定：按 mm 带宽换算
        sigma = max((period_hi_mm / max(pixel_mm, 1e-6)) / 2.0, 4.0)
        f_lo = 1.0 / (period_hi_mm / pixel_mm)
        f_hi = 1.0 / (period_lo_mm / pixel_mm)
    else:                     # 未标定：纯像素域默认（周期 8~60 px）
        sigma, f_lo, f_hi = 20.0, 1.0 / 60.0, 1.0 / 8.0
    z = m - cv2.GaussianBlur(m, (0, 0), sigma)
    z = z * np.hanning(h).astype(np.float32)[:, None] * np.hanning(w).astype(np.float32)[None, :]
    F = np.fft.fftshift(np.abs(np.fft.fft2(z)) ** 2)
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    freq = np.sqrt(fy ** 2 + fx ** 2)
    band = (freq >= f_lo) & (freq <= f_hi)
    if not band.any():
        return []
    total = float(F[band].sum())
    if total <= 0:
        return []

    work = np.where(band, F, 0.0).copy()
    peaks = []
    for _ in range(max_peaks):
        iy, ix = np.unravel_index(np.argmax(work), work.shape)
        energy = float(work[iy, ix])
        if energy <= 0:
            break
        f_mag = float(freq[iy, ix])
        theta = float(np.degrees(np.arctan2(float(fy[iy, 0]), float(fx[0, ix]))))
        peaks.append({
            "period_mm": round(pixel_mm / f_mag, 2) if f_mag > 0 else None,
            "energy": energy,
            "direction_deg": round((theta + 90.0) % 180.0, 1),
        })
        # 抑制邻域，寻找次峰
        y0, y1 = max(0, iy - 6), min(work.shape[0], iy + 7)
        x0, x1 = max(0, ix - 6), min(work.shape[1], ix + 7)
        work[y0:y1, x0:x1] = 0
    # 相对能量
    base = peaks[0]["energy"]
    for p in peaks:
        p["rel"] = round(p["energy"] / base, 3)
        del p["energy"]
    return peaks


def _principal_axis_periodicity(blocks: np.ndarray, pixel_mm: float) -> Tuple[Optional[float], float, Optional[float]]:
    """兼容旧接口: 返回 (period_mm, score, direction_deg)。"""
    peaks = _fft2_peaks(blocks, pixel_mm)
    if not peaks:
        return None, 0.0, None
    return peaks[0]["period_mm"], peaks[0]["rel"], peaks[0]["direction_deg"]


def classify_pattern(elements: List[TreadElement], period_mm: Optional[float],
                     score: float, contact_ratio: float, directions: List[float]) -> str:
    """基于接触比 + 周期显著性 + 方向模式数的规则分类。

    矫正图上鞋长轴≈竖直(90°)。herringbone 两峰对称分布于长轴两侧；
    waffle 两峰接近 0°/90° 正交；单峰 → directional（波浪/平行条纹）。
    """
    if contact_ratio > 0.88 and len(elements) <= 6:
        return "smooth"
    if score < 0.12 or period_mm is None:
        return "random_lug" if len(elements) >= 12 else "unknown"

    # 合并相近方向（±12°）
    modes: List[float] = []
    for d in directions:
        if not any(min(abs(d - m), 180 - abs(d - m)) < 12 for m in modes):
            modes.append(d)
    if len(modes) == 1:
        return "directional"
    if len(modes) == 2:
        a, b = sorted(modes)
        dev = [min(abs(x - 90), 90) for x in (a, b)]
        if all(25 <= v <= 65 for v in dev):
            return "herringbone"
        near_axis = all(min(abs(x % 90), 90 - (x % 90)) < 15 for x in (a, b))
        if near_axis:
            return "waffle"
        return "directional"
    return "random_lug"


def groove_geometry(grooves: np.ndarray, height_mm: np.ndarray, pixel_mm: float,
                    blocks: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[float]]:
    """(groove_width_mm, groove_depth_mm)。宽度取距离变换脊线（局部极大≈中轴线）。"""
    if grooves.sum() == 0:
        return None, None
    dist = cv2.distanceTransform(grooves, cv2.DIST_L2, 5).astype(np.float32)
    from scipy.ndimage import maximum_filter
    ridge = (dist == maximum_filter(dist, footprint=np.ones((5, 5)))) & (grooves > 0) & (dist > 0.5)
    if ridge.sum() >= 10:
        width_px = float(np.median(dist[ridge]) * 2.0)
    else:
        width_px = float(np.median(dist[grooves > 0]) * 2.0)
    # 沟深: 块顶面高度中位数 - 沟底高度中位数
    if blocks is not None and (blocks > 0).any():
        block_h = float(np.median(height_mm[blocks > 0]))
    else:
        block_h = float(np.median(height_mm[height_mm > 0])) if (height_mm > 0).any() else 0.0
    groove_h = float(np.median(height_mm[grooves > 0])) if grooves.any() else 0.0
    depth = max(block_h - groove_h, 0.0)
    return round(width_px * pixel_mm, 2), round(depth, 2)


def analyze(blocks: np.ndarray, grooves: np.ndarray, height_mm: np.ndarray,
            pixel_mm: float, outline_area_px: int,
            outline_mask: Optional[np.ndarray] = None) -> dict:
    """汇总花纹分析。返回 schema.Pattern 所需字段。"""
    # 元素提取限制在去除边缘包边的内部区域（包边会把所有块连成一个连通域）
    interior = None
    if outline_mask is not None:
        interior = cv2.erode(outline_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if interior.sum() < 0.3 * outline_mask.sum():
            interior = None
    elements = extract_elements(blocks, height_mm, pixel_mm, restrict_mask=interior)
    peaks = _fft2_peaks(blocks, pixel_mm)
    if peaks:
        period_mm = peaks[0]["period_mm"]
        score = peaks[0]["rel"]
        direction = peaks[0]["direction_deg"]
        directions = [p["direction_deg"] for p in peaks if p["rel"] >= 0.4]
    else:
        period_mm, score, direction, directions = None, 0.0, None, []
    gw, gd = groove_geometry(grooves, height_mm, pixel_mm, blocks=blocks)
    contact = float(blocks.sum()) / max(outline_area_px, 1)
    pclass = classify_pattern(elements, period_mm, score, contact, directions)
    return {
        "elements": elements,
        "graph": build_graph(elements),
        "pattern_class": pclass,
        "period_mm": period_mm,
        "direction_deg": direction,
        "groove_width_mm": gw,
        "groove_depth_mm": gd,
        "contact_area_ratio": round(contact, 3),
        "periodicity_score": round(score, 3),
        "element_count": len(elements),
    }
