"""SoleVision 公共工具。"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np


def load_bgr(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)  # 兼容非 ASCII 路径
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """保留最大连通域（可设最小面积过滤）。mask: uint8 0/1"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = (labels == best).astype(np.uint8)
    if min_area > 0 and areas.max() < min_area:
        return np.zeros_like(mask)
    return out


def keep_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """过滤掉面积小于 min_area 的连通域。mask: uint8 0/1"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep


def ellipse_profile(mask: np.ndarray) -> dict:
    """椭圆拟合轮廓 mask，返回主轴信息。"""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) < 20:
        return {"angle": 0.0, "center": (0.0, 0.0), "axes": (0.0, 0.0)}
    (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
    return {"angle": float(angle), "center": (float(cx), float(cy)), "axes": (float(w), float(h))}


def rotate_mask_and_image(image: np.ndarray, mask: np.ndarray, angle_deg: float, center: tuple) -> tuple:
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    h, w = image.shape[:2]
    # 扩大画布避免裁切
    size = int(np.hypot(w, h)) + 2
    dx, dy = (size - w) // 2, (size - h) // 2
    M[0, 2] += dx
    M[1, 2] += dy
    rot_img = cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR)
    rot_mask = cv2.warpAffine(mask * 255, M, (size, size), flags=cv2.INTER_NEAREST) > 127
    return rot_img, rot_mask.astype(np.uint8)


def to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1) if x.dtype != np.uint8 else x
    return (x * 255).astype(np.uint8) if x.dtype != np.uint8 else x
