"""合成鞋底测试图生成器——带 ground truth，用于验证管线的几何测量。

用法: python tools/make_synthetic_sole.py --out data/synthetic --pattern herringbone
"""
from __future__ import annotations

import argparse
import json
import os
from math import radians, sin, cos

import cv2
import numpy as np

# 鞋码 42 → 外底长 285mm（ground truth）
L_MM, W_MM = 285.0, 104.0
SCALE = 3.0            # 渲染 px/mm
PERSPECTIVE = 0.12     # 模拟拍摄倾斜


def sole_outline(w_px: int, h_px: int) -> np.ndarray:
    """生成鞋底轮廓 mask（前掌宽、腰身收窄、后跟圆）。"""
    mask = np.zeros((h_px, w_px), np.uint8)
    pts = []
    for t in np.linspace(0, 1, 240):
        y = int(t * (h_px * 0.92) + h_px * 0.04)
        # 宽度曲线: 前掌(t~0.18)最宽，中腰(t~0.5)收窄，后跟(t~0.95)中等
        width = (0.62 + 0.38 * np.exp(-((t - 0.16) / 0.14) ** 2)
                 + 0.18 * np.exp(-((t - 0.92) / 0.12) ** 2)
                 - 0.28 * np.exp(-((t - 0.52) / 0.18) ** 2))
        ww = width * w_px
        pts.append((w_px / 2 - ww / 2, y))
        pts.append((w_px / 2 + ww / 2, y))
    pts = np.array(pts, np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 1)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    return mask


def draw_herringbone(img, mask, period_px, groove_px):
    """人字纹: 两组相反角度的斜条。"""
    h, w = mask.shape
    groove = np.zeros((h, w), np.uint8)
    ang = radians(38)
    dx, dy = cos(ang), sin(ang)
    step = period_px
    normal = np.array([-dy, dx])
    for k in range(-int((h + w) / step), int((h + w) / step)):
        base = np.array([w / 2, h / 2]) + normal * (k * step)
        p1 = (base + np.array([dx, dy]) * (-w)).astype(int)
        p2 = (base + np.array([dx, dy]) * w).astype(int)
        cv2.line(groove, tuple(p1), tuple(p2), 1, groove_px)
    # 镜像一组
    groove2 = cv2.flip(groove, 1)
    groove = ((groove | groove2) & mask).astype(np.uint8)
    return groove


def draw_waffle(img, mask, period_px, groove_px):
    h, w = mask.shape
    groove = np.zeros((h, w), np.uint8)
    for x in range(0, w, period_px):
        cv2.line(groove, (x, 0), (x, h), 1, groove_px)
    for y in range(0, h, period_px):
        cv2.line(groove, (0, y), (w, y), 1, groove_px)
    return (groove & mask).astype(np.uint8)


def draw_wave(img, mask, period_px, groove_px):
    h, w = mask.shape
    groove = np.zeros((h, w), np.uint8)
    for k in (period_px, period_px * 2):
        for x0 in range(0, w + 60, k):
            pts = [(int(x0 + 10 * np.sin(y / 60.0)), y) for y in range(0, h, 4)]
            cv2.polylines(groove, [np.array(pts)], False, 1, groove_px)
    return (groove & mask).astype(np.uint8)


def render(pattern: str, out_dir: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    w_px, h_px = int(W_MM * SCALE), int(L_MM * SCALE)
    mask = sole_outline(w_px, h_px)

    # 真实鞋底渲染: 中底浅色(EVA边) + 外底深色橡胶
    img = np.full((h_px, w_px, 3), (200, 200, 200), np.uint8)      # 背景浅灰
    sole = img.copy()
    sole[mask > 0] = (60, 55, 55)                                  # 深灰橡胶底

    period_px = int(9.0 * SCALE)                                   # gt 周期 9mm
    groove_px = max(int(2.8 * SCALE), 3)                           # gt 沟宽 2.8mm
    fn = {"herringbone": draw_herringbone, "waffle": draw_waffle, "wave": draw_wave}[pattern]
    groove = fn(sole, mask, period_px, groove_px)
    sole[groove > 0] = (28, 24, 24)                                # 沟槽更深色
    # 花纹块上加细噪声纹理（模拟橡胶表面）
    noise = rng.normal(0, 6, sole.shape).astype(np.int16)
    sole = np.clip(sole.astype(np.int16) + noise * (mask[..., None] > 0), 0, 255).astype(np.uint8)

    # 模拟拍摄: 轻微透视 + 旋转（边界填充背景色，模拟真实桌面上拍摄）
    BG = 205
    M = cv2.getRotationMatrix2D((w_px / 2, h_px / 2), 7, 1.0)
    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(sole, M3, (w_px + 240, h_px + 240),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(BG, BG, BG))
    src = np.float32([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]])
    dst = np.float32([[0 + PERSPECTIVE * 60, 30], [w_px - PERSPECTIVE * 20, 0],
                      [w_px + 30, h_px - PERSPECTIVE * 60], [40, h_px + 10]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(warped, Hm, (w_px + 240, h_px + 240),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(BG, BG, BG))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"synthetic_{pattern}.jpg")
    cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(out)
    gt = {"pattern": pattern, "length_mm": L_MM, "width_mm": W_MM,
          "period_mm": 9.0, "groove_width_mm": 2.8, "scale_px_per_mm": SCALE}
    with open(os.path.join(out_dir, f"synthetic_{pattern}_gt.json"), "w") as f:
        json.dump(gt, f, indent=2)
    print(f"已生成 {out}  (gt: {gt})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--pattern", default="all", choices=["herringbone", "waffle", "wave", "all"])
    args = ap.parse_args()
    pats = ["herringbone", "waffle", "wave"] if args.pattern == "all" else [args.pattern]
    for p in pats:
        render(p, args.out)
