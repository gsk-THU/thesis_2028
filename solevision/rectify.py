"""Step 1b: 透视矫正 + 尺度标定。

将任意视角的外底照片规整到正射规范坐标系（跟部在下、趾部在上），
并由鞋码或参考物得到 mm/px 标定。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import ellipse_profile, largest_component, rotate_mask_and_image


@dataclass
class Rectified:
    image: np.ndarray            # BGR, 正射矫正并裁剪（含边距）
    outline_mask: np.ndarray     # uint8 0/1, 同尺寸
    pixel_mm: float              # 标定
    length_mm: float
    width_mm: float
    scale_method: str
    rotated_deg: float


# EU 鞋码 → 外底长度近似映射（对常见运动鞋拟合，误差约 ±4mm）
def estimate_length_mm(shoe_size_eu: float) -> float:
    """外底长度 ≈ 足长 × 1.05；足长(mm) ≈ (EU-2)×6.67（巴黎点）。"""
    foot_mm = (shoe_size_eu - 2.0) * 6.667
    return round(foot_mm * 1.05, 1)


def upright_orientation(rot_img: np.ndarray, rot_mask: np.ndarray) -> float:
    """判断矫正后图像是否需要再转 180°：外底前掌最宽，应位于图像上半部。

    返回还需旋转的角度（0 或 180）。
    """
    ys, xs = np.nonzero(rot_mask)
    if len(ys) < 50:
        return 0.0
    h = ys.max() - ys.min()
    top = ys < ys.min() + h / 2
    widths = []
    for half_mask in (top, ~top):
        sel_xs = xs[half_mask]
        widths.append(sel_xs.max() - sel_xs.min() if len(sel_xs) else 0)
    # top 宽度应 >= bottom（前掌宽于后跟）；若相反则翻转
    return 0.0 if widths[0] >= widths[1] else 180.0


def rectify(bgr: np.ndarray, outline_mask: np.ndarray,
            shoe_size_eu: float = None,
            sole_length_mm: float = None,
            margin_ratio: float = 0.06) -> Rectified:
    """按轮廓 mask 矫正姿态并裁剪。

    scale: 优先用显式 sole_length_mm，其次 shoe_size_eu 推算；都没有则 uncalibrated。
    """
    prof = ellipse_profile(outline_mask)
    (cx, cy), (d1, d2), angle = prof["center"], prof["axes"], prof["angle"]
    # cv2.fitEllipse: angle 描述第一根轴 (d1)；长轴若是 d2，其方向角为 angle+90。
    # 目标：把长轴转到竖直方向。
    long_angle = angle if d1 >= d2 else angle + 90.0
    rot_deg = long_angle - 90.0
    rot_img, rot_mask = rotate_mask_and_image(bgr, outline_mask, rot_deg, (cx, cy))

    # 方向归一（跟下趾上）
    extra = upright_orientation(rot_img, rot_mask)
    if extra:
        rot_img = cv2.rotate(rot_img, cv2.ROTATE_180)
        rot_mask = (cv2.rotate(rot_mask * 255, cv2.ROTATE_180) > 127).astype(np.uint8)
    rot_deg_total = rot_deg + extra

    # 紧致裁剪 + 边距
    ys, xs = np.nonzero(rot_mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my = int((y1 - y0) * margin_ratio)
    mx = int((x1 - x0) * margin_ratio)
    H, W = rot_img.shape[:2]
    y0, y1 = max(0, y0 - my), min(H - 1, y1 + my)
    x0, x1 = max(0, x0 - mx), min(W - 1, x1 + mx)
    img_c = rot_img[y0:y1 + 1, x0:x1 + 1].copy()
    mask_c = rot_mask[y0:y1 + 1, x0:x1 + 1].copy()

    # 清理 mask 小噪点
    mask_c = largest_component(mask_c)

    length_px = float(ys.max() - ys.min())
    width_px = float(xs.max() - xs.min())

    if sole_length_mm:
        length_mm = float(sole_length_mm)
        scale_method = "explicit_length"
    elif shoe_size_eu:
        length_mm = estimate_length_mm(shoe_size_eu)
        scale_method = "shoe_size_eu"
    else:
        length_mm = float("nan")
        scale_method = "uncalibrated"

    pixel_mm = length_mm / length_px if length_mm == length_mm else float("nan")
    width_mm = width_px * pixel_mm if pixel_mm == pixel_mm else float("nan")
    return Rectified(img_c, mask_c, pixel_mm, length_mm, width_mm, scale_method, rot_deg_total)
