"""SoleVision × ShoeRinsics 交叉验证。

对 real-val 的 36 组（照片/mask/真实墨印）评测:
1. 我们的轮廓分割 vs 他们的 mask → IoU_outline
2. 我们的花纹块（极性自动） vs 真实墨印 → IoU_contact（墨印=真实接触面 GT）
3. 接触面积比对比

用法: .venv/bin/python tools/eval_shoerinsics.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from solevision import segment as seg  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray = None) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    if valid is not None:
        a, b = a & valid, b & valid
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else float("nan")


def main():
    base = os.path.join(ROOT, "data", "shoerinsics", "real_val")
    names = sorted(os.listdir(os.path.join(base, "image")))
    used = {"01", "02", "03", "04"}          # readme: 01-04 为已穿旧鞋
    rows = []
    for name in names:
        stem = os.path.splitext(name)[0]
        img = cv2.imread(os.path.join(base, "image", name))
        mask = (cv2.imread(os.path.join(base, "mask", name), 0) > 127).astype(np.uint8)
        print_p = (cv2.imread(os.path.join(base, "print", name), 0) < 127).astype(np.uint8)  # 墨迹=深色=接触

        # 我们的分割（原始图空间，不做矫正——与他们的标注同坐标系）
        outline = seg.outline_classical(img)
        blocks, grooves, pol = seg.tread_split(img, outline & mask)

        iou_outline = iou(outline, mask)
        iou_contact = iou(blocks, print_p, valid=mask.astype(bool))
        ours_cr = blocks.sum() / max(mask.sum(), 1)
        gt_cr = print_p.sum() / max(mask.sum(), 1)
        rows.append((stem, "used" if stem.split("_")[0] in used else "new",
                     pol, iou_outline, iou_contact, ours_cr, gt_cr))

    print(f"{'样本':<8}{'类别':<6}{'极性':<7}{'轮廓IoU':<10}{'接触IoU':<10}{'接触比(我们)':<14}{'接触比(墨印GT)':<12}")
    for stem, cat, pol, io, ic, ocr, gcr in rows:
        print(f"{stem:<8}{cat:<6}{str(pol):<7}{io:<10.3f}{ic:<10.3f}{ocr:<14.3f}{gcr:<12.3f}")
    arr = np.array([[r[3], r[4], r[5], r[6]] for r in rows], dtype=float)
    used_mask = np.array([r[1] == "used" for r in rows])
    print("-" * 62)
    print(f"均值     轮廓IoU={arr[:,0].mean():.3f}  接触IoU={arr[:,1].mean():.3f}  "
          f"接触比误差={np.abs(arr[:,2]-arr[:,3]).mean():.3f}")
    print(f"旧鞋(n={used_mask.sum()})   接触IoU={arr[used_mask,1].mean():.3f}   "
          f"新鞋(n={(~used_mask).sum()})  接触IoU={arr[~used_mask,1].mean():.3f}")


if __name__ == "__main__":
    main()
