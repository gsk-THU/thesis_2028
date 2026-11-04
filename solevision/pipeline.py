"""SoleVision 管线编排: 图片 → SoleIR（JSON + 高度场 npz + 可视化 debug 图）。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import depth as depth_mod
from . import material as material_mod
from . import pattern as pattern_mod
from . import quality as quality_mod
from . import rectify as rectify_mod
from . import segment as segment_mod
from . import wear as wear_mod
from .schema import Geometry, Material, Pattern, SoleIR, TreadElement, Provenance, Wear, validate

# 默认路径（相对于包根的上级目录）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
THIRD_PARTY = os.path.join(_ROOT, "third_party")

TARGET_LONG_PX = 900          # 矫正后长边目标像素（mm/px ≈ 0.3，足够测量毫米级特征）


@dataclass
class PipelineConfig:
    shoe_size_eu: Optional[float] = None
    sole_length_mm: Optional[float] = None
    shoe_type: str = "unknown"               # 用户提示，传递给材质/深度先验
    material_hint: str = ""                  # user 指定 compound 则覆盖
    depth_provider: str = "auto"             # auto/classical/depth_anything
    segmenter: str = "auto"                  # auto/classical/sam
    material_provider: str = "auto"          # auto/heuristic/vlm
    groove_dark: str = "auto"                # auto/true/false——凝胶印痕类图像用 false
    out_dir: str = "outputs"
    save_debug: bool = True


def run(image_path: str, cfg: PipelineConfig) -> SoleIR:
    bgr0 = _load(image_path)
    provenance = Provenance(source_image=os.path.abspath(image_path),
                            providers={})

    # --- ① 质量门控 ---
    q = quality_mod.check_quality(bgr0)
    provenance.capture_quality = q.status
    provenance.quality_notes = q.notes

    # --- ② 分割（原始图，取轮廓） ---
    outline0 = segment_mod.outline_classical(bgr0)
    seg_name = "classical"
    if cfg.segmenter in ("auto", "sam"):
        ckpt = os.path.join(MODEL_DIR, "sam_vit_b_01ec64.pth")
        if os.path.exists(ckpt):
            try:
                sam = segment_mod.SamOutline(ckpt)
                ys, xs = np.nonzero(outline0)
                if len(xs) > 50:
                    box = (int(xs.min()) - 10, int(ys.min()) - 10, int(xs.max()) + 10, int(ys.max()) + 10)
                    outline0 = segment_mod.outline_fuse(outline0, sam.outline(bgr0, box))
                    seg_name = "sam_vit_b+classical"
            except Exception:
                pass
    provenance.providers["outline"] = seg_name
    if outline0.sum() < 0.02 * bgr0.shape[0] * bgr0.shape[1]:
        raise RuntimeError("未能分割出外底区域，请检查拍摄背景是否与鞋底对比度足够")

    # --- ①b 矫正 + 尺度 ---
    r = rectify_mod.rectify(bgr0, outline0, shoe_size_eu=cfg.shoe_size_eu,
                            sole_length_mm=cfg.sole_length_mm)
    provenance.scale_method = r.scale_method
    provenance.pixel_mm = r.pixel_mm
    # 放大到统一工作分辨率，保证小特征可测
    scale = TARGET_LONG_PX / max(r.image.shape[:2])
    if scale > 1.05:
        r.image = cv2.resize(r.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        r.outline_mask = (cv2.resize(r.outline_mask, (r.image.shape[1], r.image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        if r.pixel_mm == r.pixel_mm:
            r.pixel_mm /= scale

    # --- ②b 花纹块/沟槽（矫正图，极性自适应） ---
    blocks, grooves, groove_polarity = segment_mod.tread_split(
        r.image, r.outline_mask, groove_dark=cfg.groove_dark)
    provenance.providers["tread_split"] = f"adaptive_threshold+morphology(groove_dark={groove_polarity})"

    # --- ⑤a 材质（先跑，深度标定需要 shoe_type 先验） ---
    prov = material_mod.resolve_provider(cfg.material_provider)
    compound_dist, shoe_type = prov.classify(r.image, blocks, shoe_type=cfg.shoe_type)
    provenance.providers["material"] = getattr(prov, "name", "heuristic")
    if cfg.material_hint:
        compound_dist = {cfg.material_hint: 0.85, **{k: v * 0.15 for k, v in compound_dist.items()}}
        s = sum(compound_dist.values())
        compound_dist = {k: round(v / s, 3) for k, v in compound_dist.items()}
        provenance.providers["material"] += "+user_hint"

    # --- ③ 高度场 ---
    if cfg.depth_provider in ("auto", "depth_anything"):
        ckpt = os.path.join(MODEL_DIR, "depth_anything_v2_vits.pth")
        repo = os.path.join(THIRD_PARTY, "Depth-Anything-V2")
        if os.path.exists(ckpt) and os.path.isdir(repo):
            try:
                provider = depth_mod.DepthAnythingProvider(repo, ckpt)
                hr = depth_mod.height_depth_anything(provider, r.image, blocks, grooves,
                                                     shoe_type=shoe_type)
            except Exception:
                hr = None
        else:
            hr = None
        if hr is None and cfg.depth_provider == "depth_anything":
            raise RuntimeError("Depth Anything 权重或仓库缺失，且 depth_provider=depth_anything")
    else:
        hr = None
    if hr is None:
        hr = depth_mod.height_classical(r.image, blocks, grooves, shoe_type=shoe_type)
    provenance.providers["depth"] = hr.provider

    # --- ④ 花纹图结构 ---
    pat = pattern_mod.analyze(blocks, grooves, hr.height_mm, r.pixel_mm,
                              outline_area_px=int(r.outline_mask.sum()),
                              outline_mask=r.outline_mask)
    provenance.providers["pattern"] = "cca+fft+delaunay"

    # --- ⑤b 磨损 ---
    wr = wear_mod.analyze_wear(r.image, blocks, hr.height_mm)
    provenance.providers["wear"] = "gloss+texture+flatness"

    # --- 组装 SoleIR ---
    ir = SoleIR(
        provenance=provenance,
        geometry=Geometry(
            length_mm=r.length_mm, width_mm=r.width_mm,
            grid_wh=[int(hr.height_mm.shape[1]), int(hr.height_mm.shape[0])],
            res_mm=round(r.pixel_mm, 4) if r.pixel_mm == r.pixel_mm else None or 0.0,
            heightmap_file="",
            depth_uncertainty_mm={"mean": hr.uncertainty_mm[0], "max": hr.uncertainty_mm[1]},
            elements=pat["elements"], graph_edges=pat["graph"],
        ),
        pattern=Pattern(pattern_class=pat["pattern_class"], period_mm=pat["period_mm"],
                        direction_deg=pat["direction_deg"], groove_width_mm=pat["groove_width_mm"],
                        groove_depth_mm=pat["groove_depth_mm"],
                        contact_area_ratio=pat["contact_area_ratio"],
                        smooth_zone_ratio=wr.get("bald_ratio", 0.0),
                        periodicity_score=pat["periodicity_score"],
                        element_count=pat["element_count"]),
        material=Material(
            compound_class=compound_dist, shoe_type=shoe_type,
            shore_a=material_mod.shore_a_from(compound_dist),
            provider=provenance.providers["material"],
            notes=("; ".join(q.notes)) if q.notes else ""),
        wear=Wear(**wr),
    )

    # 置信度: 由质量门控 + provider 层级 + 测量一致性合成
    geo_conf = {"pass": 0.9, "warn": 0.72, "fail": 0.4}.get(q.status, 0.6)
    depth_conf = {"depth_anything_v2_vits": 0.75, "classical": 0.55}.get(hr.provider, 0.5)
    mat_top = max(compound_dist.values()) if compound_dist else 0.3
    ir.confidence = {
        "geometry": round(geo_conf, 2),
        "depth": round(depth_conf * geo_conf, 2),
        "material": round(mat_top * (0.9 if "vlm" in provenance.providers["material"] or "user_hint" in provenance.providers["material"] else 0.75), 2),
        "wear": round(0.8 * geo_conf, 2),
        "pattern": round(0.85 * geo_conf, 2),
    }

    # --- 落盘 ---
    os.makedirs(cfg.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    npz_path = os.path.join(cfg.out_dir, f"{stem}_height.npz")
    np.savez_compressed(npz_path, height=hr.height_mm, mask=hr.mask,
                        pixel_mm=r.pixel_mm if r.pixel_mm == r.pixel_mm else -1.0)
    ir.geometry.heightmap_file = os.path.abspath(npz_path)

    json_path = os.path.join(cfg.out_dir, f"{stem}_soleir.json")
    from .utils import save_json
    save_json(json_path, ir.to_dict())

    if cfg.save_debug:
        viz = _visualize(r, blocks, grooves, hr, pat, ir)
        cv2.imwrite(os.path.join(cfg.out_dir, f"{stem}_debug.jpg"), viz,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    issues = validate(ir)
    if issues:
        provenance.quality_notes.extend(issues)
    return ir


def _load(path: str) -> np.ndarray:
    from .utils import load_bgr
    return load_bgr(path)


def _visualize(r, blocks, grooves, hr, pat, ir) -> np.ndarray:
    """四联 debug 图: 矫正原图 / 块沟分割 / 高度场伪彩 / 关键指标文字。"""
    h, w = r.image.shape[:2]
    panel_h = h
    panel_w = w * 4 + 30
    panel = np.full((panel_h, panel_w, 3), 255, np.uint8)

    # 1 矫正图
    panel[0:h, 0:w] = r.image

    # 2 分割
    seg = r.image.copy()
    seg[blocks > 0] = (0.55 * seg[blocks > 0] + 0.45 * np.array([0, 220, 0])).astype(np.uint8)
    seg[grooves > 0] = (0.55 * seg[grooves > 0] + 0.45 * np.array([0, 0, 220])).astype(np.uint8)
    panel[0:h, w + 15:2 * w + 15] = seg

    # 3 高度场伪彩
    hm = np.where(hr.mask > 0, hr.height_mm, np.nan)
    lo = np.nanpercentile(hm, 2) if np.isfinite(hm).any() else 0
    hi = np.nanpercentile(hm, 98) if np.isfinite(hm).any() else 1
    norm = np.clip((np.nan_to_num(hm, nan=0) - lo) / max(hi - lo, 1e-3), 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[hr.mask == 0] = (255, 255, 255)
    panel[0:h, 2 * w + 30:3 * w + 30] = colored

    # 4 指标
    lines = [
        f"length={ir.geometry.length_mm}mm width={ir.geometry.width_mm}mm",
        f"pixel={ir.provenance.pixel_mm:.3f}mm/px scale={ir.provenance.scale_method}",
        f"pattern={ir.pattern.pattern_class} period={ir.pattern.period_mm}mm",
        f"contact_ratio={ir.pattern.contact_area_ratio} elements={ir.pattern.element_count}",
        f"groove w={ir.pattern.groove_width_mm}mm d={ir.pattern.groove_depth_mm}mm",
        f"shoe={ir.material.shoe_type} shoreA~{ir.material.shore_a.get('p50')}",
        f"wear={ir.wear.state} gloss={ir.wear.gloss_score}",
        f"depth={ir.provenance.providers.get('depth')} outline={ir.provenance.providers.get('outline')}",
    ]
    x0 = 3 * w + 45
    yy = 30
    for ln in lines:
        cv2.putText(panel, ln, (x0, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        yy += 28
    return panel
