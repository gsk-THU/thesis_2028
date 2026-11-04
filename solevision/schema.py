"""SoleIR v1.0 结构化数据定义（鞋底中间表示）。

对应《设计方案》§3.3。字段语义见 docs；所有测量值单位 mm / mm²。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

# 鞋类 → 典型沟槽深度先验 (min_mm, max_mm)，Stage 1 深度标定用
TREAD_DEPTH_PRIOR_MM = {
    "running":    (3.0, 5.0),
    "basketball": (2.0, 4.0),
    "hiking":     (4.0, 7.0),
    "canvas":     (2.0, 4.0),
    "dress":      (1.0, 2.5),
    "sandal":     (1.0, 3.0),
    "slipper":    (2.0, 5.0),
    "unknown":    (2.0, 5.0),
}

# 鞋类 → 材质先验分布（compound class → prior weight）
MATERIAL_PRIOR_BY_TYPE = {
    "running":    {"carbon_rubber": 0.55, "eva_foam": 0.25, "tpu": 0.10, "gum_rubber": 0.05, "other": 0.05},
    "basketball": {"gum_rubber": 0.40, "carbon_rubber": 0.45, "tpu": 0.10, "eva_foam": 0.05},
    "hiking":     {"carbon_rubber": 0.60, "tpu": 0.15, "gum_rubber": 0.15, "other": 0.10},
    "canvas":     {"gum_rubber": 0.60, "carbon_rubber": 0.30, "other": 0.10},
    "dress":      {"leather": 0.50, "tpu": 0.25, "carbon_rubber": 0.20, "other": 0.05},
    "sandal":     {"gum_rubber": 0.30, "eva_foam": 0.40, "carbon_rubber": 0.20, "other": 0.10},
    "slipper":    {"eva_foam": 0.55, "gum_rubber": 0.20, "other": 0.25},
    "unknown":    {"carbon_rubber": 0.35, "gum_rubber": 0.25, "eva_foam": 0.20, "tpu": 0.10, "other": 0.10},
}

# 材质类别 → Shore A 先验 (p10, p50, p90)
SHORE_A_PRIOR = {
    "gum_rubber":    (42, 50, 58),
    "carbon_rubber": (58, 65, 72),
    "tpu":           (68, 75, 85),
    "eva_foam":      (35, 45, 55),
    "leather":       (75, 82, 90),
    "other":         (50, 60, 70),
}


@dataclass
class Provenance:
    source_image: str = ""
    capture_quality: str = "unknown"          # pass / warn / fail
    quality_notes: List[str] = field(default_factory=list)
    scale_method: str = "unknown"             # shoe_size_eu / reference_object / uncalibrated
    pixel_mm: float = float("nan")            # 矫正后图像标定 (mm/px)
    providers: Dict[str, str] = field(default_factory=dict)  # 各步骤使用的实现


@dataclass
class TreadElement:
    id: int
    type: str                                 # lug / siping / smooth_zone
    centroid_xy_mm: List[float]
    area_mm2: float
    height_mm: float
    angle_deg: float
    roundness: float


@dataclass
class Pattern:
    pattern_class: str = "unknown"            # herringbone / waffle / wave / hexagonal / smooth / unknown
    period_mm: Optional[float] = None
    direction_deg: Optional[float] = None
    groove_width_mm: Optional[float] = None
    groove_depth_mm: Optional[float] = None
    contact_area_ratio: float = float("nan")
    smooth_zone_ratio: float = 0.0
    periodicity_score: float = 0.0            # 0~1, FFT 峰显著性
    element_count: int = 0


@dataclass
class Geometry:
    length_mm: float = float("nan")
    width_mm: float = float("nan")
    grid_wh: List[int] = field(default_factory=lambda: [0, 0])
    res_mm: float = float("nan")
    heightmap_file: str = ""                  # npz: h (mm), mask (bool)
    depth_uncertainty_mm: Dict[str, float] = field(default_factory=dict)
    elements: List[TreadElement] = field(default_factory=list)
    graph_edges: List[List[float]] = field(default_factory=list)  # [i, j, dist_mm]


@dataclass
class Material:
    compound_class: Dict[str, float] = field(default_factory=dict)   # class→prob
    shoe_type: str = "unknown"
    shore_a: Dict[str, float] = field(default_factory=dict)          # p10/p50/p90
    provider: str = "heuristic"               # heuristic / vlm / user_hint
    notes: str = ""


@dataclass
class Wear:
    state: str = "unknown"                    # new / light / worn / bald
    gloss_score: float = float("nan")
    texture_index: float = float("nan")       # 局部纹理能量，越低越磨平
    flatness_index: float = float("nan")
    bald_ratio: float = 0.0


@dataclass
class SoleIR:
    schema_version: str = SCHEMA_VERSION
    provenance: Provenance = field(default_factory=Provenance)
    geometry: Geometry = field(default_factory=Geometry)
    pattern: Pattern = field(default_factory=Pattern)
    material: Material = field(default_factory=Material)
    wear: Wear = field(default_factory=Wear)
    confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        _drop_nan(d)
        return d

    def summary(self) -> str:
        p = self.pattern
        m = self.material
        top = max(m.compound_class.items(), key=lambda kv: kv[1])[0] if m.compound_class else "?"
        return (
            f"SoleIR v{self.schema_version}\n"
            f"  尺度: {self.geometry.length_mm:.0f}×{self.geometry.width_mm:.0f} mm  "
            f"({self.provenance.pixel_mm:.3f} mm/px, {self.provenance.scale_method})\n"
            f"  花纹: {p.pattern_class}  周期={p.period_mm}mm  沟宽={p.groove_width_mm}mm  "
            f"沟深={p.groove_depth_mm}mm  接触面积比={p.contact_area_ratio:.2f}\n"
            f"  材质: {top}({max(m.compound_class.values(), default=0):.0%})  "
            f"ShoreA p50={m.shore_a.get('p50', '?')}  鞋类={m.shoe_type}\n"
            f"  磨损: {self.wear.state}  gloss={self.wear.gloss_score}  flat={self.wear.flatness_index}\n"
            f"  置信度: {self.confidence}"
        )


def _drop_nan(d: Any) -> None:
    if isinstance(d, dict):
        for k in list(d.keys()):
            v = d[k]
            if isinstance(v, float) and v != v:  # NaN
                del d[k]
            else:
                _drop_nan(v)
    elif isinstance(d, list):
        for i in range(len(d) - 1, -1, -1):
            if isinstance(d[i], float) and d[i] != d[i]:
                del d[i]
            else:
                _drop_nan(d[i])


def validate(ir: SoleIR) -> List[str]:
    """轻量 schema 校验，返回问题列表。"""
    issues = []
    g = ir.geometry
    if g.length_mm <= 0:
        issues.append("geometry.length_mm 未标定（缺尺度参考）")
    if g.res_mm <= 0:
        issues.append("geometry.res_mm 缺失")
    if not (0.0 <= ir.pattern.contact_area_ratio <= 1.0) and ir.pattern.contact_area_ratio == ir.pattern.contact_area_ratio:
        issues.append("pattern.contact_area_ratio 超出 [0,1]")
    if sum(ir.material.compound_class.values()) < 0.5:
        issues.append("material.compound_class 概率未归一")
    return issues
