"""SoleVision: 鞋底照片 → SoleIR（Stage 1 视觉管线）。"""
from .schema import SoleIR, SCHEMA_VERSION
from .pipeline import PipelineConfig, run

__all__ = ["SoleIR", "SCHEMA_VERSION", "PipelineConfig", "run"]
