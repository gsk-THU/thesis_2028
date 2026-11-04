"""命令行入口: python -m solevision.cli <image> [options]"""
from __future__ import annotations

import argparse
import sys

from .pipeline import PipelineConfig, run
from .schema import validate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="solevision",
                                 description="鞋底照片 → SoleIR 结构化数据（Stage 1）")
    ap.add_argument("image", help="鞋底照片路径")
    ap.add_argument("--shoe-size-eu", type=float, default=None, help="EU 鞋码（尺度标定）")
    ap.add_argument("--sole-length-mm", type=float, default=None, help="显式外底长度 mm（优先于鞋码）")
    ap.add_argument("--shoe-type", default="unknown",
                    choices=["running", "basketball", "hiking", "canvas", "dress",
                             "sandal", "slipper", "unknown"])
    ap.add_argument("--material-hint", default="", help="材质提示 gum_rubber/carbon_rubber/tpu/eva_foam/leather")
    ap.add_argument("--depth-provider", default="auto", choices=["auto", "classical", "depth_anything"])
    ap.add_argument("--segmenter", default="auto", choices=["auto", "classical", "sam"])
    ap.add_argument("--material-provider", default="auto", choices=["auto", "heuristic", "vlm"])
    ap.add_argument("--groove-dark", default="auto", choices=["auto", "true", "false"],
                    help="沟槽极性：auto 自动；凝胶印痕类图像（暗印迹/白空隙）用 false")
    ap.add_argument("--out", default="outputs", help="输出目录")
    ap.add_argument("--no-viz", action="store_true", help="不生成 debug 可视化")
    args = ap.parse_args(argv)

    cfg = PipelineConfig(
        shoe_size_eu=args.shoe_size_eu, sole_length_mm=args.sole_length_mm,
        shoe_type=args.shoe_type, material_hint=args.material_hint,
        depth_provider=args.depth_provider, segmenter=args.segmenter,
        material_provider=args.material_provider, groove_dark=args.groove_dark,
        out_dir=args.out, save_debug=not args.no_viz)

    ir = run(args.image, cfg)
    print(ir.summary())
    issues = validate(ir)
    if issues:
        print("\n[警告] schema 校验:")
        for i in issues:
            print("  -", i)
    print(f"\n输出目录: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
