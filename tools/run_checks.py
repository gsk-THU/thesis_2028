"""端到端自检: 对合成图跑管线并与 ground truth 对比。

用法: .venv/bin/python tools/run_checks.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(img, extra=None):
    cmd = [sys.executable, "-m", "solevision.cli", img, "--no-viz", "--out", "outputs",
           "--shoe-size-eu", "42"] + (extra or [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError(f"管线失败: {img}")
    stem = os.path.splitext(os.path.basename(img))[0]
    with open(os.path.join(ROOT, "outputs", f"{stem}_soleir.json")) as f:
        return json.load(f), r.stdout


def main():
    gts = {}
    for p in ("herringbone", "waffle", "wave"):
        with open(os.path.join(ROOT, "data", "synthetic", f"synthetic_{p}_gt.json")) as f:
            gts[p] = json.load(f)

    print("=" * 70)
    ok_all = True
    for p, gt in gts.items():
        img = os.path.join(ROOT, "data", "synthetic", f"synthetic_{p}.jpg")
        ir, stdout = run_cli(img)
        geo = ir["geometry"]
        pat = ir["pattern"]

        def rel_err(pred, gt_val):
            if pred in (None, 0) or gt_val in (None, 0):
                return float("nan")
            return abs(pred - gt_val) / gt_val * 100

        checks = {
            "长度误差%": rel_err(geo.get("length_mm"), gt["length_mm"]),
            "周期误差%": rel_err(pat.get("period_mm"), gt["period_mm"]),
            "沟宽误差%": rel_err(pat.get("groove_width_mm"), gt["groove_width_mm"]),
            "接触面积比": pat.get("contact_area_ratio"),
            "花纹类型": pat.get("pattern_class"),
            "元素数": pat.get("element_count"),
        }
        print(f"\n[{p}]  gt: {gt}")
        for k, v in checks.items():
            print(f"   {k}: {v}")
        # 验收: 长度<5%; 周期<25%（wave 为蜿蜒曲线，谱带天然展宽，放宽到 55%）; 沟宽<40%
        period_tol = 55.0 if p == "wave" else 25.0
        passed = (checks["长度误差%"] < 5 and checks["周期误差%"] < period_tol
                  and checks["沟宽误差%"] < 40 and checks["花纹类型"] not in ("unknown",))
        ok_all &= passed
        print(f"   => {'PASS' if passed else 'FAIL'}")
    print("=" * 70)
    print("总体:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
