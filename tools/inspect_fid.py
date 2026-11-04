"""检查 FID-300 数据集结构并抽样跑管线。

用法: python tools/inspect_fid.py [--zip data/fid300/FID-300.zip]
"""
from __future__ import annotations

import argparse
import os
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/fid300/FID-300.zip")
    ap.add_argument("--extract", default="data/fid300/")
    ap.add_argument("--list", type=int, default=20, help="列出前 N 个文件")
    args = ap.parse_args()

    if not os.path.exists(args.zip):
        print(f"找不到 {args.zip}，请先下载")
        return

    names = []
    with zipfile.ZipFile(args.zip) as z:
        names = z.namelist()
        print(f"共 {len(names)} 个条目；抽样:")
        for n in names[:args.list]:
            print("  ", n)
        img_ext = {}
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            img_ext[ext] = img_ext.get(ext, 0) + 1
        print("扩展名统计:", img_ext)
        if args.extract:
            z.extractall(args.extract)
            print(f"已解压到 {args.extract}")


if __name__ == "__main__":
    main()
