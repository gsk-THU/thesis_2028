# SoleVision — 鞋底图片 → SoleIR（Stage 1 实现）

将一张鞋底照片转换为结构化的 **SoleIR**（Sole Intermediate Representation）数据：
几何（高度场/花纹图/接触面积比）+ 材质先验分布 + 磨损评估 + 各字段置信度。
输出可直接作为 Stage 2（SqueakAgent 物理推理）的输入。

完整设计见 `设计方案.md`；数据格式定义见 `solevision/schema.py`（SoleIR v1.0）。

## 环境准备

```bash
cd sole-squeak
python3 -m venv .venv
.venv/bin/pip install opencv-python-headless scipy scikit-image   # 基础（必装）

# 可选增强（神经网络 provider）
.venv/bin/pip install torch torchvision segment-anything          # SAM v1 轮廓分割
# Depth Anything V2: 权重经 HF 镜像下载
curl -L -o models/depth_anything_v2_vits.pth \
  https://hf-mirror.com/Depth-Anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
# SAM 权重（Meta CDN）
curl -L -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

权重齐备时管线自动启用神经网络 provider；缺失时自动降级到经典 CV（所有功能仍可运行，
深度置信度会下调）。VLM 材质分类需设置 `VLM_API_KEY`（OpenAI 兼容接口，可用
`VLM_BASE_URL`/`VLM_MODEL` 指定服务商与模型）。

## 使用

```bash
# 基本用法（用鞋码做尺度标定）
.venv/bin/python -m solevision.cli data/synthetic/synthetic_herringbone.jpg \
    --shoe-size-eu 42 --shoe-type running

# 指定 provider
.venv/bin/python -m solevision.cli mysole.jpg --sole-length-mm 288 \
    --segmenter sam --depth-provider depth_anything --material-provider vlm \
    --out outputs

# 凝胶印痕类图像（如 FID-300：暗色印迹、白色空隙，物理极性与照片相反）
.venv/bin/python -m solevision.cli data/fid300/FID-300/references/00001.png \
    --groove-dark false --sole-length-mm 285
```

输出（`--out` 目录）：

| 文件 | 内容 |
|---|---|
| `<stem>_soleir.json` | SoleIR v1.0（几何/材质/磨损/置信度，NaN 字段自动剔除） |
| `<stem>_height.npz` | 高度场 `height`(mm) + `mask` + `pixel_mm` |
| `<stem>_debug.jpg` | 四联可视化：矫正图 / 块沟分割 / 高度场伪彩 / 指标面板 |

## 合成数据自测（带 ground truth）

```bash
.venv/bin/python tools/make_synthetic_sole.py --out data/synthetic
.venv/bin/python -m solevision.cli data/synthetic/synthetic_herringbone.jpg --shoe-size-eu 42
```

生成器内置 gt（外底 285×104mm、花纹周期 9mm、沟宽 2.8mm），管线输出的
`geometry.length_mm` / `pattern.period_mm` / `pattern.groove_width_mm` 应与其接近。

## 免费鞋底数据集（调研结论）

| 数据集 | 内容 | 获取 | 许可 |
|---|---|---|---|
| [FID-300](https://fid.dmi.unibas.ch) | 300 张案发现场鞋印 + 1175 张参考外底印痕（凝胶提取器扫描）；子集 [CSFID-170](https://fid.dmi.unibas.ch)（周期花纹） | 直接下载 zip（103MB） | 仅限研究，CSFID 需引用 [Kortylewski et al., ACCV 2014] |
| [ShoeCase](https://iastate.figshare.com/articles/dataset/ShoeCase_A_data_set_of_mock_crime_scene_footwear_impressions/22814438)（Iowa State） | 936 双鞋 / 3275 个文件的模拟现场鞋印（论文 [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10539887)） | Figshare 开放获取 | 开放 |
| [CVL IMPress](https://cvl.tuwien.ac.at/impress-dataset)（TU Wien） | 300 双鞋、多条件采集的鞋印 | 官网申请 | 研究用 |
| [ShoeRinsics](https://github.com/Samia067/ShoeRinsics) | 基于 FID300 的 41 组鞋底图+鞋印+**mask**（Real-FID-val） | GitHub | 开放 |
| [NIJ/SUNY 外底照片库](https://www.ojp.gov/pdffiles1/nij/grants/233981.pdf)（Srihari 组） | 全新鞋外底照片 + 花纹分类编码 | 报告附录 | 研究用 |
| [CSAFE 外底磨损数据](https://forensicstats.org/news-posts/csafe-researchers-propose-method-to-quantify-the-similarity-between-footwear-outsole-images) | 24 双 Nike 六个月磨损跟踪外底图 | 联系 CSAFE | 研究用 |

注意：上述数据集以**鞋印/印痕**为主，适合花纹表征与 Stage 1 分割验证；
"彩色外底照片 + 材质/磨损标签"暂无公开数据集——这正是本项目自建校准集（设计方案 §5）的原因。

## 模块结构

```
solevision/
├── schema.py    SoleIR v1.0 数据定义 + 先验表 + 校验
├── quality.py   ① 拍摄质量门控（Laplacian 模糊度 / 曝光 / 对比度）
├── rectify.py   ① 透视矫正 + 方向归一 + 尺度标定（鞋码/显式长度）
├── segment.py   ② 轮廓（classical / SAM v1 框提示融合）+ 花纹块/沟槽二分割
├── depth.py     ③ 高度场（classical 亮度浮雕 / Depth Anything V2）+ 类别先验标定
├── pattern.py   ④ 花纹元素 CCA / 周期方向 FFT / Delaunay 图 / 接触面积比
├── material.py  ⑤ 材质分类（heuristic 贝叶斯 / VLM 可插拔）+ Shore A 分布
├── wear.py      ⑤ 磨损（光泽占比 / 局部纹理能量 / 平坦度）
├── pipeline.py  编排 → SoleIR + debug 可视化
└── cli.py       命令行入口
```

## 已知限制（v1）

1. 深度绝对标定依赖鞋类先验（±1mm 级），置信度字段已体现；补拍侧光照片做光度立体是 v2 路线。
2. 材质分类 heuristic 仅用颜色统计，区分度有限——生产环境应启用 VLM provider；凝胶印痕（无色彩语义）不适用。
3. FID-300 是凝胶印痕（灰度、无材质信息），用于花纹几何验证合适，不能验证材质/磨损；其参考印痕为**部分印痕**，长度/宽度可能小于真实外底。
4. 矫正假设外底近似平面（成立），但强透视（>30°）下轮廓椭圆拟合误差增大。
5. 波浪纹（wave）在规则分类器下可能被归为 waffle（正弦调制的二次谐波产生伪方向峰）；
   蜿蜒花纹的"周期"本身谱带较宽，误差容限应放宽（合成验证中放宽至 55%）。

## 自检基线（tools/run_checks.py，合成图带 ground truth）

| 指标 | herringbone | waffle | wave | 验收线 |
|---|---|---|---|---|
| 长度误差 | 1.8% | 1.8% | 1.8% | <5% |
| 周期误差 | 8.7% | 5.2% | 8.0% | <25%（wave 55%） |
| 沟宽误差 | 8.2% | 2.9% | 28.6% | <40% |
| 花纹类型 | herringbone ✓ | waffle ✓ | (waffle，见限制5) | 非 unknown |

重复运行：`.venv/bin/python tools/run_checks.py`
