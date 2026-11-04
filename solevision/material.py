"""Step 5a: 材质/鞋类推断。

provider 优先级: user_hint > vlm(若配置) > heuristic(颜色证据 × 类别先验)。
所有 provider 输出统一的概率分布，供 Stage 2 蒙特卡洛使用。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .schema import MATERIAL_PRIOR_BY_TYPE, SHORE_A_PRIOR

COMPOUNDS = list(SHORE_A_PRIOR.keys())


def _color_evidence(bgr: np.ndarray, blocks: np.ndarray) -> Dict[str, float]:
    """从花纹块区域颜色统计给出各材质的似然（手调、可解释）。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32) / 255, hsv[..., 2].astype(np.float32) / 255
    sel = blocks > 0
    if sel.sum() < 100:
        sel = np.ones_like(blocks, bool)
    hm, sm, vm = h[sel].mean(), s[sel].mean(), v[sel].mean()

    like = {}
    if sm < 0.18 and vm > 0.72:
        like = {"eva_foam": 0.45, "gum_rubber": 0.25, "carbon_rubber": 0.10, "tpu": 0.10, "leather": 0.02, "other": 0.08}
    elif 12 <= hm <= 40 and sm >= 0.20:
        like = {"gum_rubber": 0.50, "leather": 0.18, "eva_foam": 0.08, "carbon_rubber": 0.08, "tpu": 0.06, "other": 0.10}
    elif vm < 0.28:
        like = {"carbon_rubber": 0.55, "tpu": 0.12, "gum_rubber": 0.10, "eva_foam": 0.05, "leather": 0.08, "other": 0.10}
    else:
        like = {"carbon_rubber": 0.30, "gum_rubber": 0.22, "eva_foam": 0.15, "tpu": 0.13, "leather": 0.08, "other": 0.12}
    return like


def _bayes(prior: Dict[str, float], like: Dict[str, float]) -> Dict[str, float]:
    post = {}
    for k in prior:
        post[k] = prior[k] * like.get(k, 1e-3)
    z = sum(post.values()) + 1e-9
    return {k: round(v / z, 3) for k, v in post.items()}


class HeuristicMaterial:
    name = "heuristic"

    def classify(self, bgr: np.ndarray, blocks: np.ndarray,
                 shoe_type: str = "unknown") -> Tuple[Dict[str, float], str]:
        prior = MATERIAL_PRIOR_BY_TYPE.get(shoe_type, MATERIAL_PRIOR_BY_TYPE["unknown"])
        like = _color_evidence(bgr, blocks)
        post = _bayes(prior, like)
        return post, shoe_type


class VLMOpenAICompatible:
    """可插拔 VLM 分类器（OpenAI 兼容 /v1/chat/completions）。

    环境变量: VLM_API_KEY / VLM_BASE_URL / VLM_MODEL
    """

    name = "vlm"

    PROMPT = (
        "You are given a photo of a shoe outsole (bottom of a shoe). "
        "Return STRICT JSON only: {\"shoe_type\": one of [running, basketball, hiking, "
        "canvas, dress, sandal, slipper, unknown], \"compound\": one of [gum_rubber, "
        "carbon_rubber, tpu, eva_foam, leather, other], \"confidence\": 0..1} "
        "based on the sole's visible texture, color and pattern."
    )

    def __init__(self):
        self.key = os.environ.get("VLM_API_KEY")
        self.base = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.environ.get("VLM_MODEL", "gpt-4o-mini")
        if not self.key:
            raise RuntimeError("VLM_API_KEY 未配置")

    def classify(self, bgr: np.ndarray, blocks: np.ndarray,
                 shoe_type: str = "unknown") -> Tuple[Dict[str, float], str]:
        import base64
        import json as _json
        import urllib.request

        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode()
        body = _json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "max_tokens": 120,
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = _json.loads(r.read().decode())
        text = payload["choices"][0]["message"]["content"]
        data = _json.loads(text[text.index("{"): text.rindex("}") + 1])
        st = data.get("shoe_type", "unknown")
        comp = data.get("compound", "other")
        conf = float(data.get("confidence", 0.5))
        dist = {c: (1 - conf) / len(COMPOUNDS) for c in COMPOUNDS}
        dist[comp] = dist.get(comp, 0) + conf
        s = sum(dist.values())
        return {k: round(v / s, 3) for k, v in dist.items()}, st


def resolve_provider(prefer: str = "auto") -> object:
    if prefer == "vlm":
        return VLMOpenAICompatible()
    if prefer == "heuristic":
        return HeuristicMaterial()
    for cls in (VLMOpenAICompatible, HeuristicMaterial):
        try:
            return cls()
        except Exception:
            continue
    return HeuristicMaterial()


def shore_a_from(compound_dist: Dict[str, float]) -> Dict[str, float]:
    """混合材质 → Shore A 分布分位数。"""
    samples = []
    rng = np.random.default_rng(0)
    draws = rng.random(4000)
    picks = np.searchsorted(np.cumsum([compound_dist.get(c, 0.0) for c in COMPOUNDS]), draws)
    for p in picks:
        c = COMPOUNDS[min(p, len(COMPOUNDS) - 1)]
        lo, mid, hi = SHORE_A_PRIOR[c]
        samples.append(rng.triangular(lo, mid, hi))
    q = np.percentile(np.array(samples), [10, 50, 90])
    return {"p10": round(float(q[0]), 1), "p50": round(float(q[1]), 1), "p90": round(float(q[2]), 1)}
