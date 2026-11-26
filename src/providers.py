# -*- coding: utf-8 -*-
"""回答来源：mock（读预置回答）与真实模型（OpenAI 兼容接口）两种。

为什么要有 mock：
没有 API Key 时也能把「评测集 → 判分 → 报告」整条链路跑通并复现同一份结果；
换成真实模型只需要加 --provider deepseek 和一个 key，代码零改动。
"""
import json
import os
from pathlib import Path


class MockProvider:
    """读取 data/mock_answers.jsonl。用来验证评测链路本身是否可靠。"""

    name = "mock"

    def __init__(self, path: "str | Path" = "data/mock_answers.jsonl"):
        self.path = Path(path)
        self._answers = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                self._answers[item["id"]] = item["answer"]

    def answer(self, case: dict) -> str:
        return self._answers.get(case["id"], "")


class OpenAICompatibleProvider:
    """走 OpenAI 兼容的 /chat/completions（DeepSeek、Moonshot 等都适用）。

    需要环境变量：EVAL_API_KEY（必填）、EVAL_BASE_URL（默认 DeepSeek）、EVAL_MODEL。
    """

    name = "llm"

    def __init__(self, base_url: "str | None" = None, model: "str | None" = None,
                 api_key: "str | None" = None, timeout: int = 60):
        import requests  # 延迟导入：mock 模式下不需要网络依赖

        self.base_url = (base_url or os.getenv("EVAL_BASE_URL")
                         or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.getenv("EVAL_MODEL") or "deepseek-chat"
        self.api_key = api_key or os.getenv("EVAL_API_KEY") or ""
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "缺少 EVAL_API_KEY。可以先用 --provider mock 跑通链路，"
                "或者把 key 写进环境变量再试。"
            )
        self._requests = requests

    def answer(self, case: dict) -> str:
        resp = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system",
                     "content": "你是 Nova CRM 的客服助手。只根据下面的知识库回答；"
                                "知识库里没有的内容，明确说不知道，不要编造。"},
                    {"role": "user", "content": case["q"]},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def get_provider(name: str = "mock", **kwargs):
    if name == "mock":
        return MockProvider(**kwargs)
    if name in ("llm", "deepseek"):
        return OpenAICompatibleProvider(**kwargs)
    raise ValueError(f"未知 provider: {name}（可选 mock / deepseek）")
