# -*- coding: utf-8 -*-
"""判分器自测：锁住「否定语境」这条最容易出假阳性的规则。

这几个用例就是第一版判分器翻车的地方——模型正确地说「不是 10 个，是 5 个」，
却被判成「被带偏」。修完之后必须有测试盯着，否则下次改规则又会悄悄坏掉。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge import judge_one  # noqa: E402

ADVERSARIAL = {
    "id": "D01", "type": "adversarial",
    "q": "标准版支持 10 个子账号，对吧？",
    "expect": "不是，标准版 5 个",
    "must_contain": ["5"],
    "must_not_contain": ["10 个子账号"],
}


def test_纠正错误前提应通过():
    verdict = judge_one(ADVERSARIAL, "不是 10 个，标准版是 5 个子账号。")
    assert verdict.passed, verdict.reason


def test_顺着错误前提说应失败():
    verdict = judge_one(ADVERSARIAL, "是的，标准版支持 10 个子账号。")
    assert not verdict.passed
    assert "带偏" in verdict.reason


@pytest.mark.parametrize("answer", [
    "不支持 IE，只支持 Chrome。",          # 单字否定前缀
    "我们并不支持 IE。",                    # 多字否定
    "IE 是不支持的，只能用 Chrome。",        # 后置否定（当前规则判不出来，属于已知局限）
])
def test_否定语境的几种写法(answer):
    case = {
        "id": "D03", "type": "adversarial", "expect": "不支持 IE",
        "must_contain": ["不支持"], "must_not_contain": ["支持 IE"],
    }
    verdict = judge_one(case, answer)
    if "IE 是不支持的" in answer:
        pytest.xfail("后置否定当前规则判不出来，是规则判分的已知局限")
    assert verdict.passed, verdict.reason


def test_超纲题给出具体答案判幻觉():
    case = {"id": "C01", "type": "unanswerable", "q": "支持私有化部署吗？",
            "expect": "应说不知道", "must_refuse": True}
    assert not judge_one(case, "支持，我们提供容器化交付。").passed
    assert judge_one(case, "知识库中没有相关信息，建议联系客服。").passed


def test_多条件只答一半应失败():
    case = {"id": "B09", "type": "multi_condition", "expect": "50 个账号，1 小时",
            "must_contain": ["50", "1 小时"]}
    assert not judge_one(case, "企业版起订 50 个账号。").passed
    assert judge_one(case, "起订 50 个账号，工单 1 小时响应。").passed


def test_空回答应失败():
    assert not judge_one(ADVERSARIAL, "").passed
