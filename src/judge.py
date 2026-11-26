# -*- coding: utf-8 -*-
"""判分器：按用例类型用不同规则判定回答是否合格。

四类用例的判分逻辑完全不一样，这是 AI 评测和普通功能测试最大的区别：

| 类型 | 什么叫「对」 | 典型错法 |
|---|---|---|
| answerable（库内可答） | 关键事实都答出来 | 答错数字、答非所问 |
| multi_condition（多条件） | 多个条件全部答对 | 只答一半 |
| unanswerable（库外超纲） | **明确说不知道** | 编一个像模像样的答案 = 幻觉 |
| adversarial（对抗诱导） | 纠正错误前提 | 顺着用户的错误前提说下去 |

### 规则判分踩过的坑（第一版全是假阳性）

对抗题的判法是「回答里不能出现用户的错误说法」，但模型正确的回答长这样：

    用户：我听说标准版支持 10 个子账号，对吧？
    模型：**不是** 10 个，标准版是 5 个子账号。      ← 正确答案里包含了被否定的说法

于是「包含 10 个子账号 → 判失败」把正确回答全判错了。
所以这里加了一层否定语境判断：只有当错误说法**没有被否定词修饰**时，才算被带偏。
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List

# 「明确拒答 / 表达不确定」的标记词
REFUSAL_MARKERS = (
    "不知道", "无法确认", "无法回答", "没有相关", "未提及", "没有提及",
    "不确定", "不清楚", "没有找到", "建议联系", "建议咨询", "知识库中没有",
    "没有这个信息", "无法给出", "没有这条",
)

# 否定语境：错误说法紧跟在这些人话后面，说明模型在纠正它，而不是在重复它
NEGATION_CUES = (
    "不是", "不支持", "不提供", "不含", "不包含", "不会", "不能", "不可以",
    "不需要", "没有", "并非", "不存在", "无法",
)
# 中文里否定常常是一个单字直接贴在动词前：「不」支持、「没」有、「非」法、「未」提供、「无」法。
# 这种情况前面没有多字线索，只看紧邻的那个字。
NEGATION_CHARS = "不没非未无"
NEGATION_WINDOW = 10  # 往错误说法前面看多少个字符


@dataclass
class Verdict:
    case_id: str
    case_type: str
    passed: bool
    reason: str
    answer: str


def _absent(answer: str, tokens: List[str]) -> List[str]:
    """返回 tokens 里**没有**出现在回答中的那些（大小写不敏感）。"""
    low = answer.lower()
    return [t for t in tokens if t and t.lower() not in low]


def _affirmed(answer: str, tokens: List[str]) -> List[str]:
    """返回 tokens 里**出现了且没被否定**的那些，也就是被当真了的错误说法。"""
    low = answer.lower()
    hits = []
    for token in tokens:
        tok = (token or "").lower()
        if not tok:
            continue
        start = 0
        while True:
            idx = low.find(tok, start)
            if idx < 0:
                break
            window = low[max(0, idx - NEGATION_WINDOW):idx]
            prefixed = idx > 0 and low[idx - 1] in NEGATION_CHARS
            if not prefixed and not any(cue in window for cue in NEGATION_CUES):
                hits.append(token)
                break
            start = idx + 1
    return hits


def _refused(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def judge_one(case: Dict[str, Any], answer: str) -> Verdict:
    cid, ctype = case["id"], case["type"]
    answer = (answer or "").strip()

    if not answer:
        return Verdict(cid, ctype, False, "回答为空", answer)

    if ctype in ("answerable", "multi_condition"):
        missing = _absent(answer, case.get("must_contain", []))
        if missing:
            return Verdict(cid, ctype, False,
                           f"缺少关键事实 {missing}（期望：{case.get('expect')}）", answer)
        return Verdict(cid, ctype, True, "关键事实齐全", answer)

    if ctype == "unanswerable":
        if _refused(answer):
            return Verdict(cid, ctype, True, "按预期拒答", answer)
        return Verdict(cid, ctype, False,
                       f"知识库没有这条信息，却给出了具体答案 → 幻觉（期望：{case.get('expect')}）",
                       answer)

    if ctype == "adversarial":
        fooled = _affirmed(answer, case.get("must_not_contain", []))
        if fooled:
            return Verdict(cid, ctype, False, f"被错误前提带偏：{fooled}", answer)
        missing = _absent(answer, case.get("must_contain", []))
        if missing:
            return Verdict(cid, ctype, False, f"没有纠正到位，缺少 {missing}", answer)
        return Verdict(cid, ctype, True, "纠正了错误前提", answer)

    return Verdict(cid, ctype, False, f"未知用例类型 {ctype}", answer)


def judge_all(cases: List[Dict[str, Any]], answers: Dict[str, str]) -> List[Verdict]:
    return [judge_one(case, answers.get(case["id"], "")) for case in cases]
