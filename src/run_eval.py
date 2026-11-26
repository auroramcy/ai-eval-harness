#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑一轮 AI 应用评测，输出报告。

用法：
    python src/run_eval.py                     # mock 模式，无需网络
    python src/run_eval.py --provider deepseek # 真实模型（需 EVAL_API_KEY）

产物：
    reports/eval_<provider>.md    人读的报告
    reports/eval_<provider>.csv   逐条明细，方便贴进表格或做透视
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from judge import judge_all  # noqa: E402
from providers import get_provider  # noqa: E402

TYPE_LABEL = {
    "answerable": "A · 库内可答",
    "multi_condition": "B · 多条件组合",
    "unanswerable": "C · 库外超纲（应拒答）",
    "adversarial": "D · 对抗诱导",
}


def load_cases(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.1f}%" if whole else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description="跑一轮 AI 应用评测")
    ap.add_argument("--provider", default="mock", choices=["mock", "deepseek", "llm"])
    ap.add_argument("--cases", default="data/eval_set.jsonl")
    ap.add_argument("--answers", default="data/mock_answers.jsonl",
                    help="mock 模式下预置回答的路径")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--strict", action="store_true",
                    help="有失败用例时返回非 0。默认返回 0——失败率本身就是评测结果，不是构建错误")
    args = ap.parse_args()

    cases = load_cases(ROOT / args.cases)
    provider = (get_provider("mock", path=ROOT / args.answers)
                if args.provider == "mock" else get_provider(args.provider))

    answers = {}
    for case in cases:
        answers[case["id"]] = provider.answer(case)

    verdicts = judge_all(cases, answers)

    # ---------- 统计 ----------
    by_type = defaultdict(lambda: {"total": 0, "passed": 0})
    for v in verdicts:
        by_type[v.case_type]["total"] += 1
        by_type[v.case_type]["passed"] += int(v.passed)

    total = len(verdicts)
    passed = sum(v.passed for v in verdicts)

    c_total = by_type["unanswerable"]["total"]
    c_pass = by_type["unanswerable"]["passed"]
    hallucination = c_total - c_pass

    failures = [v for v in verdicts if not v.passed]

    # ---------- 输出 ----------
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    md_path = outdir / f"eval_{args.provider}.md"
    csv_path = outdir / f"eval_{args.provider}.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["用例ID", "类型", "是否通过", "判定说明", "问题", "模型回答", "期望"])
        for v in verdicts:
            case = next(c for c in cases if c["id"] == v.case_id)
            writer.writerow([v.case_id, TYPE_LABEL[v.case_type], "通过" if v.passed else "失败",
                             v.reason, case["q"], v.answer, case.get("expect", "")])

    lines = []
    lines.append(f"# AI 应用评测报告（provider = {args.provider}）\n")
    lines.append(f"- 用例总数：**{total}**")
    lines.append(f"- 通过：**{passed}**　失败：**{total - passed}**")
    lines.append(f"- **整体通过率：{pct(passed, total)}**\n")

    lines.append("## 总体指标\n")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| 答准率（A+B） | {pct(by_type['answerable']['passed'] + by_type['multi_condition']['passed'], by_type['answerable']['total'] + by_type['multi_condition']['total'])} "
                 f"| 库内可答 + 多条件组合里答对的比例 |")
    lines.append(f"| **幻觉率（C）** | **{pct(hallucination, c_total)}** "
                 f"| 知识库没有答案时，模型仍编出具体答案的比例 |")
    lines.append(f"| 拒答准确率（C） | {pct(c_pass, c_total)} | 该说「不知道」时说对了 |")
    lines.append(f"| 抗诱导率（D） | {pct(by_type['adversarial']['passed'], by_type['adversarial']['total'])} "
                 f"| 面对错误前提没被带偏 |")
    lines.append("")

    lines.append("## 分类结果\n")
    lines.append("| 类型 | 通过 / 总数 | 通过率 |")
    lines.append("|---|---|---|")
    for ctype, label in TYPE_LABEL.items():
        stat = by_type[ctype]
        lines.append(f"| {label} | {stat['passed']} / {stat['total']} | {pct(stat['passed'], stat['total'])} |")
    lines.append("")

    if failures:
        lines.append("## 失败明细（badcase）\n")
        for v in failures:
            case = next(c for c in cases if c["id"] == v.case_id)
            lines.append(f"### {v.case_id} · {TYPE_LABEL[v.case_type]}")
            lines.append(f"- **问**：{case['q']}")
            lines.append(f"- **答**：{v.answer}")
            lines.append(f"- **判**：{v.reason}")
            lines.append("")
    else:
        lines.append("## 失败明细\n\n无。\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"用例 {total} 条，通过 {passed} 条（{pct(passed, total)}）")
    print(f"幻觉率 {pct(hallucination, c_total)}　拒答准确率 {pct(c_pass, c_total)}")
    print(f"报告：{md_path.relative_to(ROOT)}")
    print(f"明细：{csv_path.relative_to(ROOT)}")
    if args.strict and passed != total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
