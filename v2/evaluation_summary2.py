import json
from statistics import mean
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "v2" / "evaluated_results2.jsonl"
if not file_path.exists():
    file_path = project_root / "evaluated_results2.jsonl"

with open(file_path, "r", encoding="utf-8") as file:
    results = [json.loads(line) for line in file if line.strip()]

metrics = [
    "answer_relevancy",
    "faithfulness",
    "context_relevancy",
    "correctness"
]

threshold = 0.7

print("\n================ RAG Evaluation Summary (v2) ================\n")

for metric in metrics:
    scores = [result[metric] for result in results if result.get(metric) is not None]
    if not scores:
        print(f"{metric.replace('_', ' ').title()}: No valid scores recorded")
        continue

    average_score = mean(scores)
    passed = sum(score >= threshold for score in scores)
    failed = len(scores) - passed
    pass_rate = (passed / len(scores)) * 100

    print(metric.replace("_", " ").title())
    print(f"Average Score: {average_score:.2f}")
    print(f"Passed       : {passed}/{len(scores)}")
    print(f"Failed       : {failed}/{len(scores)}")
    print(f"Pass Rate    : {pass_rate:.1f}%")
    print()