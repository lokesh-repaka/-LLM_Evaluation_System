import sys
from pathlib import Path
import json
import time

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from RAG1.agent import ask

dataset_path = project_root / "eval_dataset.jsonl"
results_path = project_root / "v3" / "results3.jsonl"
if not results_path.parent.exists():
    results_path = project_root / "results3.jsonl"

with open(dataset_path, "r", encoding="utf-8") as file:
    test_cases = [json.loads(line) for line in file if line.strip()]

results = []

for i, test_case in enumerate(test_cases):
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]

    print(f"\n[{i+1}/{len(test_cases)}] Question: {question}")

    # Retry loop with backoff if rate limits are hit
    max_retries = 3
    rag_result = None
    for attempt in range(max_retries):
        try:
            rag_result = ask(question)
            break
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                print(f"Rate limited (attempt {attempt+1}/{max_retries}). Waiting 40s for quota reset...")
                time.sleep(40)
            else:
                raise e

    if rag_result is None:
        print(f"Failed to get answer for: {question}")
        continue

    result = {
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": rag_result["answer"],
        "context": rag_result["context"]
    }

    results.append(result)
    print(f"Answer: {rag_result['answer']}")

    # Delay between test cases keeps requests safely within rate limits
    time.sleep(4)

with open(results_path, "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\nDone! Results saved to {results_path}")
