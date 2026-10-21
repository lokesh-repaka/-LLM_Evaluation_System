import sys
from pathlib import Path
import json
import time
import os
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval
)
from deepeval.models import GeminiModel

# Configure Gemini as the evaluation judge model
gemini_model = GeminiModel(
    model="gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize all 4 metrics using Gemini
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=gemini_model)
faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=gemini_model)
contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model=gemini_model)
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output",
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    model=gemini_model
)

results_in_path = project_root / "v1" / "results1.jsonl"
if not results_in_path.exists():
    results_in_path = project_root / "results1.jsonl"

eval_output_path = project_root / "v1" / "eval_results1.jsonl"

with open(results_in_path, "r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

eval_results = []

print(f"Starting evaluation of {len(records)} test cases with Gemini...")

for i, record in enumerate(records):
    print(f"\n[{i+1}/{len(records)}] Evaluating: {record['question']}")

    context_list = [record["context"]] if isinstance(record["context"], str) else record["context"]

    test_case = LLMTestCase(
        input=record["question"],
        actual_output=record["actual_answer"],
        expected_output=record["expected_answer"],
        retrieval_context=context_list
    )

    # Function to safely measure a metric with rate limit and server error backoff
    def run_metric_safe(metric, name):
        for attempt in range(5):
            try:
                metric.measure(test_case)
                return {
                    "score": metric.score,
                    "reason": getattr(metric, "reason", None)
                }
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    print(f"  Rate limited on {name} (attempt {attempt+1}/5). Waiting 40s...")
                    time.sleep(40)
                elif "503" in err_msg or "UNAVAILABLE" in err_msg or "500" in err_msg or "INTERNAL" in err_msg:
                    print(f"  Server busy/503 on {name} (attempt {attempt+1}/5). Waiting 15s...")
                    time.sleep(15)
                else:
                    print(f"  Error on {name}: {e}")
                    return {"score": None, "reason": str(e)}
        return {"score": None, "reason": "Max retries exceeded"}

    # Measure each metric
    ar = run_metric_safe(answer_relevancy_metric, "Answer Relevancy")
    time.sleep(2)

    faith = run_metric_safe(faithfulness_metric, "Faithfulness")
    time.sleep(2)

    cr = run_metric_safe(contextual_relevancy_metric, "Contextual Relevancy")
    time.sleep(2)

    corr = run_metric_safe(correctness_metric, "Correctness")
    time.sleep(2)

    item_result = {
        "question": record["question"],
        "expected_answer": record["expected_answer"],
        "actual_answer": record["actual_answer"],
        "context": record["context"],
        "answer_relevancy": ar,
        "faithfulness": faith,
        "contextual_relevancy": cr,
        "correctness": corr
    }

    eval_results.append(item_result)
    print(f"  AR: {ar['score']} | Faith: {faith['score']} | CR: {cr['score']} | Corr: {corr['score']}")

# Save complete evaluation output
with open(eval_output_path, "w", encoding="utf-8") as file:
    for res in eval_results:
        file.write(json.dumps(res) + "\n")

print(f"\nAll evaluations complete! Results saved to {eval_output_path}")
