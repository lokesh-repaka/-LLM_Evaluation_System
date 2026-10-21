import sys
from pathlib import Path
import json
import os
import time
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models.llms.openai_model import GPTModel

# Subclass GPTModel to enforce global rate-limiting and automatic backoff on 429 errors
class RateLimitedGroqModel(GPTModel):
    _last_request_time: float = 0.0
    _min_interval: float = 1.8  # Enforces maximum ~33 requests per minute

    def _throttle(self):
        elapsed = time.time() - RateLimitedGroqModel._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        RateLimitedGroqModel._last_request_time = time.time()

    def generate(self, prompt, schema=None):
        for attempt in range(8):
            self._throttle()
            try:
                return super().generate(prompt, schema)
            except Exception as e:
                err_msg = str(e).lower()
                if "rate" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg:
                    wait_time = 15 + (attempt * 5)
                    print(f"\n[RateLimit Encountered] Groq burst reached. Sleeping {wait_time}s before retry {attempt+1}/8...")
                    time.sleep(wait_time)
                else:
                    raise e
        return super().generate(prompt, schema)

    async def a_generate(self, prompt, schema=None):
        import asyncio
        for attempt in range(8):
            self._throttle()
            try:
                return await super().a_generate(prompt, schema)
            except Exception as e:
                err_msg = str(e).lower()
                if "rate" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg:
                    wait_time = 15 + (attempt * 5)
                    print(f"\n[RateLimit Encountered] Groq burst reached (async). Sleeping {wait_time}s before retry {attempt+1}/8...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        return await super().a_generate(prompt, schema)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment!")

eval_model = RateLimitedGroqModel(
    model="qwen/qwen3.8-27b",
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

results_in_path = project_root / "v2" / "results2.jsonl"
if not results_in_path.exists():
    results_in_path = project_root / "results2.jsonl"

eval_output_path = project_root / "v2" / "evaluated_results2.jsonl"

with open(results_in_path, "r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

# Initialize metric objects
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=eval_model)
faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=eval_model)
contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7, model=eval_model)
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output",
    evaluation_params=[
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model=eval_model
)

# Load existing progress if available to resume seamlessly
evaluated_records = []
processed_indices = set()
if eval_output_path.exists():
    with open(eval_output_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line)
                    # Check if this record already has all 4 valid numeric scores
                    if (data.get("answer_relevancy") is not None and
                        data.get("faithfulness") is not None and
                        data.get("context_relevancy") is not None and
                        data.get("correctness") is not None):
                        evaluated_records.append(data)
                        processed_indices.add(idx)
                except Exception:
                    pass

print(f"Total test cases: {len(records)} | Already completed with full scores: {len(processed_indices)}")

def evaluate_metric_safely(metric, test_case, name):
    for attempt in range(5):
        try:
            metric.measure(test_case)
            return metric.score
        except Exception as e:
            err_msg = str(e).lower()
            if "rate" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg:
                wait_time = 20 + (attempt * 10)
                print(f"  [RateLimit] {name} hit 429. Backing off for {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [Error] {name}: {e}")
                time.sleep(3)
    return None

# Sequential evaluation loop for strictly controlled rate flow
for i, record in enumerate(records):
    if i in processed_indices:
        print(f"[{i+1}/{len(records)}] Skipping already evaluated test case.")
        continue

    print(f"\n[{i+1}/{len(records)}] Evaluating: {record['question'][:70]}...")

    context = record["context"]
    retrieval_context = [context] if isinstance(context, str) else context

    test_case = LLMTestCase(
        input=record["question"],
        actual_output=record["actual_answer"],
        expected_output=record["expected_answer"],
        retrieval_context=retrieval_context
    )

    # Measure Answer Relevancy
    ar_score = evaluate_metric_safely(answer_relevancy_metric, test_case, "Answer Relevancy")
    print(f"  -> Answer Relevancy: {ar_score}")
    time.sleep(1.0)

    # Measure Faithfulness
    faith_score = evaluate_metric_safely(faithfulness_metric, test_case, "Faithfulness")
    print(f"  -> Faithfulness: {faith_score}")
    time.sleep(1.0)

    # Measure Contextual Relevancy
    cr_score = evaluate_metric_safely(contextual_relevancy_metric, test_case, "Contextual Relevancy")
    print(f"  -> Contextual Relevancy: {cr_score}")
    time.sleep(1.0)

    # Measure Correctness
    corr_score = evaluate_metric_safely(correctness_metric, test_case, "Correctness")
    print(f"  -> Correctness: {corr_score}")
    time.sleep(1.0)

    evaluated_result = {
        **record,
        "answer_relevancy": ar_score,
        "faithfulness": faith_score,
        "context_relevancy": cr_score,
        "correctness": corr_score
    }
    evaluated_records.append(evaluated_result)

    # Write incrementally to disk so progress is never lost
    with open(eval_output_path, "w", encoding="utf-8") as file:
        for res in evaluated_records:
            file.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"\n==========================================")
print(f"All {len(evaluated_records)} test cases evaluated successfully!")
print(f"Results saved to: {eval_output_path}")
print(f"==========================================\n")