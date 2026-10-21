import json
from statistics import mean
from pathlib import Path

project_root = Path(__file__).resolve().parent

def save_experiment_summary(version: str, retriever_k: int):
    results_path = project_root / version / f"evaluated_results{version[-1]}.jsonl"
    if not results_path.exists():
        results_path = project_root / f"evaluated_results{version[-1]}.jsonl"

    if not results_path.exists():
        print(f"File not found: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as file:
        results = [json.loads(line) for line in file if line.strip()]

    def safe_mean(metric_key):
        scores = [r[metric_key] for r in results if r.get(metric_key) is not None]
        return mean(scores) if scores else None

    experiment = {
        "experiment": f"experiment_{version}_k{retriever_k}",
        "retriever_k": retriever_k,
        "number_of_test_cases": len(results),
        "metrics": {
            "answer_relevancy": safe_mean("answer_relevancy"),
            "faithfulness": safe_mean("faithfulness"),
            "context_relevancy": safe_mean("context_relevancy"),
            "correctness": safe_mean("correctness"),
        }
    }

    out_path = project_root / f"experiment_{version}.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(experiment, file, indent=4)

    print(f"Experiment saved to {out_path}")

if __name__ == "__main__":
    # Save for v1 (k=1), v2 (k=2), v3 (k=3)
    save_experiment_summary("v1", retriever_k=1)
    save_experiment_summary("v2", retriever_k=2)
    save_experiment_summary("v3", retriever_k=3)