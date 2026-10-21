import json
from pathlib import Path

project_root = Path(__file__).resolve().parent

def load_experiment(filename: str):
    path = project_root / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def compare_all(experiment_files):
    experiments = []
    for f in experiment_files:
        exp = load_experiment(f)
        if exp:
            experiments.append(exp)

    if not experiments:
        print("No experiment files found!")
        return

    # Table Header
    print("\n" + "=" * 95)
    print(" " * 32 + "EXPERIMENT COMPARISON (ALL VERSIONS)")
    print("=" * 95 + "\n")

    header_cols = [f"{'Metric':<22}"]
    for exp in experiments:
        k_val = exp.get("retriever_k", "?")
        exp_name = exp.get("experiment", "Exp")
        header_cols.append(f"{f'{exp_name} (k={k_val})':<22}")

    print(" ".join(header_cols))
    print("-" * 95)

    # Collect all metrics
    metrics = [
        "answer_relevancy",
        "faithfulness",
        "context_relevancy",
        "correctness"
    ]

    for metric in metrics:
        row = [f"{metric.replace('_', ' ').title():<22}"]
        for exp in experiments:
            score = exp.get("metrics", {}).get(metric)
            if score is not None:
                row.append(f"{score:<22.2f}")
            else:
                row.append(f"{'N/A':<22}")
        print(" ".join(row))

    print("\n" + "=" * 95 + "\n")

if __name__ == "__main__":
    files = ["experiment_v1.json", "experiment_v2.json", "experiment_v3.json"]
    compare_all(files)