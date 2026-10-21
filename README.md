# LLM & RAG Evaluation System 🚀

An end-to-end framework for evaluating, comparing, and benchmarking Retrieval-Augmented Generation (RAG) pipelines across multiple retriever configurations using **DeepEval**, **LangChain**, and **Streamlit**.

---

## 📌 Architecture & Overview

This system benchmarks a question-answering RAG pipeline over domain documents (e.g. Penguins ACL domain knowledge) by varying retrieval parameters (such as $k \in \{1, 2, 3\}$) and evaluating the performance against four core RAG triads and generation metrics:

1. **Answer Relevancy**: Assesses how directly and pertinently the model's answer addresses the user prompt.
2. **Faithfulness**: Measures whether the generated answer is grounded in the retrieved context without hallucinations.
3. **Contextual Relevancy**: Measures the signal-to-noise ratio in the retrieved passages.
4. **Correctness (G-Eval)**: Evaluates factual alignment of the answer against gold-standard expected answers.

---

## 📊 Benchmark Results

| Experiment | Retriever $k$ | Answer Relevancy | Faithfulness | Context Relevancy | Correctness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`v1` (experiment_v1)** | **$k=1$** | 0.92 | **1.00** | **0.98** | **0.93** |
| **`v2` (experiment_v2)** | **$k=2$** | **1.00** | **1.00** | 0.78 | 0.91 |
| **`v3` (experiment_v3)** | **$k=3$** | **1.00** | **1.00** | 0.79 | 0.89 |

> **Key Finding:** $k=1$ achieved the highest Context Relevancy (0.98) and Correctness (0.93) with minimal noise and lowest token cost/latency. Higher $k$ introduced context drag/noise without improving factual correctness.

---

## 📁 Repository Structure

```
.
├── RAG1/                           # RAG pipeline implementation
│   ├── agent.py                    # Ask function and query coordinator
│   ├── indexing.py                 # Vector store indexing (Chroma / Embeddings)
│   ├── tools.py                    # Retriever tools
│   ├── prompts.py                  # System prompts
│   ├── objects.py                  # Pydantic schemas & state representations
│   └── file_reader.py              # PDF extraction & text processing
├── pdfs/                           # Source domain knowledge PDFs
├── eval_dataset.jsonl              # 20 curated QA evaluation test cases
├── v1/                             # Experiment 1 (k=1) evaluation runner & results
├── v2/                             # Experiment 2 (k=2) evaluation runner & results
├── v3/                             # Experiment 3 (k=3) evaluation runner & results
├── single_test/                    # Standalone metric unit tests
├── save_experiment.py              # Consolidates & serializes metric summaries
├── compare_experiments.py          # Terminal CLI comparison across runs
├── dashboard.py                    # Interactive Streamlit analytics dashboard
├── pyproject.toml                  # Project dependencies & configuration
├── .env.example                    # Environment variable template
└── README.md                       # Documentation
```

---

## 🛠️ Quick Start

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/lokesh-repaka/-LLM_Evaluation_System.git
cd -LLM_Evaluation_System
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or uv sync
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
In `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Running RAG Generation & Evaluation
To run the evaluation for any experiment version (e.g., `v1`):
```bash
python v1/eval_runner1.py            # Generates RAG answers
python v1/eval_all_metrics1_fast.py  # Evaluates metrics using rate-limited judge
python v1/evaluation_summary1.py     # Prints summary stats for v1
```

### 4. Comparing Experiments (CLI)
Compare results across $k=1$, $k=2$, and $k=3$:
```bash
python save_experiment.py
python compare_experiments.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run dashboard.py
```
View the interactive charts and run comparisons at `http://localhost:8501`.

---

## 🛡️ Robust Rate-Limiting & Flow Control
This system implements `RateLimitedGroqModel`, a resilient wrapper around the LLM judge client:
- **Automatic Request Throttling**: Enforces pacing between metric calls to stay under provider rate limits.
- **Exponential Backoff**: Intercepts `429` (Quota/Rate Limit) bursts and retries with backoff intervals.
- **Incremental Checkpointing**: Saves test evaluations incrementally so progress is preserved even if interrupted.

---

## 📜 License
MIT License.
