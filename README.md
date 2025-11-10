Vibe Matcher — README

Overview

**Vibe Matcher** is a small prototype recommendation system that maps a user "vibe" query (e.g., "energetic urban chic") to the top-3 matching fashion products using vector similarity.

This project supports two embedding modes:

* **OpenAI embeddings** (`text-embedding-ada-002`) — semantic, production-like embeddings (requires an API key).
* **TF-IDF fallback** — offline, deterministic embeddings for testing and demos.

The script automatically detects an `OPENAI_API_KEY` in the environment and will use OpenAI when available; otherwise it falls back to TF-IDF.

---

## Files included

* `vibe_matcher_openai_fallback_plot.py` — main script (OpenAI auto-detect + TF-IDF fallback + latency plotting)
* `vibe_matcher_openai_fallback_fixed.py` — corrected script (legacy; kept for reference)
* `vibe_matcher.py` — minimal TF-IDF-only script (simple demo)
* `requirements.txt` — dependencies
* `latency.png` (generated at runtime) — latency plot sample (saved by the script)

---

## Requirements

Install the Python dependencies (prefer a virtualenv):

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

* pandas
* numpy
* scikit-learn
* matplotlib
* openai (optional; only needed if you plan to use OpenAI)

---

## Setup: OpenAI API key (optional)

If you want to use OpenAI embeddings, create an API key from [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) and **do not** hardcode it in repository files.

Set it in your environment before running the script.

**macOS / Linux (bash):**

```bash
export OPENAI_API_KEY="sk-..."
```

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Google Colab:**

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

Alternatively, use a `.env` file and `python-dotenv` (not included by default). If the key is missing or the OpenAI call fails, the script automatically uses TF-IDF.

---

## How to run

1. Install requirements.
2. (Optional) Set `OPENAI_API_KEY` if you want semantic embeddings.
3. Run the script:

```bash
python vibe_matcher_openai_fallback_plot.py
```

The script will:

* Build embeddings for the sample product dataset (OpenAI if available, otherwise TF-IDF)
* Run three demo queries
* Print top-3 matches + scores + latency
* Save a latency plot as `latency.png` in the same folder

---

## Output explanation

* **Top matches**: The script prints product name, description, and cosine similarity score for the top-3 matches.
* **`is_good` flag**: A demo threshold (`> 0.7`) marks whether the top score is considered a strong match. This is arbitrary and should be tuned for real datasets.
* **Latency plot**: `latency.png` shows per-query matching time. In TF-IDF mode with a small dataset, latencies will be tiny. OpenAI mode will show higher times due to network/API call overhead.

---

## Tips & Next steps

* For production/scale: persist embeddings and use a vector database (Pinecone, FAISS, Weaviate) for fast ANN search.
* Use OpenAI embeddings for better semantic matching; batch embedding calls and cache results to reduce cost.
* Add evaluation: human-labeled queries, compute precision@k / recall@k / MRR, and tune threshold.
* Improve UI: add a simple Streamlit / Flask UI so users can type queries interactively.

---

## Troubleshooting

* `ModuleNotFoundError: No module named 'openai'`: install `openai` or run without OpenAI key (TF-IDF fallback will work).
* `latency.png` not created: ensure the script ran to completion (no exceptions); the plot saving code executes at the end of `main()`.
* If OpenAI calls fail due to rate limits or network, the script prints an error and falls back to TF-IDF.

---

## License

MIT — feel free to reuse and adapt for your assignment.

---

If you want, I can package all files into a ZIP for download (script + requirements + README). Say "create zip" and I will prepare it.
