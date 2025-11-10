
"""
vibe_matcher_openai_fallback_plot.py

Enhanced Vibe Matcher prototype with:
- OpenAI embedding support (auto-detect via OPENAI_API_KEY)
- TF-IDF fallback
- Latency plot generation (saves latency.png automatically)
"""

import os
import time
import traceback
from timeit import default_timer as timer

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# optional import
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ----------------------------
# Sample product data
# ----------------------------
products = [
    {"name":"Boho Dress", "desc":"Flowy maxi dress with earthy tones, tassels and floral prints for festival and relaxed beach vibes.", "tags":["boho","festival","relaxed"]},
    {"name":"Urban Bomber Jacket", "desc":"Lightweight bomber jacket with reflective accents — street-ready, energetic and edgy.", "tags":["urban","energetic","street"]},
    {"name":"Cozy Knit Sweater", "desc":"Chunky knit, oversized sweater for warm, cozy nights and coffee shop hangs.", "tags":["cozy","casual","warm"]},
    {"name":"Minimal Blazer", "desc":"Tailored minimal blazer in neutral tones, perfect for modern office and smart-casual looks.", "tags":["minimal","office","smart"]},
    {"name":"Athleisure Leggings", "desc":"High-waist leggings with breathable fabric — sporty, energetic and comfortable for active days.", "tags":["sporty","energetic","comfortable"]},
    {"name":"Vintage Denim Jacket", "desc":"Washed denim jacket with retro patches — casual, timeless and slightly rebellious.", "tags":["vintage","casual","retro"]},
    {"name":"Silk Slip Dress", "desc":"Sleek silk slip dress, elegant and night-out ready with a touch of glamour.", "tags":["elegant","night","glam"]},
    {"name":"Plaid Flannel Shirt", "desc":"Soft flannel, rustic plaid pattern — perfect for cozy layered looks and autumn walks.", "tags":["cozy","rustic","layered"]},
]
df = pd.DataFrame(products)
df['id'] = df.index

# ----------------------------
# TF-IDF helpers
# ----------------------------
def tfidf_fit_and_embed(texts):
    vect = TfidfVectorizer().fit(texts)
    arr = vect.transform(texts).toarray()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms, vect)

def tfidf_embed_query(query, vect):
    arr = vect.transform([query]).toarray()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms)[0]

# ----------------------------
# OpenAI embedding helper (batch)
# ----------------------------
def openai_get_embeddings(texts, model="text-embedding-ada-002", batch_size=16, api_key=None):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed.")
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY set in environment.")
    openai.api_key = api_key

    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = openai.Embedding.create(model=model, input=chunk)
        for item in resp['data']:
            embeddings.append(item['embedding'])
    arr = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

# ----------------------------
# Build embeddings with fallback
# ----------------------------
def build_embeddings_with_fallback(descriptions):
    api_key = os.getenv("OPENAI_API_KEY")
    # Try OpenAI if key present and package available
    if api_key and OPENAI_AVAILABLE:
        try:
            start = timer()
            emb = openai_get_embeddings(descriptions, api_key=api_key)
            elapsed = timer() - start
            return emb, "openai", {"time_s": elapsed}
        except Exception as e:
            print("OpenAI attempt failed; falling back to TF-IDF.")
            print("Error:", str(e))
            traceback.print_exc()

    # Fallback to TF-IDF
    start = timer()
    emb, vect = tfidf_fit_and_embed(descriptions)
    elapsed = timer() - start
    return emb, "tfidf", {"time_s": elapsed, "vectorizer": vect}

# ----------------------------
# Search function (fixed)
# ----------------------------
def top_k_matches(query, embeddings, df, method_meta, k=3):
    q_emb = None

    if method_meta.get("method") == "openai":
        try:
            q_emb = openai_get_embeddings([query], api_key=os.getenv("OPENAI_API_KEY"))[0]
        except Exception as e:
            print("OpenAI query embedding failed; falling back to TF-IDF for query embedding.")
            print("Error:", str(e))
            traceback.print_exc()
            q_emb = None

    if q_emb is None:
        vect = method_meta.get("vectorizer")
        if vect is None:
            _, vect = tfidf_fit_and_embed(df['desc'].tolist())
        q_emb = tfidf_embed_query(query, vect)

    sims = cosine_similarity([q_emb], embeddings)[0]
    idxs = np.argsort(-sims)[:k]
    results = [{
        "id": int(df.loc[i,'id']),
        "name": df.loc[i,'name'],
        "desc": df.loc[i,'desc'],
        "score": float(sims[i])
    } for i in idxs]
    return results, float(np.max(sims)), sims

# ----------------------------
# Plot helper
# ----------------------------
def plot_latency(records, out_path="latency.png"):
    queries = [r['query'] for r in records]
    latencies = [r['latency_s'] for r in records]
    plt.figure(figsize=(6,3))
    plt.plot(queries, latencies, marker='o')
    plt.title("Search Latency per Query (s)")
    plt.ylabel("Latency (s)")
    plt.xlabel("Query")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Saved latency plot to: {out_path}")
    try:
        plt.show()
    except Exception:
        pass

# ----------------------------
# Main demo
# ----------------------------
def main():
    descs = df['desc'].tolist()
    print("Building embeddings (OpenAI if key present and package installed; otherwise TF-IDF)...")
    emb_result, method, meta = build_embeddings_with_fallback(descs)
    method_meta = {"method": method}
    method_meta.update(meta)

    print(f"Embeddings method: {method} (time={meta.get('time_s', 0):.3f}s)")

    queries = ["energetic urban chic", "cozy boho weekend", "minimal office wear"]
    threshold = 0.7
    records = []

    for q in queries:
        t0 = timer()
        results, top_score, sims = top_k_matches(q, emb_result, df, method_meta, k=3)
        latency = timer() - t0
        is_good = top_score > threshold
        records.append({"query": q, "results": results, "top_score": top_score, "latency_s": latency, "is_good": is_good})

    print("\nVibe Matcher — Results\n" + "-"*50)
    for r in records:
        print(f"Query: {r['query']}")
        for res in r['results']:
            print(f"  - {res['name']} (score={res['score']:.4f})")
        print(f"Top score: {r['top_score']:.4f} | Good (> {threshold}): {r['is_good']}")
        print(f"Latency: {r['latency_s']:.6f} s\n")

    # Plot latency and save
    plot_latency(records)

if __name__ == "__main__":
    main()
