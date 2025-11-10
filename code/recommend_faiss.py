from pathlib import Path
import sys, re, joblib, pandas as pd
import numpy as np
import faiss
import time


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data" / "processed" # vectorizer, SVD, FAISS index, meta
RESULTS_DIR = ROOT / "results" 
TOPK = 5

def clean_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z\s]", " ", q)
    return " ".join(w for w in q.split() if len(w) > 1)

def squash_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def authors_to_str(a) -> str:
    """
    handles diff authors formats like
        string: "Doe, Smith, ..."
        list[str]: ["A. Doe", "B. Smith"]
        list[dict]: [{"name": "A. Doe"}, ...]
    """
    if a is None:
        return "N/A"
    if isinstance(a, str):
        return squash_ws(a) or "N/A"
    if isinstance(a, (list, tuple)):
        names = []
        for item in a:
            if isinstance(item, str):
                names.append(squash_ws(item))
            elif isinstance(item, dict):
                name = item.get("name") or item.get("fullname") or ""
                if name:
                    names.append(squash_ws(name))
        if not names:
            return "N/A"
        s = ", ".join(names[:3])
        if len(names) > 3:
            s += " et al."
        return s
    return "N/A"

def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "query"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 code/recommend.py "your query text here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    # load artifacts
    vectorizer = joblib.load(ARTIFACTS / "tfidf_vectorizer.joblib")
    svd = joblib.load(ARTIFACTS / "svd_transformer.joblib")
    index = faiss.read_index(str(ARTIFACTS / "faiss_index_ivf.idx"))
    meta = pd.read_parquet(ARTIFACTS / "meta.parquet")

    # Transform query
    q_vec = vectorizer.transform([clean_query(query)])
    q_vec = svd.transform(q_vec)
    q_vec = q_vec.astype("float32")
    faiss.normalize_L2(q_vec)

    # Search top-K
    similarities, indices = index.search(q_vec, TOPK)
    similarities = similarities[0]
    
    # build display dataframe
    valid_idx = [i for i in indices[0] if i < len(meta)]
    if not valid_idx:
        print("No valid recommendations found.")
        sys.exit(0)

    recs = meta.iloc[valid_idx].copy().reset_index(drop=True)
    similarities = similarities[:len(valid_idx)]
    recs["similarity"] = np.round(similarities, 4)
    recs["title"] = recs["title"].fillna("").map(squash_ws)
    recs["categories"] = recs.get("categories", "").map(squash_ws)
    recs["authors"] = recs.get("authors", "").map(authors_to_str)

    # pretty print
    print(f'\nQuery: "{query}"\n')
    print("Top-5 Recommended Papers:\n")
    print(f"{'Title':70} | {'Authors':30} | {'Categories':22} | {'Cosine Similarity'}")
    print("-" * 150)
    for _, row in recs.iterrows():
        title = (row["title"][:67] + "...") if len(row["title"]) > 70 else row["title"]
        authors = (row["authors"][:27] + "...") if len(row["authors"]) > 30 else row["authors"]
        cats = (row["categories"][:19] + "...") if len(row["categories"]) > 22 else row["categories"]
        print(f"{title:70} | {authors:30} | {cats:22} | {row['similarity']:.3f}")

    # save a per-query CSV for analysis
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    doc_id_col = "id" if "id" in recs.columns else ("doc_id" if "doc_id" in recs.columns else None)
    doc_ids = recs[doc_id_col] if doc_id_col else pd.Series(indices[0], name="doc_id")

    topk_df = pd.DataFrame({
        "query": [query] * len(recs),
        "rank": np.arange(1, len(recs) + 1),
        "doc_id": doc_ids.values,
        "title": recs["title"].values,
        "categories": recs["categories"].values,
        "similarity": recs["similarity"].astype(float).values,
    })

    out_path = RESULTS_DIR / f"{slug(query)}_{int(time.time())}.csv"
    topk_df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")
