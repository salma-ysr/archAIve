from pathlib import Path
import sys, re, joblib, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

def clean_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z\s]", " ", q)
    return " ".join(w for w in q.split() if len(w) > 1)

def squash_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def authors_to_str(a) -> str:
    """
    Handle diff authors formats like:
    >>string: "Doe, Smith, ..."
    >>list[str]: ["A. Doe", "B. Smith"]
    >>list[dict]: [{"name": "A. Doe"}, ...]
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
                # arXiv JSON sometimes uses {"name": "..."}
                name = item.get("name") or item.get("fullname") or ""
                if name:
                    names.append(squash_ws(name))
        if not names:
            return "N/A"
        # show first 3 to keep width tidy
        s = ", ".join(names[:3])
        if len(names) > 3:
            s += " et al."
        return s
    return "N/A"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 code/recommend.py \"your query text here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    vectorizer = joblib.load(OUT / "tfidf_vectorizer.joblib")
    knn = joblib.load(OUT / "knn_index.joblib")
    meta = pd.read_parquet(OUT / "meta.parquet")

    q_vec = vectorizer.transform([clean_query(query)])
    distances, indices = knn.kneighbors(q_vec, n_neighbors=5)

    recs = meta.iloc[indices[0]].copy()
    recs["similarity"] = (1 - distances[0]).round(4)

    # sanitize fields
    recs["title"] = recs["title"].map(squash_ws)
    recs["categories"] = recs["categories"].map(squash_ws)
    recs["authors"] = recs["authors"].map(authors_to_str)

    # Reorder columns
    recs = recs[["title", "authors", "categories", "similarity"]]

    print(f"\nQuery: \"{query}\"\n")
    print("Top-5 Recommended Papers:\n")
    print(f"{'Title':70} | {'Authors':30} | {'Categories':22} | {'Cosine Similarity'}")
    print("-" * 150)

    for _, row in recs.iterrows():
        title = (row["title"][:67] + "...") if len(row["title"]) > 70 else row["title"]
        authors = (row["authors"][:27] + "...") if len(row["authors"]) > 30 else row["authors"]
        cats = (row["categories"][:19] + "...") if len(row["categories"]) > 22 else row["categories"]
        print(f"{title:70} | {authors:30} | {cats:22} | {row['similarity']:.3f}")
