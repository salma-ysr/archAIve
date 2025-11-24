from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path
import faiss


ROOT = Path(__file__).resolve().parents[0]
ARTIFACTS = ROOT / "data" / "processed"
TOPK = 5

# Load artifacts
vectorizer = joblib.load(ARTIFACTS / "tfidf_vectorizer.joblib")
svd = joblib.load(ARTIFACTS / "svd_transformer.joblib")

faiss_index = faiss.read_index(str(ARTIFACTS / "faiss_index_ivf.idx"))

meta = pd.read_parquet(ARTIFACTS / "meta.parquet")

# Ensure link columns exist
if "arxiv_link" not in meta.columns:
    meta["arxiv_link"] = "https://arxiv.org/abs/" + meta["id"].astype(str)

app = Flask(__name__)

# Utilities
def clean_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z\s]", " ", q)
    return " ".join(w for w in q.split() if len(w) > 1)

def authors_to_str(a):
    if a is None:
        return "N/A"
    if isinstance(a, str):
        return a
    if isinstance(a, (list, tuple)):
        names = []
        for item in a:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                names.append(item.get("name") or "")
        return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
    return "N/A"

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    query = ""
    if request.method == "POST":
        query = request.form.get("query", "")
        cleaned = clean_query(query)

        # ----- Vectorize -----
        tfidf_vec = vectorizer.transform([cleaned])
        svd_vec = svd.transform(tfidf_vec).astype("float32")
        faiss.normalize_L2(svd_vec)

        # ----- FAISS search -----
        distances, indices = faiss_index.search(svd_vec, TOPK)
        sims = distances[0]

        # q_vec = vectorizer.transform([clean_query(query)])
        # distances, indices = knn.kneighbors(q_vec, n_neighbors=TOPK)
        # cosine = (1.0 - distances[0]).astype(float)
        
        recs = meta.iloc[indices[0]].copy().reset_index(drop=True)
        # convert to percentage
        recs["similarity"] = [f"{val*100:.1f}%" for val in sims]
        recs["authors"] = recs["authors"].apply(authors_to_str)

        recs["arxiv_link"] = recs.get("arxiv_link", "")
        recommendations = recs.to_dict(orient="records")

    return render_template("index.html", query=query, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
