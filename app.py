from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
ARTIFACTS = ROOT / "data" / "processed"
TOPK = 5

# Load artifacts
vectorizer = joblib.load(ARTIFACTS / "tfidf_vectorizer.joblib")
knn = joblib.load(ARTIFACTS / "knn_index.joblib")
meta = pd.read_parquet(ARTIFACTS / "meta.parquet")

# Ensure link columns exist
if "arxiv_link" not in meta.columns:
    meta["arxiv_link"] = "https://arxiv.org/abs/" + meta["id"].astype(str)
if "doi" in meta.columns and "doi_link" not in meta.columns:
    meta["doi_link"] = meta["doi"].apply(lambda d: f"https://doi.org/{d}" if pd.notna(d) else "")
else:
    meta["doi_link"] = ""

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
        q_vec = vectorizer.transform([clean_query(query)])
        distances, indices = knn.kneighbors(q_vec, n_neighbors=TOPK)
        cosine = (1.0 - distances[0]).astype(float)
        recs = meta.iloc[indices[0]].copy().reset_index(drop=True)
        recs["similarity"] = np.round(cosine, 4)
        recs["authors"] = recs["authors"].apply(authors_to_str)

        recs["arxiv_link"] = recs.get("arxiv_link", "")
        recs["doi_link"] = recs.get("doi_link", "")

        recommendations = recs.to_dict(orient="records")
    return render_template("index.html", query=query, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
