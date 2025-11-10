from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import joblib

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "arxiv_small_clean.parquet"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_parquet(CLEAN)
    texts = df["clean_abstract"].fillna("")

    vectorizer = TfidfVectorizer(max_features=5000) #term frequency-inverse document frequency (how rare term is across documents)
    X = vectorizer.fit_transform(texts) #learn idf and convert abstracts into tf-idf vectors

    knn = NearestNeighbors(n_neighbors=10, metric="cosine") #find nearest neighbors based on cosine similarity (1 max similarity, 0 min)
    knn.fit(X) 

    joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib") #weighted importance of words (common vs rare)
    joblib.dump(knn, OUT / "knn_index.joblib")
    sparse.save_npz(OUT / "tfidf_matrix.npz", X)
    df[["id", "title", "categories", "authors", "abstract", "clean_abstract"]].to_parquet(
    OUT / "meta.parquet", index=False
    )
    print("Built TF-IDF + KNN and saved to data/processed/")
