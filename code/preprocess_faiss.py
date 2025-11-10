from pathlib import Path
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from load_data_faiss import load_papers  # import the generator

# Ensure stopwords exist
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP = set(stopwords.words("english"))

# Paths
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "arxiv_clean.parquet"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(w for w in text.split() if w not in STOP and len(w) > 1)
    return text

if __name__ == "__main__":
    print("Preprocessing arXiv dataset in chunks...")

    parquet_files = []
    for i, chunk in enumerate(load_papers(chunk_size=50000)):
        # drop rows with missing abstracts and duplicates
        chunk = chunk.dropna(subset=["abstract"]).drop_duplicates(subset=["abstract"])

        # clean title and abstract
        chunk["clean_title"] = chunk["title"].map(clean_text)
        chunk["clean_abstract"] = chunk["abstract"].map(clean_text)

        # save chunk to a temp parquet file
        chunk_file = OUT_DIR / f"arxiv_chunk_{i+1}.parquet"
        chunk.to_parquet(chunk_file, index=False)
        parquet_files.append(chunk_file)
        print(f"  → Saved chunk {i+1} with shape {chunk.shape}")

    # merge all chunk files into a single parquet
    print("Merging all chunks into one parquet file...")
    all_chunks = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    all_chunks.to_parquet(OUT_FILE, index=False)
    print(f"Saved final cleaned dataset to {OUT_FILE} with shape {all_chunks.shape}")
