from pathlib import Path
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from load_data_faiss import load_papers

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(w for w in text.split() if w not in STOP and len(w) > 1)
    return text

if __name__ == "__main__":
    for i, chunk in enumerate(load_papers(chunk_size=50000), start=1):
        chunk = chunk.dropna(subset=["abstract"]).drop_duplicates(subset=["abstract"])
        chunk["clean_title"] = chunk["title"].map(clean_text)
        chunk["clean_abstract"] = chunk["abstract"].map(clean_text)
        out = OUT_DIR / f"arxiv_chunk_{i}.parquet"
        chunk.to_parquet(out, index=False)
        print(f"saved {out}")
    print("done.")
