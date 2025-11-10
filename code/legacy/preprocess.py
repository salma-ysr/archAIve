# preprocess and clean a small subset of arXiv data
import json
import re
from pathlib import Path
import pandas as pd
import nltk

# make sure stopwords exist
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))

# paths
ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "data" / "arxiv-metadata-oai-snapshot.json"

if not RAW_JSON.exists():
    raise FileNotFoundError(f"Could not find dataset at {RAW_JSON}. "
                            f"Run `ls {RAW_JSON.parent}` to verify the filename.")

OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "arxiv_small_clean.parquet"

def load_subset(max_rows: int = 5000) -> pd.DataFrame: #only using 5000 rows for now! Todo: increase to full dataset
    """Load a small subset line-by-line from the giant JSON."""
    rows = []
    with open(RAW_JSON, "r") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    useful_cols = ["id", "title", "abstract", "categories", "authors"]
    df = df[useful_cols]
    return df

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(w for w in text.split() if w not in STOP and len(w) > 1)
    return text

if __name__ == "__main__":
    df = load_subset(max_rows=5000)

    # rm any duplicates or missing abstracts
    df = df.dropna(subset=["abstract"]).drop_duplicates(subset=["abstract"])

    # cleaned fields
    df["clean_title"] = df["title"].map(clean_text)
    df["clean_abstract"] = df["abstract"].map(clean_text)

    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved cleaned subset to {OUT_FILE} with shape={df.shape}")
