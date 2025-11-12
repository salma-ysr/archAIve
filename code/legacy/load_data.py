# load a
import json
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "arxiv-metadata-oai-snapshot.json"

def load_subset(max_rows=1000_000):
    rows = []
    with open(DATA_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    useful_cols = ["id", "title", "abstract", "categories", "authors"]
    if "doi" in df.columns:
        useful_cols.append("doi")
    df = df[useful_cols]
    df['arxiv_link'] = "https://arxiv.org/abs/" + df['id'].astype(str)
    if 'doi' in df.columns:
        df['doi_link'] = df['doi'].apply(lambda d: f"https://doi.org/{d}" if pd.notna(d) else "")
    return df

if __name__ == "__main__":
    df = load_subset()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(2))