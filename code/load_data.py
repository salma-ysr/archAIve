import json
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "arxiv-metadata-oai-snapshot.json"

def load_subset(max_rows=5000):
    rows = []
    with open(DATA_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    useful_cols = ["id", "title", "abstract", "categories", "authors"]
    df = df[useful_cols]
    return df

if __name__ == "__main__":
    df = load_subset()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(2))