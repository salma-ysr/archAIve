import json
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "arxiv-metadata-oai-snapshot.json"

def load_papers(chunk_size=50000):
    """
    Generator: yields DataFrames of raw papers in chunks.
    chunk_size: number of lines to read per iteration
    """
    rows = []
    with open(DATA_PATH, "r") as f:
        for i, line in enumerate(f):
            rows.append(json.loads(line))
            if (i + 1) % chunk_size == 0:
                df = pd.DataFrame(rows)
                df = df[["id", "title", "abstract", "categories", "authors"]]
                df["arxiv_link"] = "https://arxiv.org/abs/" + df['id'].astype(str)
                yield df
                rows = []
        # yield any remaining rows
        if rows:
            df = pd.DataFrame(rows)
            df = df[["id", "title", "abstract", "categories", "authors"]]
            yield df
