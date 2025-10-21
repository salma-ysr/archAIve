import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_dir: Path) -> pd.DataFrame:
    csvs = list(results_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {results_dir}. Put your per-query results there.")

    frames = []
    for p in csvs:
        df = pd.read_csv(p)

        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # guess query from filename if missing
        if "query" not in df.columns:
            df["query"] = p.stem

        # required columns present check
        required = {"query", "rank", "cosine"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} missing required columns: {sorted(missing)}")

        # remove nums, strip whitespace in case
        df["rank"]   = pd.to_numeric(df["rank"], errors="coerce")
        df["cosine"] = pd.to_numeric(
            df["cosine"]
                .astype(str)
                .str.replace("%","", regex=False)   # in case someone pasted percents
                .str.replace(",",".", regex=False)  # stray commas to dots
                .str.strip(),
            errors="coerce"
        )

        # relevant column for precision (yes = 1, no = 0, labeled 'kinda' as 1)
        if "relevant" in df.columns:
            map_bool = {
                "y":1, "yes":1, "true":1, "t":1, "1":1,
                "n":0, "no":0, "false":0, "f":0, "0":0, "":0
            }
            df["relevant"] = (
                df["relevant"]
                  .astype(str).str.strip().str.lower()
                  .map(map_bool)
                  .fillna(pd.to_numeric(df["relevant"], errors="coerce"))
            ).fillna(0).astype(int)

        # drop junk rows
        df = df.dropna(subset=["rank","cosine","query"])

        # drop num or null entries
        df = df[df["query"].apply(lambda q: isinstance(q, str) and q.strip() != "" and not q.strip().isdigit())]

        # keep only columns we use
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def compute_precision_at_k(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if "relevant" not in df.columns:
        return pd.DataFrame()

    # top-k per query by 'rank'
    topk = df[df["rank"] <= k].copy()
    # make sure int 0/1
    topk["relevant"] = pd.to_numeric(topk["relevant"], errors="coerce").fillna(0).astype(int)

    prec = (topk.groupby("query")["relevant"]
                 .agg(["sum","count"])
                 .reset_index())
    prec["precision_at_k"] = prec["sum"] / prec["count"].clip(lower=1)
    return prec[["query","precision_at_k"]]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Folder with per-query CSVs")
    parser.add_argument("--out_dir", type=str, default="figures", help="Where to save PNGs")
    parser.add_argument("--k", type=int, default=5, help="k for Precision@k")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)

    # per-query cosine stats
    cos_stats = (
    df.groupby("query", dropna=True, sort=False)
      .agg(avg_cosine=("cosine", "mean"),
           max_cosine=("cosine", "max"))
      .reset_index()
    )


    # graph 1: bar chart of avg cosine per query
    x = np.arange(len(cos_stats))
    y = cos_stats["avg_cosine"].astype(float).values

    plt.figure()
    plt.bar(x, y)
    plt.ylabel("Average Top-k Cosine Similarity")
    plt.title("Average similarity per query")
    plt.xticks(x, cos_stats["query"].astype(str).values, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_cosine_per_query.png", dpi=200)
    plt.close()
    
    # graph 2: distribution of cosine similarities (all top-k across queries)
    all_cos = df["cosine"].values

    plt.figure()
    plt.boxplot(all_cos, vert=True)
    plt.ylabel("Cosine similarity")
    plt.title("Distribution of cosine similarities (all recommendations)")
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_distribution_boxplot.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(all_cos, bins=20)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Histogram of cosine similarities")
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_histogram.png", dpi=200)
    plt.close()

    # 3 precision@k and cosine–precision relationship
    prec_df = compute_precision_at_k(df, k=args.k)
    if not prec_df.empty:
        # merge w avg cosine for scatter
        merged = prec_df.merge(cos_stats[["query","avg_cosine"]], on="query", how="left")

        # graph 3a: bar chart of Precision@k per query
        x = np.arange(len(merged))
        y = merged["precision_at_k"].astype(float).values

        plt.figure()
        plt.bar(x, y)
        plt.ylabel(f"Precision@{args.k}")
        plt.title(f"Precision@{args.k} per query")
        plt.ylim(0, 1)
        plt.xticks(x, merged["query"].astype(str).values, rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"precision_at_{args.k}_per_query.png", dpi=200)
        plt.close()


        # graph 3b: scatter avg cosine vs Precision@k
        plt.figure()
        plt.scatter(merged["avg_cosine"], merged["precision_at_k"])
        plt.xlabel("Average Top-k Cosine Similarity")
        plt.ylabel(f"Precision@{args.k}")
        plt.title("Cosine similarity vs. Precision")
        plt.tight_layout()
        plt.savefig(out_dir / "cosine_vs_precision_scatter.png", dpi=200)
        plt.close()

        # print summary
        avg_p_at_k = merged["precision_at_k"].mean()
        print(f"Average Precision@{args.k}: {avg_p_at_k:.2f}")
        print("Per-query Precision@k:")
        print(merged.sort_values("precision_at_k", ascending=False)[["query","precision_at_k"]].to_string(index=False))
    else:
        print("No 'relevant' column found. Skipping Precision@k and scatter. "
              "Add a 'relevant' (0/1) column to enable Option B.")

    # save per-query table
    summary = cos_stats.copy()
    if not prec_df.empty:
        summary = summary.merge(prec_df, on="query", how="left")
    summary.to_csv(out_dir / "per_query_summary.csv", index=False)
    print(f"Saved figures to: {out_dir.resolve()}")
    print(f"Saved per-query summary CSV to: { (out_dir / 'per_query_summary.csv').resolve() }")

if __name__ == "__main__":
    main()