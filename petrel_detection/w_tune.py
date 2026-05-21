import pandas as pd

# Path to the experiment summary CSV file
SUMMARY_CSV = r"E:\petrel_project\experiment_summary.csv"

summary_df = pd.read_csv(SUMMARY_CSV)

# Test different weights for Miss_rate in the scoring function
for weight in [0.5, 0.4, 0.3, 0.2, 0.1]:
    # Calculate composite score: FDR + weight * Miss_rate
    summary_df["score"] = summary_df["FDR"] + weight * summary_df["Miss_rate"]
    # Sort by score (ascending = best performance first)
    summary_df = summary_df.sort_values("score").reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"Weight = {weight}  (Score = FDR + {weight} × Miss_rate)")
    print(f"{'=' * 60}")
    # Print sorted results table
    print(
        summary_df[[
            "tag", "beta", "gamma",
            "FDR", "Miss_rate", "Recall", "score"
        ]].to_string(index=False)
    )
    # Get the best (top-ranked) result
    best = summary_df.iloc[0]
    print(f"\n✅ Best: {best['tag']}  FDR={best['FDR']:.4f}  Miss_rate={best['Miss_rate']:.4f}")