import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import *

df = pd.read_parquet("../data/score_dataset_48h.parquet")
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# For each target Elo band, compute group-level statistics
for elo in ELOS:
    group_count_col = f"group_count_{elo}"
    winrate_col = f"winrate_{elo}"
    hist_col = f"historical_best_{elo}"
    is_hist_col = f"is_historical_best_{elo}"
    top2_col = f"historical_top2_{elo}"
    is_top2_col = f"is_historical_top2_{elo}"
    top3_col = f"historical_top3_{elo}"
    is_top3_col = f"is_historical_top3_{elo}"

    # Filter games played by players around this Elo level
    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    # Compute historical winrate for each position-move pair for this Elo range
    winrate = (
        df_elo.groupby(["fen", "next_move"])["win_pov"]
        .agg([("count", "count"), ("mean", "mean")])
        .rename(columns={"count": group_count_col, "mean": winrate_col})
        .reset_index()
    )
    df = df.merge(winrate, on=["fen", "next_move"], how="left")

    # Determine the historically most successful moves for each position
    max_wr = winrate.groupby("fen")[winrate_col].transform("max")
    winrate["max_wr"] = max_wr

    # historical_best
    best_thresh = winrate[winrate[winrate_col] >= winrate["max_wr"] - WINRATE_THRESHOLD]
    historical = best_thresh.groupby("fen")["next_move"].agg(list)
    hist_map = historical.to_dict()
    df[hist_col] = df["fen"].map(hist_map).apply(lambda x: x if isinstance(x, list) else [])
    df[is_hist_col] = df.apply(lambda r: r["next_move"] in r[hist_col], axis=1).astype("boolean")

    # historical_top2
    sorted_moves = (
        winrate.sort_values(by=["fen", winrate_col], ascending=[True, False])
        .groupby("fen")["next_move"]
        .apply(list)
        .to_dict()
    )
    df[top2_col] = df["fen"].map(lambda f: sorted_moves.get(f, [])[:2])
    df[is_top2_col] = df.apply(lambda r: r["next_move"] in r[top2_col], axis=1).astype("boolean")

    # historical_top3
    df[top3_col] = df["fen"].map(lambda f: sorted_moves.get(f, [])[:3])
    df[is_top3_col] = df.apply(lambda r: r["next_move"] in r[top3_col], axis=1).astype("boolean")

df.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Updated dataset with historical_best, top2 and top3 flags")
