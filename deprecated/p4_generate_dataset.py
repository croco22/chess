import pandas as pd
from sklearn.preprocessing import StandardScaler

PARQUET_PATH = "../data/score_dataset.parquet"
SAMPLE_ELOS = [800, 1000, 1200, 1500, 1800, 2000, 2200]
ELO_DEV = 300
MIN_SAMPLES_PER_GROUP = 20
WINRATE_THRESHOLD = 0.01

# Utility function to check if (fen, move) pair is in the set of top 3 moves
def in_top3(fen_series, move_series, top3):
    return [(f, m) in top3 for f, m in zip(fen_series, move_series)]


df = pd.read_parquet(PARQUET_PATH)

# Standardize the key evaluation columns
scaler = StandardScaler()
scaled_cols = ["delta", "fragility_score", "variance"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

for elo in SAMPLE_ELOS:
    group_count_col = f"group_count_{elo}"
    winrate_col = f"winrate_{elo}"
    score_col = f"score_{elo}"
    rec_col = f"recommended_move_{elo}"
    hist_col = f"historical_best_{elo}"
    is_best_col = f"is_best_{elo}"
    is_engine_col = f"is_engine_best_{elo}"
    is_rec_top3_col = f"is_top3_{elo}"
    is_engine_top3_col = f"is_engine_top3_{elo}"

    # Filter games played by players around this Elo level
    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    # Compute historical winrate per (fen, move) pair for this Elo range
    winrate = (
        df_elo.groupby(["fen", "next_move"])["win_pov"]
        .agg([("count", "count"), ("mean", "mean")])
        .rename(columns={"count": group_count_col, "mean": winrate_col})
        .reset_index()
    )
    df = df.merge(winrate, on=["fen", "next_move"], how="left")

    # Assign score
    delta_weight = 0.5
    factor = min(max((elo - 1000) / 1000, 0.0), 1.0)
    fragility_weight = 0.5 * (1 - factor)
    variance_weight = 1.0 - delta_weight - fragility_weight
    df[score_col] = (
        fragility_weight * df["fragility_score"] +
        delta_weight * df["delta"] +
        variance_weight * df["variance"]
    )

    # Determine the recommended move per FEN based on the score
    idx_best_score = df.groupby("fen")[score_col].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[rec_col] = df["fen"].map(recommended)

    # Determine the historically most successful moves per FEN
    max_winrate = winrate.groupby("fen")[winrate_col].transform("max")
    winrate["max_winrate"] = max_winrate
    best_thresh = winrate[winrate[winrate_col] >= winrate["max_winrate"] - WINRATE_THRESHOLD]
    historical = best_thresh.groupby("fen")["next_move"].agg(list)
    df[hist_col] = df["fen"].map(historical).apply(lambda x: x if isinstance(x, list) else [])

    # Compare recommendations and engine move with historical best
    df[is_best_col] = df.apply(lambda r: r[rec_col] in r[hist_col], axis=1).astype('boolean')
    df[is_engine_col] = df.apply(lambda r: r["engine_move"] in r[hist_col], axis=1).astype('boolean')

    # Mask out positions with too few samples or missing data
    invalid_mask = (
        (df[group_count_col] < MIN_SAMPLES_PER_GROUP) |
        (df[group_count_col].isna())
    )
    df.loc[invalid_mask, [is_best_col, is_engine_col]] = pd.NA

    # Identify top 3 highest-winrate moves per FEN
    top3_moves = (
        df.groupby(["fen", "next_move"])[winrate_col]
        .first()
        .reset_index()
        .sort_values(by=["fen", winrate_col], ascending=[True, False])
        .groupby("fen")
        .head(3)
    )
    top3_set = set(zip(top3_moves["fen"], top3_moves["next_move"]))

    # Check if recommended or engine move is in the top 3
    df[is_rec_top3_col] = in_top3(df["fen"], df[rec_col], top3_set)
    df[is_engine_top3_col] = in_top3(df["fen"], df["engine_move"], top3_set)
    df[is_rec_top3_col] = df[is_rec_top3_col].astype("boolean")
    df[is_engine_top3_col] = df[is_engine_top3_col].astype("boolean")
    df.loc[invalid_mask, [is_rec_top3_col, is_engine_top3_col]] = pd.NA

    # Exclude rows where engine move is not among the observed moves for that position
    valid_moves_per_fen = df.groupby("fen")["next_move"].agg(set).to_dict()
    df["engine_valid_move"] = df.apply(
        lambda row: row["engine_move"] in valid_moves_per_fen.get(row["fen"], set()),
        axis=1
    )
    df.loc[~df["engine_valid_move"], [is_engine_col, is_engine_top3_col]] = pd.NA
    df.drop(columns="engine_valid_move", inplace=True)

# Aggregate to one row per (FEN, next_move) pair
agg_dict = {
    "games_count": "first",
    "played_by_elo": "mean",
    "win_pov": "mean",
    "engine_move": "first",
    "delta": "first",
    "fragility_score": "first",
    "variance": "first",
}
for elo in SAMPLE_ELOS:
    agg_dict.update({
        f"group_count_{elo}": "first",
        f"winrate_{elo}": "first",
        f"score_{elo}": "first",
        f"recommended_move_{elo}": "first",
        f"historical_best_{elo}": "first",
        f"is_best_{elo}": "first",
        f"is_engine_best_{elo}": "first",
        f"is_top3_{elo}": "first",
        f"is_engine_top3_{elo}": "first",
    })

grouped_df = df.groupby(["fen", "next_move"]).agg(agg_dict).reset_index()
grouped_df = grouped_df.rename(columns={
    "played_by_elo": "global_avg_elo",
    "win_pov": "global_winrate"
})

grouped_df.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Saved enriched dataset to 'data/stats_dataset.parquet'")
