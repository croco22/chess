import pandas as pd

PARQUET_PATH = "../data/score_dataset.parquet"
SAMPLE_ELOS = [1000, 1200, 1500, 1800, 2000, 2200]
ELO_DEV = 200
MIN_SAMPLES_PER_GROUP = 20

df = pd.read_parquet(PARQUET_PATH)

# scaler = StandardScaler()
# scaled_cols = ["delta", "fragility_score", "variance"]
# df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Todo: Multiply factors using the coefficient signs
df["score"] = df["delta"]  # df[scaled_cols].mean(axis=1)

for elo in SAMPLE_ELOS:
    winrate_col = f"winrate_{elo}"
    group_count_col = f"group_count_{elo}"
    hist_col = f"historical_best_{elo}"
    score_col = f"score_{elo}"
    rec_col = f"recommended_move_{elo}"
    is_best_col = f"is_best_{elo}"
    is_engine_col = f"is_engine_best_{elo}"
    is_rec_top3_col = f"is_top3_{elo}"
    is_engine_top3_col = f"is_engine_top3_{elo}"

    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    winrate = (
        df_elo.groupby(["fen", "next_move"])["win_pov"]
        .agg([("mean", "mean"), ("count", "count")])
        .rename(columns={"mean": winrate_col, "count": group_count_col})
        .reset_index()
    )
    df = df.merge(winrate, on=["fen", "next_move"], how="left")

    idx_best = df.groupby("fen")[winrate_col].idxmax().dropna().astype(int)
    historical_best = df.loc[idx_best, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[hist_col] = df["fen"].map(historical_best)

    # Aktueller Score → später gewichtet ersetzen
    df[score_col] = df["score"]

    idx_best_score = df.groupby("fen")[score_col].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[rec_col] = df["fen"].map(recommended)

    df[is_best_col] = df[rec_col] == df[hist_col]
    df[is_engine_col] = df["engine_move"] == df[hist_col]

    df[is_best_col] = df[is_best_col].astype("boolean")
    df[is_engine_col] = df[is_engine_col].astype("boolean")
    invalid_mask = (df[group_count_col] < MIN_SAMPLES_PER_GROUP) | (df[group_count_col].isna())
    df.loc[invalid_mask, [is_best_col, is_engine_col]] = pd.NA


    def in_top3(fen_series, move_series):
        return [(f, m) in top3_set for f, m in zip(fen_series, move_series)]


    top3_moves = (
        df.groupby(["fen", "next_move"])[winrate_col]
        .first()
        .reset_index()
        .sort_values(by=["fen", winrate_col], ascending=[True, False])
        .groupby("fen")
        .head(3)
    )
    top3_set = set(zip(top3_moves["fen"], top3_moves["next_move"]))

    df[is_rec_top3_col] = in_top3(df["fen"], df[rec_col])
    df[is_engine_top3_col] = in_top3(df["fen"], df["engine_move"])

    df[is_rec_top3_col] = df[is_rec_top3_col].astype("boolean")
    df[is_engine_top3_col] = df[is_engine_top3_col].astype("boolean")
    df.loc[invalid_mask, [is_rec_top3_col, is_engine_top3_col]] = pd.NA


    # NOCH WAS NEUES
    # Setze NA, wenn der engine_move für die FEN nicht unter den beobachteten next_moves war
    valid_moves_per_fen = df.groupby("fen")["next_move"].agg(set).to_dict()

    df["engine_valid_move"] = df.apply(
        lambda row: row["engine_move"] in valid_moves_per_fen.get(row["fen"], set()),
        axis=1
    )

    # Engine-Zug ist nicht im Datensatz enthalten → NA setzen
    df.loc[~df["engine_valid_move"], [is_engine_col, is_engine_top3_col]] = pd.NA

    # Hilfsspalte wieder entfernen
    df.drop(columns="engine_valid_move", inplace=True)

agg_dict = {
    "games_count": "first",
    "played_by_elo": "mean",  # Average Elo rating of players who played the move
    "win_pov": "mean",  # Average win rate from the point of view of the moving player
    "engine_move": "first",  # Move recommended by the chess engine
    "delta": "first",  # Evaluation difference from the engine's best move
    "fragility_score": "first",  # Fragility score of the position
    "variance": "first",  # Variance across top engine move evaluations
    "score": "first",  # Base score combining fragility, delta, and variance
}

for elo in SAMPLE_ELOS:
    agg_dict.update({
        f"winrate_{elo}": "first",  # Win rate for players around this Elo level
        f"group_count_{elo}": "first",  # Number of entries around this Elo level
        f"historical_best_{elo}": "first",  # Historically most successful move at this Elo level
        f"score_{elo}": "first",  # Elo-specific score (currently same as base score)
        f"recommended_move_{elo}": "first",  # Move recommended based on the score at this Elo
        f"is_best_{elo}": "first",  # Whether the recommendation matches the historical best
        f"is_engine_best_{elo}": "first",  # Whether the engine move matches the historical best
        f"is_top3_{elo}": "first",  # Whether recommended move is in top 3 historical moves
        f"is_engine_top3_{elo}": "first",  # Whether engine move is in top 3 historical moves
    })

grouped_df = df.groupby(["fen", "next_move"]).agg(agg_dict).reset_index()
grouped_df = grouped_df.rename(columns={
    "played_by_elo": "global_avg_elo",
    "win_pov": "global_winrate"
})

grouped_df.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Saved enriched dataset to 'data/stats_dataset.parquet'")
