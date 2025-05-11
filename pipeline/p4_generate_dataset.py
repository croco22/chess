import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PARQUET_PATH = "../data/score_dataset.parquet"
SAMPLE_ELOS = [500, 850, 1200, 1500, 2200]
ELO_DEV = 200

df = pd.read_parquet(PARQUET_PATH)

scaler = MinMaxScaler()
scaled_cols = ["fragility_score", "delta", "variance"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

df["fragility_score"] = 1 - df["fragility_score"]
df["variance"] = 1 - df["variance"]

df["score_base"] = df[scaled_cols].mean(axis=1)

for elo in SAMPLE_ELOS:
    winrate_col = f"winrate_{elo}"
    hist_col = f"historical_best_{elo}"
    score_col = f"score_{elo}"
    rec_col = f"recommended_move_{elo}"
    is_best_col = f"is_best_{elo}"
    is_engine_col = f"is_engine_best_{elo}"

    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    winrate = (
        df_elo.groupby(["fen", "next_move"])["win_pov"]
        .mean()
        .rename(winrate_col)
        .reset_index()
    )
    df = df.merge(winrate, on=["fen", "next_move"], how="left")
    df[winrate_col] = df[winrate_col].fillna(0.0) # Todo: What to do if not enough data

    idx_best = df.groupby("fen")[winrate_col].idxmax().dropna().astype(int)
    historical_best = df.loc[idx_best, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[hist_col] = df["fen"].map(historical_best)

    # Aktueller Score → später gewichtet ersetzen
    df[score_col] = df["score_base"]

    idx_best_score = df.groupby("fen")[score_col].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[rec_col] = df["fen"].map(recommended)

    df[is_best_col] = df[rec_col] == df[hist_col]
    df[is_engine_col] = df["engine_move"] == df[hist_col]

    # only fragility score as decider
    frag_rec_col = f"frag_rec_{elo}"
    is_frag_best_col = f"is_frag_best_{elo}"
    idx_best_score = df.groupby("fen")["fragility_score"].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[frag_rec_col] = df["fen"].map(recommended)
    df[is_frag_best_col] = df[frag_rec_col] == df[hist_col]

    # only delta as decider
    delta_rec_col = f"delta_rec_{elo}"
    is_delta_best_col = f"is_delta_best_{elo}"
    idx_best_score = df.groupby("fen")["delta"].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[delta_rec_col] = df["fen"].map(recommended)
    df[is_delta_best_col] = df[delta_rec_col] == df[hist_col]

    # weighted mix as decider
    mix_score_col = f"mix_score_{elo}"
    mix_rec_col = f"mix_rec_{elo}"
    is_mix_best_col = f"is_mix_best_{elo}"
    factor = min(max((elo - 1000) / 1000, 0.0), 1.0)
    fragility_weight = 0.5 * (1 - factor)
    delta_weight = 1.0 - fragility_weight
    df[mix_score_col] = (
            fragility_weight * df["fragility_score"] +
            delta_weight * df["delta"]
    )
    idx_best_score = df.groupby("fen")[mix_score_col].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[mix_rec_col] = df["fen"].map(recommended)
    df[is_mix_best_col] = df[mix_rec_col] == df[hist_col]

agg_dict = {
    "played_by_elo": "mean",    # Average Elo rating of players who played the move
    "win_pov": "mean",          # Average win rate from the point of view of the moving player
    "pair_freq": "first",       # Frequency of how often the (position, move) pair occurs
    "engine_move": "first",     # Move recommended by the chess engine
    "fragility_score": "first", # Fragility score of the position
    "delta": "first",           # Evaluation difference from the engine's best move
    "variance": "first",        # Variance across top engine move evaluations
    "score_base": "first",      # Base score combining fragility, delta, and variance
}

for elo in SAMPLE_ELOS:
    agg_dict.update({
        f"winrate_{elo}": "first",          # Win rate for players around this Elo level
        f"historical_best_{elo}": "first",  # Historically most successful move at this Elo level
        f"score_{elo}": "first",            # Elo-specific score (currently same as base score)
        f"recommended_move_{elo}": "first", # Move recommended based on the score at this Elo
        f"is_best_{elo}": "first",          # Whether the recommendation matches the historical best
        f"is_engine_best_{elo}": "first",   # Whether the engine move matches the historical best
        f"frag_rec_{elo}": "first",         # Move recommended based only on fragility score
        f"is_frag_best_{elo}": "first",     # Whether the fragility-based recommendation matches the historical best
        f"delta_rec_{elo}": "first",        # Move recommended based only on delta
        f"is_delta_best_{elo}": "first",    # Whether the delta-based recommendation matches the historical best
        f"mix_score_{elo}": "first",        # Score based on a weighted mix of fragility and delta
        f"mix_rec_{elo}": "first",          # Move recommended based on the mixed score
        f"is_mix_best_{elo}": "first",      # Whether the mixed-score recommendation matches the historical best
    })

grouped_df = df.groupby(["fen", "next_move"]).agg(agg_dict).reset_index()
grouped_df = grouped_df.rename(columns={
    "played_by_elo": "avg_elo",
    "win_pov": "global_winrate",
    "pair_freq": "count"
})

grouped_df.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Saved enriched dataset to 'data/stats_dataset.parquet'")
