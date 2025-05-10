import pandas as pd
from sklearn.preprocessing import QuantileTransformer

PARQUET_PATH = "../data/score_dataset.parquet"
SAMPLE_ELOS = [500, 850, 1200, 1500, 2200]
ELO_DEV = 1000

df = pd.read_parquet(PARQUET_PATH)

scaler = QuantileTransformer(output_distribution="uniform")
scaled_cols = ["fragility_score", "delta", "variance"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])
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

    # NEW: fragility_score as decider
    frag_rec_col = f"frag_rec_{elo}"
    is_frag_best_col = f"is_frag_best_{elo}"
    idx_best_score = df.groupby("fen")["fragility_score"].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[frag_rec_col] = df["fen"].map(recommended)
    df[is_frag_best_col] = df[frag_rec_col] == df[hist_col]

    # NEW: mix as decider
    mix_score_col = f"mix_score_{elo}"
    mix_rec_col = f"mix_rec_{elo}"
    is_mix_best_col = f"is_mix_best_{elo}"
    factor = max(0.0, (elo - 1000) / 1000)
    fragility_weight = max(0.0, 0.5 * (1 - factor))
    delta_weight = 1 - fragility_weight
    df[mix_score_col] = (
            fragility_weight * df["fragility_score"] +
            delta_weight * df["delta"]
    )
    idx_best_score = df.groupby("fen")[mix_score_col].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
    df[mix_rec_col] = df["fen"].map(recommended)
    df[is_mix_best_col] = df[mix_rec_col] == df[hist_col]

agg_dict = {
    "played_by_elo": "mean",
    "win_pov": "mean",
    "engine_move": "first",
    "fragility_score": "first",
    "delta": "first",
    "variance": "first",
    "score_base": "first"
}

for elo in SAMPLE_ELOS:
    agg_dict.update({
        f"winrate_{elo}": "first",
        f"historical_best_{elo}": "first",
        f"score_{elo}": "first",
        f"recommended_move_{elo}": "first",
        f"is_best_{elo}": "first",
        f"is_engine_best_{elo}": "first",
        f"frag_rec_{elo}": "first",
        f"is_frag_best_{elo}": "first",
        f"mix_score_{elo}": "first",
        f"mix_rec_{elo}": "first",
        f"is_mix_best_{elo}": "first",
    })

grouped_df = df.groupby(["fen", "next_move"]).agg(agg_dict).reset_index()

grouped_df.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Saved enriched dataset to 'data/stats_dataset.parquet'")
