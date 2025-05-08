import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PARQUET_PATH = "../data/score_dataset.parquet"
ENGINE_PATH = "../stockfish/stockfish-windows-x86-64-avx2.exe"
ELO_MIN, ELO_MAX = 500, 2500
ELO_DEV = 2000  # quatsch

df = pd.read_parquet(PARQUET_PATH)
df["random_elo"] = np.random.randint(ELO_MIN, ELO_MAX + 1, size=len(df))
df_filtered = df[np.abs(df.played_by_elo - df["random_elo"]) <= ELO_DEV]

grouped_win = (
    df_filtered
    .groupby(["fen", "next_move"])["win_pov"]
    .mean()
    .rename("winrate")
)
df = df.merge(grouped_win, on=["fen", "next_move"], how="left")

scaler = MinMaxScaler()
scaled_cols = ["fragility_score", "delta", "variance"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

df["score"] = df[scaled_cols].sum(axis=1)

idx_best = df.groupby("fen")["winrate"].idxmax()
idx_best = idx_best.dropna().astype(int)
best_moves = df.loc[idx_best, ["fen", "next_move"]].set_index("fen")["next_move"]
df["historical_best"] = df["fen"].map(best_moves)

idx_best_score = df.groupby("fen")["score"].idxmax()
best_score_moves = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
df["recommended_move"] = df["fen"].map(best_score_moves)

df["is_best"] = df["recommended_move"].eq(df["historical_best"])

df.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Saved stats dataset to 'data/stats_dataset.parquet'")
