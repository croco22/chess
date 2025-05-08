import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PARQUET_PATH = "../data/positions_2025_01.parquet"
ENGINE_PATH = "../stockfish/stockfish-windows-x86-64-avx2.exe"
ELO_MIN, ELO_MAX = 500, 2500
ELO_DEV = 250
SAMPLE_SIZE = 10_000

df = pd.read_parquet(PARQUET_PATH).sample(SAMPLE_SIZE).reset_index(drop=True)

# 1) Assign a random elo to each position in [ELO_MIN, ELO_MAX]
df["elo"] = np.random.randint(ELO_MIN, ELO_MAX + 1, size=len(df))

# 2) Compute winrate per (fen, next_move), but only using rows whose played_by_elo is within +/- ELO_DEV of each row's elo
#   - First, filter rows by elo proximity
df_filtered = df[np.abs(df.played_by - df.elo) <= ELO_DEV] # TODO: BACK TO PLAYED BY ELO
#   - Compute mean win_pov in the filtered set
grouped_win = (
    df_filtered
    .groupby(["fen", "next_move"])["win_pov"]
    .mean()
    .rename("winrate")
)
#   - Join back to the full df on (fen, next_move)
df = df.join(grouped_win, on=["fen", "next_move"])

# 3) Scale factors for score so they contribute equally
scaler = MinMaxScaler()
scaled_cols = ["fragility_score", "delta", "variance"]
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# 4) Recompute combined score
df["score"] = df[scaled_cols].sum(axis=1)

# 5) Determine historical_best per fen (based on winrate)
idx_best = df.groupby("fen")["winrate"].idxmax()
best_moves = df.loc[idx_best, ["fen", "next_move"]].set_index("fen")["next_move"]
df["historical_best"] = df["fen"].map(best_moves)

# 6) Determine recommended_move based on our new score
idx_best_score = df.groupby("fen")["score"].idxmax()
best_score_moves = df.loc[idx_best_score, ["fen", "next_move"]].set_index("fen")["next_move"]
df["recommended_move"] = df["fen"].map(best_score_moves)

# 7) Flag where recommendation matches history
df["is_best"] = df["recommended_move"].eq(df["historical_best"])

# 8) Save result
df.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Saved stats dataset to 'data/stats_dataset.parquet'")
