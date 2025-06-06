import pandas as pd
from sklearn.preprocessing import StandardScaler

PARQUET_PATH = "../data/score_dataset.parquet"
FEATURES = ["delta", "fragility_score", "variance"]
ELOS = list(range(1000, 2000, 100))
ELO_DEV = 300
WINRATE_THRESHOLD = 0.01
MIN_SAMPLES_PER_GROUP = 15

df = pd.read_parquet(PARQUET_PATH)
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# For each target Elo band, compute group-level statistics
for elo in ELOS:
    group_count_col = f"group_count_{elo}"
    winrate_col = f"winrate_{elo}"
    hist_col = f"historical_best_{elo}"
    is_hist_col = f"is_historical_best_{elo}"

    # NEU für Top2 und Top3:
    top2_col = f"historical_top2_{elo}"
    is_top2_col = f"is_historical_top2_{elo}"
    top3_col = f"historical_top3_{elo}"
    is_top3_col = f"is_historical_top3_{elo}"

    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    # Winrate pro (fen, next_move)
    winrate = (
        df_elo.groupby(["fen", "next_move"])["win_pov"]
            .agg([("count", "count"), ("mean", "mean")])
            .rename(columns={"count": group_count_col, "mean": winrate_col})
            .reset_index()
    )
    df = df.merge(winrate, on=["fen", "next_move"], how="left")

    # Max-Winrate pro FEN bestimmen
    max_wr = winrate.groupby("fen")[winrate_col].transform("max")
    winrate["max_wr"] = max_wr

    # ─── 1. historical_best ───
    best_thresh = winrate[winrate[winrate_col] >= winrate["max_wr"] - WINRATE_THRESHOLD]
    historical = best_thresh.groupby("fen")["next_move"].agg(list)
    hist_map = historical.to_dict()
    df[hist_col] = df["fen"].map(hist_map).apply(lambda x: x if isinstance(x, list) else [])
    df[is_hist_col] = df.apply(lambda r: r["next_move"] in r[hist_col], axis=1).astype("boolean")

    # ─── 2. historical_top2 ───
    # Wir sortieren zuerst alle Moves pro FEN nach winrate absteigend
    sorted_moves = (
        winrate.sort_values(by=["fen", winrate_col], ascending=[True, False])
        .groupby("fen")["next_move"]
        .apply(list)
        .to_dict()
    )
    # sorted_moves[f] ist nun eine Liste aller Züge für FEN f nach absteigender Winrate.
    # Top 2 nimmt man also sorted_moves[f][:2] (oder weniger, wenn es nur 1 Move gibt).
    df[top2_col] = df["fen"].map(lambda f: sorted_moves.get(f, [])[:2])
    df[is_top2_col] = df.apply(lambda r: r["next_move"] in r[top2_col], axis=1).astype("boolean")

    # ─── 3. historical_top3 ───
    df[top3_col] = df["fen"].map(lambda f: sorted_moves.get(f, [])[:3])
    df[is_top3_col] = df.apply(lambda r: r["next_move"] in r[top3_col], axis=1).astype("boolean")


# ─── Aggregations‐Dictionary erweitern ───
agg_dict = {
    "games_count": "first",
    "played_by_elo": "mean",
    "win_pov": "mean",
    "engine_move": "first",
    "delta": "first",
    "fragility_score": "first",
    "variance": "first",
}
for elo in ELOS:
    agg_dict.update({
        f"group_count_{elo}": "first",
        f"winrate_{elo}": "first",
        f"historical_best_{elo}": "first",
        f"is_historical_best_{elo}": "first",
        # ─── Hier neu hinzufügen: top2 & top3 Spalten ───
        f"historical_top2_{elo}": "first",
        f"is_historical_top2_{elo}": "first",
        f"historical_top3_{elo}": "first",
        f"is_historical_top3_{elo}": "first"
    })

grouped = (
    df.groupby(["fen", "next_move"])
      .agg(agg_dict)
      .reset_index()
      .rename(columns={"played_by_elo": "global_avg_elo", "win_pov": "global_winrate"})
)
grouped.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Updated dataset mit historical_best, top2 und top3 Flags")
