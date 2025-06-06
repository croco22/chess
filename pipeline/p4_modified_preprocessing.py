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

    # Filter games played by players around this Elo level
    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    # Compute historical winrate per position-move pair for this Elo range
    winrate = (
        df_elo.groupby(["fen", "next_move"])['win_pov']
        .agg([('count', 'count'), ('mean', 'mean')])
        .rename(columns={'count': group_count_col, 'mean': winrate_col})
        .reset_index()
    )
    df = df.merge(winrate, on=["fen", "next_move"], how="left")

    # Determine the historically most successful moves per FEN
    max_wr = winrate.groupby('fen')[winrate_col].transform('max')
    winrate['max_wr'] = max_wr
    best_thresh = winrate[winrate[winrate_col] >= winrate['max_wr'] - WINRATE_THRESHOLD]
    historical = best_thresh.groupby('fen')['next_move'].agg(list)
    hist_map = historical.to_dict()
    df[hist_col] = df['fen'].map(hist_map).apply(lambda x: x if isinstance(x, list) else [])

    # Create a boolean column indicating whether the actual next move is one of the historically best moves
    df[is_hist_col] = df.apply(lambda r: r['next_move'] in r[hist_col], axis=1).astype('boolean')

# Aggregate to one row per position-move pair
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
        f'group_count_{elo}': 'first',
        f'winrate_{elo}': 'first',
        f'historical_best_{elo}': 'first',
        f'is_historical_best_{elo}': 'first'
    })

grouped = (
    df.groupby(['fen', 'next_move'])
    .agg(agg_dict)
    .reset_index()
    .rename(columns={'played_by_elo': 'global_avg_elo', 'win_pov': 'global_winrate'})
)

grouped.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Updated dataset with historical flags")
