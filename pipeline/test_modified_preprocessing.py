import pandas as pd

PARQUET_PATH = "../data/score_dataset.parquet"
MIN_SAMPLES_PER_GROUP = 20
WINRATE_THRESHOLD = 0.01
SAMPLE_ELOS = [800, 1000, 1200, 1500, 1800, 2000, 2200]
ELO_DEV = 300

def in_historical_best(fen_series, move_series, historical_map):
    return [(m in hist) for m, hist in zip(move_series, fen_series.map(historical_map))]

# Lade das Dataset
f = pd.read_parquet(PARQUET_PATH)
df = f.copy()

for elo in SAMPLE_ELOS:
    group_count_col = f"group_count_{elo}"
    winrate_col = f"winrate_{elo}"
    hist_col = f"historical_best_{elo}"
    is_hist_col = f"is_historical_best_{elo}"

    # Filter nach Elo-Range
    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    # Historische Winrate berechnen
    winrate = (
        df_elo.groupby(["fen", "next_move"])['win_pov']
        .agg([('count', 'count'), ('mean', 'mean')])
        .rename(columns={'count': group_count_col, 'mean': winrate_col})
        .reset_index()
    )
    df = df.merge(winrate, on=["fen", "next_move"], how="left")

    # Historisch beste Züge bestimmen
    max_wr = winrate.groupby('fen')[winrate_col].transform('max')
    winrate['max_wr'] = max_wr
    best_thresh = winrate[winrate[winrate_col] >= winrate['max_wr'] - WINRATE_THRESHOLD]
    historical = best_thresh.groupby('fen')['next_move'].agg(list)

    # Mapfen -> Liste historischer Züge
    hist_map = historical.to_dict()
    df[hist_col] = df['fen'].map(hist_map).apply(lambda x: x if isinstance(x, list) else [])

    # Boolean-Spalte, ob next_move in historisch bester Liste
    df[is_hist_col] = df.apply(lambda r: r['next_move'] in r[hist_col], axis=1)

    # Ungültige Fälle maskieren
    invalid = (df[group_count_col] < MIN_SAMPLES_PER_GROUP) | (df[group_count_col].isna())
    df.loc[invalid, is_hist_col] = pd.NA

# Aggregieren nach (fen, next_move)
agg_dict = {col: 'first' for col in ['games_count', 'played_by_elo', 'win_pov',
                                     'engine_move', 'delta', 'fragility_score', 'variance']}
for elo in SAMPLE_ELOS:
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