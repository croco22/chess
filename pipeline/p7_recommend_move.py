import pandas as pd

from config import *

df = pd.read_parquet("../data/stats_dataset.parquet")
coefficients = pd.read_parquet("../data/coefficients.parquet")

for elo in ELOS:
    elo_col = f"elo_{elo}"
    group_count_col = f"group_count_{elo}"
    score_col = f"score_{elo}"
    rec_col = f"recommended_move_{elo}"
    is_best_col = f"is_best_{elo}"
    is_top2_col = f"is_top2_{elo}"
    is_top3_col = f"is_top3_{elo}"
    is_engine1_col = f"is_engine_best_{elo}"
    is_engine2_col = f"is_engine_top2_{elo}"
    is_engine3_col = f"is_engine_top3_{elo}"
    hist_best_col = f"historical_best_{elo}"
    hist_top2_col = f"historical_top2_{elo}"
    hist_top3_col = f"historical_top3_{elo}"

    # Score function
    df[score_col] = (
            coefficients.loc["delta", elo_col] * df["delta"] +
            coefficients.loc["fragility_score", elo_col] * df["fragility_score"] +
            coefficients.loc["variance", elo_col] * df["variance"]
    )

    # Determine the move recommendation for each position based on the score
    idx_best = df.groupby('fen')[score_col].idxmax().dropna().astype(int)
    recommendation = df.loc[idx_best, ['fen', 'next_move']].set_index('fen')['next_move']
    df[rec_col] = df['fen'].map(recommendation)

    # Top flags
    df[is_best_col] = df.apply(lambda r: r[rec_col] in r[hist_best_col], axis=1).astype('boolean')
    if hist_top2_col in df.columns:
        df[is_top2_col] = df.apply(lambda r: r[rec_col] in r[hist_top2_col], axis=1).astype('boolean')
    else:
        df[is_top2_col] = pd.NA
    if hist_top3_col in df.columns:
        df[is_top3_col] = df.apply(lambda r: r[rec_col] in r[hist_top3_col], axis=1).astype('boolean')
    else:
        df[is_top3_col] = pd.NA

    # Engine flags
    df[is_engine1_col] = df.apply(lambda r: r['engine_move'] in r[hist_best_col], axis=1).astype('boolean')
    if hist_top2_col in df.columns:
        df[is_engine2_col] = df.apply(lambda r: r['engine_move'] in r[hist_top2_col], axis=1).astype('boolean')
    else:
        df[is_engine2_col] = pd.NA
    if hist_top3_col in df.columns:
        df[is_engine3_col] = df.apply(lambda r: r['engine_move'] in r[hist_top3_col], axis=1).astype('boolean')
    else:
        df[is_engine3_col] = pd.NA

    # Mask out positions with too few samples
    invalid_mask = (
            (df[group_count_col] < MIN_SAMPLES_PER_GROUP) |
            (df[group_count_col].isna())
    )
    df.loc[
        invalid_mask,
        [
            is_best_col,
            is_top2_col,
            is_top3_col,
            is_engine1_col,
            is_engine2_col,
            is_engine3_col
        ]
    ] = pd.NA

    # Exclude rows where engine move is not among the observed moves for that position
    valid_moves = df.groupby('fen')['next_move'].agg(set).to_dict()
    df['engine_valid_move'] = df.apply(
        lambda r: r['engine_move'] in valid_moves.get(r['fen'], set()), axis=1
    )
    df.loc[~df['engine_valid_move'], [is_engine1_col, is_engine2_col, is_engine3_col]] = pd.NA
    df.drop(columns='engine_valid_move', inplace=True)

df.to_parquet("../data/recommendation_dataset.parquet", index=False)
print("✅ Saved final dataset to 'data/recommendation_dataset.parquet'")
