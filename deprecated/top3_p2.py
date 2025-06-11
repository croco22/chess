import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.signal import savgol_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sns.set_style("darkgrid")

# ─── 1. Konstanten definieren ───────────────────────────────────────────────────
FEATURES = ['delta', 'fragility_score', 'variance']
ELOS = list(range(1000, 2000, 100))

# ─── 2. Dataset laden ────────────────────────────────────────────────────────────
df = pd.read_parquet("../data/stats_dataset.parquet")

coefficients = pd.DataFrame(index=FEATURES)
p_values = pd.DataFrame(index=FEATURES)

for elo in ELOS:
    target = f'is_historical_best_{elo}'
    subset = df.dropna(subset=FEATURES + [target])
    X = sm.add_constant(subset[FEATURES])
    y = subset[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1, test_size=0.25
    )

    model = sm.Logit(y_train, X_train)
    result = model.fit(disp=0)
    coefficients[f'elo_{elo}'] = result.params
    p_values[f'elo_{elo}'] = result.pvalues



# ─── 6. Move-Empfehlung berechnen (Top-1, Top-2, Top-3, Engine Top-1, Engine Top-2, Engine Top-3) ─
for elo in ELOS:
    # Spalten-Namen
    score_col       = f"score_{elo}"
    rec_col         = f"recommended_move_{elo}"
    is_best_col     = f"is_best_{elo}"
    is_top2_col     = f"is_top2_{elo}"
    is_top3_col     = f"is_top3_{elo}"
    is_engine1_col  = f"is_engine_best_{elo}"
    is_engine2_col  = f"is_engine_top2_{elo}"
    is_engine3_col  = f"is_engine_top3_{elo}"

    hist_best_col   = f"historical_best_{elo}"
    hist_top2_col   = f"historical_top2_{elo}"
    hist_top3_col   = f"historical_top3_{elo}"
    elo_col         = f"elo_{elo}"
    group_count_col = f"group_count_{elo}"

    # Score aus LogReg-Koeffizienten
    df[score_col] = (
        coefficients.loc['delta', elo_col] * df['delta'] +
        coefficients.loc['fragility_score', elo_col] * df['fragility_score'] +
        coefficients.loc['variance', elo_col] * df['variance']
    )

    # Empfehlung: Move mit höchstem Score
    idx_best = df.groupby('fen')[score_col].idxmax().dropna().astype(int)
    recommended = df.loc[idx_best, ['fen', 'next_move']].set_index('fen')['next_move']
    df[rec_col] = df['fen'].map(recommended)

    # Top-Flags
    df[is_best_col] = df.apply(lambda r: r[rec_col] in r[hist_best_col], axis=1).astype('boolean')
    if hist_top2_col in df.columns:
        df[is_top2_col] = df.apply(lambda r: r[rec_col] in r[hist_top2_col], axis=1).astype('boolean')
    else:
        df[is_top2_col] = pd.NA
    if hist_top3_col in df.columns:
        df[is_top3_col] = df.apply(lambda r: r[rec_col] in r[hist_top3_col], axis=1).astype('boolean')
    else:
        df[is_top3_col] = pd.NA

    # Engine-Flags
    df[is_engine1_col] = df.apply(lambda r: r['engine_move'] in r[hist_best_col], axis=1).astype('boolean')
    if hist_top2_col in df.columns:
        df[is_engine2_col] = df.apply(lambda r: r['engine_move'] in r[hist_top2_col], axis=1).astype('boolean')
    else:
        df[is_engine2_col] = pd.NA
    if hist_top3_col in df.columns:
        df[is_engine3_col] = df.apply(lambda r: r['engine_move'] in r[hist_top3_col], axis=1).astype('boolean')
    else:
        df[is_engine3_col] = pd.NA

    # Maskiere Positionen mit <20 Samples
    invalid = (df[group_count_col] < 20) | (df[group_count_col].isna())
    df.loc[invalid, [
        is_best_col, is_top2_col, is_top3_col,
        is_engine1_col, is_engine2_col, is_engine3_col
    ]] = pd.NA

    # Ungültige Engine-Moves entfernen
    valid_moves = df.groupby('fen')['next_move'].agg(set).to_dict()
    df['engine_valid_move'] = df.apply(
        lambda r: r['engine_move'] in valid_moves.get(r['fen'], set()), axis=1
    )
    df.loc[~df['engine_valid_move'], [is_engine1_col, is_engine2_col, is_engine3_col]] = pd.NA
    df.drop(columns='engine_valid_move', inplace=True)

# ─── 7. Ergebnis abspeichern ────────────────────────────────────────────────────
df.to_parquet("../data/recommendation_dataset.parquet", index=False)
print("✅ Saved enriched dataset to 'data/recommendation_dataset.parquet'")
