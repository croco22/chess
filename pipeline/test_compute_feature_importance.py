import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

# Pfad zum angereicherten Datensatz
df = pd.read_parquet("../data/stats_dataset.parquet")
SAMPLE_ELOS = [800, 1000, 1200, 1500, 1800, 2000, 2200]
FEATURES = ['delta', 'fragility_score', 'variance']

# Für jede Elo-Stufe: XGBClassifier trainieren und Feature-Importances sammeln
importances = pd.DataFrame(index=FEATURES)
for elo in SAMPLE_ELOS:
    target = f'is_historical_best_{elo}'
    subset = df.dropna(subset=FEATURES + [target])
    X = subset[FEATURES]
    y = subset[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    imp = model.feature_importances_
    importances[f'elo_{elo}'] = imp

# Durchschnittliche Importance über alle Elos
importances['mean_importance'] = importances.mean(axis=1)

# Normalisieren
total = importances['mean_importance'].sum()
importances['weight'] = importances['mean_importance'] / total

# Score-Berechnung: gewichtete Summe der Features
# Beispiel: für jedes Sample
weights = importances['weight'].to_dict()
df['xgb_score'] = (
    df['delta'] * weights['delta'] +
    df['fragility_score'] * weights['fragility_score'] +
    df['variance'] * weights['variance']
)

# Speichern der Gewichte und Scores
importances.to_csv('../data/feature_importances.csv')
df.to_parquet('../data/scored_stats_dataset.parquet', index=False)
print("✅ Computed XGBoost feature weights and scores")