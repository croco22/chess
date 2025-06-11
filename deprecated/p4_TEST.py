import pandas as pd
import numpy as np

PARQUET_PATH = "../data/score_dataset.parquet"
FEATURES = ["delta", "fragility_score", "variance"]
ELOS = list(range(1000, 2000, 100))
ELO_DEV = 300
WINRATE_THRESHOLD = 0.01
MIN_SAMPLES_PER_GROUP = 15

# 1. Datensatz einlesen
df = pd.read_parquet(PARQUET_PATH)

# 2. Jedem Datensatz die nächste Elo‐Stufe aus ELOS zuweisen:
#    Das sorgt dafür, dass jeder Datensatz genau einer Gruppe zugeordnet wird,
#    statt in überlappenden Fenstern mehrfach normalisiert zu werden.
def assign_nearest_elo(x, elo_list):
    # Liefert das elo in elo_list, das am nächsten an x liegt
    return min(elo_list, key=lambda e: abs(e - x))

df["elo_bin"] = df["played_by_elo"].apply(lambda x: assign_nearest_elo(x, ELOS))

# 3. Innerhalb jeder Elo-Bin: Min-Max-Normalisierung der FEATURES
#    (Wenn max == min, setzen wir alle Werte auf 0.5, siehe Kommentar).
#    Danach hat jede Zeile in df für delta/fragility_score/variance Werte in [0,1].
df_norm = df.copy()

for elo in ELOS:
    mask = df_norm["elo_bin"] == elo
    group = df_norm.loc[mask, FEATURES]
    if len(group) < MIN_SAMPLES_PER_GROUP:
        # Wenn zu wenig Samples im Bin, belassen wir die FEATURE‐Werte als NaN
        df_norm.loc[mask, FEATURES] = np.nan
        continue

    # Min und Max pro Feature in dieser Elo-Gruppe
    group_min = group.min()
    group_max = group.max()
    range_ = group_max - group_min

    # Normalisiere je Feature (Min‐Max), vermeide Division durch 0:
    for feat in FEATURES:
        if range_[feat] == 0:
            # Alle Werte identisch → setze konstant 0.5
            df_norm.loc[mask, feat] = 0.5
        else:
            df_norm.loc[mask, feat] = (group[feat] - group_min[feat]) / range_[feat]

# Überschreibe die ursprünglichen Features durch die normalisierten Werte
df[FEATURES] = df_norm[FEATURES]

# Entferne die temporäre Hilfsspalte 'elo_bin',
# falls du sie später nicht mehr brauchst (optional):
df.drop(columns=["elo_bin"], inplace=True)

# 4. Für jede Elo-Stufe: historische Winrate & Flags berechnen
for elo in ELOS:
    group_count_col = f"group_count_{elo}"
    winrate_col = f"winrate_{elo}"
    hist_col = f"historical_best_{elo}"
    is_hist_col = f"is_historical_best_{elo}"

    # 4.1. Filtere alle Zeilen, deren played_by_elo im Fenster [elo-ELO_DEV, elo+ELO_DEV] liegen
    mask = df["played_by_elo"].between(elo - ELO_DEV, elo + ELO_DEV)
    df_elo = df[mask].copy()

    # 4.2. Berechne winrate und Count pro (fen, next_move) in diesem Elo-Fenster
    winrate = (
        df_elo.groupby(["fen", "next_move"])["win_pov"]
        .agg([("count", "count"), ("mean", "mean")])
        .rename(columns={"count": group_count_col, "mean": winrate_col})
        .reset_index()
    )

    # 4.3. Merge diese beiden Spalten zurück in df
    df = df.merge(winrate, on=["fen", "next_move"], how="left")

    # 4.4. Finde für jede FEN den maximalen winrate_col‐Wert
    max_wr = winrate.groupby("fen")[winrate_col].transform("max")
    winrate["max_wr"] = max_wr

    # 4.5. Bestimme alle Moves, deren winrate >= (max_wr − WINRATE_THRESHOLD)
    best_thresh = winrate[winrate[winrate_col] >= (winrate["max_wr"] - WINRATE_THRESHOLD)]
    historical = best_thresh.groupby("fen")["next_move"].agg(list)
    hist_map = historical.to_dict()

    # 4.6. Lege eine Spalte historical_best_{elo} an,
    #      die pro Zeile eine Liste mit den historisch besten Zügen enthält
    df[hist_col] = df["fen"].map(lambda f: hist_map.get(f, []))

    # 4.7. Flag, ob der tatsächlich gespielte Move in dieser Liste ist
    df[is_hist_col] = df.apply(lambda r: r["next_move"] in r[hist_col], axis=1).astype("boolean")

# 5. Nun gruppieren wir pro (fen, next_move) und schreiben alles ins neue Dataset
agg_dict = {
    "games_count": "first",
    "played_by_elo": "mean",
    "win_pov": "mean",         # bleibt globaler Winrate-Wert
    "engine_move": "first",
}

# Die normalization in Schritt 3 hat die Original‐Features bereits überschrieben,
# sodass wir hier nur noch „first“ nehmen können.
for feat in FEATURES:
    agg_dict[feat] = "first"

for elo in ELOS:
    agg_dict.update({
        f"group_count_{elo}": "first",
        f"winrate_{elo}": "first",
        f"historical_best_{elo}": "first",
        f"is_historical_best_{elo}": "first"
    })

grouped = (
    df.groupby(["fen", "next_move"])
      .agg(agg_dict)
      .reset_index()
      .rename(columns={"played_by_elo": "global_avg_elo", "win_pov": "global_winrate"})
)

# 6. Ergebnis abspeichern
grouped.to_parquet("../data/stats_dataset.parquet", index=False)
print("✅ Updated dataset with Elo‐abhängig normalisierten Features und historical flags")
