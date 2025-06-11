FEATURES = ["delta", "fragility_score", "variance"]
FEATURE_LABELS = {
    "delta": "Rating Delta",
    "fragility_score": "Fragility",
    "variance": "Variance"
}
ELOS = list(range(500, 2500, 100))
ELO_DEV = 100
WINRATE_THRESHOLD = 0.03
MIN_SAMPLES_PER_GROUP = 15
