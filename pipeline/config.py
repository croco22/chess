FEATURES = ["delta", "fragility_score", "variance"]
FEATURE_LABELS = {
    "delta": "Rating Delta",
    "fragility_score": "Fragility",
    "variance": "Variance"
}
ELOS = list(range(300, 2800, 100))
ELO_DEV = 300
WINRATE_THRESHOLD = 0.01
MIN_SAMPLES_PER_GROUP = 20
