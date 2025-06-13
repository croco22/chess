FEATURES = ["delta", "fragility_score", "variance"]
FEATURE_LABELS = {
    "delta": "Rating Delta",
    "fragility_score": "Fragility Score",
    "variance": "Variance"
}
ELOS = list(range(800, 2200, 100))
ELO_DEV = 50
WINRATE_THRESHOLD = 0.01
MIN_SAMPLES_PER_GROUP = 20
