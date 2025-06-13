FEATURES = ["delta", "fragility_score", "variance"]
FEATURE_LABELS = {
    "delta": "Rating Delta",
    "fragility_score": "Fragility Score",
    "variance": "Variance"
}
ELOS = list(range(800, 2201, 100))
ELO_DEV = 50
MIN_SAMPLES_PER_GROUP = 25
WINRATE_THRESHOLD = 0.03
