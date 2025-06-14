FEATURES = ["delta", "fragility_score", "variance"]
FEATURE_LABELS = {
    "delta": "Rating Delta",
    "fragility_score": "Fragility Score",
    "variance": "Variance"
}
ELO_DEV = 100
ELOS = list(range(1000, 1900, ELO_DEV))
MIN_SAMPLES_PER_GROUP = 15
WINRATE_THRESHOLD = 0.03
