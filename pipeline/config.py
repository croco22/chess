import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

font_path = os.getenv("PT_SANS_FONT")
if font_path and os.path.isfile(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["pdf.fonttype"] = 42  # Embed font as TrueType

FEATURES = ["delta", "fragility_score", "variance"]
FEATURE_LABELS = {
    "delta": "Rating Delta",
    "fragility_score": "Fragility Score",
    "variance": "Variance"
}
ELO_DEV = 100
ELOS = list(range(1000, 1900, ELO_DEV))
MIN_SAMPLES_PER_GROUP = 10
WINRATE_THRESHOLD = 0.01
