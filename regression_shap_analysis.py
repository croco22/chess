"""
Script to perform logistic regression on move recommendation data and
analyze feature relevance using SHAP (SHapley Additive exPlanations).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_parquet("data/stats_dataset.parquet")

X = df[["delta", "fragility_score", "variance", "played_by", "win_pov", "pair_freq", "winrate_white", "score_test"]].copy()
y = df["is_best"]

# Standardize features for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
model = LogisticRegression(penalty=None, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("--- Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Coefficients
print("\n--- Coefficients ---")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature:15s} {coef: .4f}")

# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------
print("\n--- SHAP Analysis ---")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary plot (bar)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_test)

# Optional: Dependence plot
# shap.dependence_plot("delta", shap_values.values, X_test)

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
