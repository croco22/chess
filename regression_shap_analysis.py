import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("data/stats_dataset.parquet")

X = df.drop(columns=["winrate", "is_best"]).select_dtypes(include='number')
y = df["is_best"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- Evaluation ---")
print(classification_report(y_test, y_pred))

print("--- Coefficients ---")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature:15s} {coef: .4f}")

# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary plot (bar)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Summary plot (beeswarm)
shap.summary_plot(shap_values, X_test)

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
