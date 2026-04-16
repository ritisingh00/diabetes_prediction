import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("diabetes_prediction.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Define Models (with scaling pipelines where needed) ───────────────────────
models = {
    "rf":  RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "lr":  Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42))
    ]),
}

# ── Train, Evaluate & Save ────────────────────────────────────────────────────
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, preds)           * 100, 2),
        "Precision": round(precision_score(y_test, preds)          * 100, 2),
        "Recall":    round(recall_score(y_test, preds)             * 100, 2),
        "F1 Score":  round(f1_score(y_test, preds)                 * 100, 2),
        "ROC AUC":   round(roc_auc_score(y_test, probas)           * 100, 2),
    }

    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✅ {name.upper()} saved  →  Accuracy: {results[name]['Accuracy']}%")

# Save metrics for the app to load
with open("model_metrics.pkl", "wb") as f:
    pickle.dump(results, f)

# Save test set for ROC curve in app
with open("test_data.pkl", "wb") as f:
    pickle.dump((X_test, y_test), f)

print("\n🎉 All models trained and saved!")
print(pd.DataFrame(results).T.to_string())