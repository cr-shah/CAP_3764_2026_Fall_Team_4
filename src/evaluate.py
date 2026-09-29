import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from sklearn.metrics import classification_report, roc_auc_score

from src.prepare_data import get_train_test, ROOT

X_train, X_test, y_train, y_test = get_train_test()

# Load one model to start
gb = joblib.load(ROOT / "models" / "gradient_boosting.pkl")

# Predict class labels and probabilities
y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))