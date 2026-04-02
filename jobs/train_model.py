import json
from datetime import datetime
from typing import Any, Dict, List

import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

from config import SUPABASE_URL, SUPABASE_KEY, ARTIFACTS_DIR, MODEL_PATH, MODEL_METADATA_PATH, METRICS_PATH


def _select_fact_orders_ml() -> List[Dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_*KEY must be set.")
    url = f"{SUPABASE_URL}/rest/v1/fact_orders_ml?select=*"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def train_and_save() -> None:
    rows = _select_fact_orders_ml()
    if not rows:
        raise RuntimeError("fact_orders_ml is empty; run ETL first.")

    df = pd.DataFrame(rows)

    label_col = "late_delivery"
    feature_cols = [
        "order_total",
        "shipping_fee",
        "promo_used",
        "customer_age_years",
        "order_dow",
        "order_month",
    ]

    X = df[feature_cols].copy()
    # Boolean fields may come back as true/false already
    X["promo_used"] = X["promo_used"].astype(int)
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "row_count_train": int(len(X_train)),
        "row_count_test": int(len(X_test)),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_name": "late_delivery_pipeline",
        "model_version": "1.0.0",
        "trained_at_utc": datetime.utcnow().isoformat(),
        "warehouse_table": "fact_orders_ml",
        "num_training_rows": int(len(X_train)),
        "num_test_rows": int(len(X_test)),
        "features": feature_cols,
        "label": label_col,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Metrics: {metrics}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {MODEL_METADATA_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    train_and_save()

