"""
Train a logistic regression fraud detection model using only numeric features.
Saves means, scales, coefs, and intercept to artifacts/is_fraud_lr_coefficients.json
so they can be hardcoded in the TypeScript place-order server action.
"""

import json
from datetime import datetime, timezone

import psycopg2
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

from config import SUPABASE_DB_URL, ARTIFACTS_DIR

FEATURE_COLS = [
    "order_total",
    "shipping_fee",
    "tax_amount",
    "risk_score",
    "customer_age_years",
    "order_hour",
    "order_dow",
    "order_month",
    "item_count",
    "avg_unit_price",
    "zip_mismatch",
    "promo_used",
]

QUERY = """
SELECT
    o.order_total::float8                                                        AS order_total,
    o.shipping_fee::float8                                                       AS shipping_fee,
    o.tax_amount::float8                                                         AS tax_amount,
    o.risk_score::float8                                                         AS risk_score,
    EXTRACT(YEAR FROM AGE(o.order_datetime::date, c.birthdate))::float8          AS customer_age_years,
    EXTRACT(HOUR FROM o.order_datetime)::float8                                  AS order_hour,
    EXTRACT(DOW  FROM o.order_datetime)::float8                                  AS order_dow,
    EXTRACT(MONTH FROM o.order_datetime)::float8                                 AS order_month,
    COUNT(oi.order_item_id)::float8                                              AS item_count,
    (SUM(oi.line_total) / NULLIF(COUNT(oi.order_item_id), 0))::float8           AS avg_unit_price,
    CASE WHEN o.billing_zip IS NOT NULL
              AND o.shipping_zip IS NOT NULL
              AND o.billing_zip <> o.shipping_zip THEN 1.0 ELSE 0.0 END         AS zip_mismatch,
    o.promo_used::int::float8                                                    AS promo_used,
    o.is_fraud::int                                                              AS is_fraud
FROM orders o
JOIN customers   c  ON c.customer_id   = o.customer_id
JOIN order_items oi ON oi.order_id     = o.order_id
GROUP BY
    o.order_id, o.order_total, o.shipping_fee, o.tax_amount, o.risk_score,
    c.birthdate, o.order_datetime, o.billing_zip, o.shipping_zip,
    o.promo_used, o.is_fraud
"""


def fetch_data() -> pd.DataFrame:
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL env var is not set.")
    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        df = pd.read_sql_query(QUERY, conn)
    finally:
        conn.close()
    return df


def train_and_save() -> None:
    print("Fetching data from Supabase...")
    df = fetch_data()
    print(f"  Loaded {len(df):,} rows. Fraud rate: {df['is_fraud'].mean():.2%}")

    X = df[FEATURE_COLS].copy()
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    X_train_imp = imputer.fit_transform(X_train)
    X_train_sc = scaler.fit_transform(X_train_imp)
    clf.fit(X_train_sc, y_train)

    X_test_imp = imputer.transform(X_test)
    X_test_sc = scaler.transform(X_test_imp)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]
    y_pred = clf.predict(X_test_sc)

    auc = roc_auc_score(y_test, y_prob)
    print(f"  Test ROC AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save full pipeline for notebook / offline use
    joblib.dump((imputer, scaler, clf), ARTIFACTS_DIR / "is_fraud_lr_pipeline.sav")

    # Save coefficients in the same shape as the late-delivery MODEL constant
    coefficients = {
        "description": "LR fraud model — numeric features only. Trained against Supabase orders.",
        "features": FEATURE_COLS,
        "means": imputer.statistics_.tolist(),
        "scales": scaler.scale_.tolist(),
        "coefs": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "test_roc_auc": round(auc, 4),
        "fraud_rate": round(float(df["is_fraud"].mean()), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    coef_path = ARTIFACTS_DIR / "is_fraud_lr_coefficients.json"
    with open(coef_path, "w", encoding="utf-8") as f:
        json.dump(coefficients, f, indent=2)

    print(f"\nSaved: {coef_path}")
    print("\n--- TypeScript MODEL constant (paste into place-order/actions.ts) ---")
    print(f'  means:     {coefficients["means"]},')
    print(f'  scales:    {coefficients["scales"]},')
    print(f'  coefs:     {coefficients["coefs"]},')
    print(f'  intercept: {coefficients["intercept"]},')


if __name__ == "__main__":
    train_and_save()
