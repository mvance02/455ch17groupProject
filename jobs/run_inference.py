from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import joblib

from config import SUPABASE_DB_URL, MODEL_PATH
from utils_pg import pg_conn


def run_inference() -> int:
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Unfulfilled orders: those with no shipment row yet
    sql = """
      SELECT
        o.order_id,
        o.order_datetime,
        o.order_total,
        o.shipping_fee,
        o.promo_used,
        c.birthdate
      FROM orders o
      JOIN customers c ON c.customer_id = o.customer_id
      LEFT JOIN shipments s ON s.order_id = o.order_id
      WHERE s.order_id IS NULL;
    """
    with pg_conn(SUPABASE_DB_URL) as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        print("No unfulfilled orders found for scoring.")
        return 0

    df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")

    now_year = datetime.now().year
    df["customer_age_years"] = now_year - df["birthdate"].dt.year
    df["order_dow"] = df["order_datetime"].dt.dayofweek
    df["order_month"] = df["order_datetime"].dt.month

    X_live = df[
        [
            "order_total",
            "shipping_fee",
            "promo_used",
            "customer_age_years",
            "order_dow",
            "order_month",
        ]
    ].copy()
    X_live["promo_used"] = X_live["promo_used"].astype(int)

    probs = model.predict_proba(X_live)[:, 1]
    preds = model.predict(X_live)
    ts = datetime.utcnow().isoformat()

    out_rows: List[Dict[str, Any]] = []
    for order_id, p, yhat in zip(df["order_id"], probs, preds):
        out_rows.append(
            {
                "order_id": int(order_id),
                "late_delivery_probability": float(p),
                "predicted_late_delivery": bool(yhat),
                "prediction_timestamp": ts,
            }
        )

    tuples = [
        (
            r["order_id"],
            r["late_delivery_probability"],
            r["predicted_late_delivery"],
            r["prediction_timestamp"],
        )
        for r in out_rows
    ]

    with pg_conn(SUPABASE_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO order_predictions (
                  order_id, late_delivery_probability, predicted_late_delivery, prediction_timestamp
                ) VALUES (%s,%s,%s,%s)
                ON CONFLICT (order_id) DO UPDATE SET
                  late_delivery_probability = EXCLUDED.late_delivery_probability,
                  predicted_late_delivery = EXCLUDED.predicted_late_delivery,
                  prediction_timestamp = EXCLUDED.prediction_timestamp
                """,
                tuples,
            )
        conn.commit()
    written = len(tuples)
    print(f"Inference complete. Predictions written: {written}")
    return written


if __name__ == "__main__":
    run_inference()

