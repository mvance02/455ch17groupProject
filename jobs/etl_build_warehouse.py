import pandas as pd
from datetime import datetime

from config import SUPABASE_DB_URL
from utils_pg import pg_conn


def build_modeling_table() -> int:
    # Extract and join operational data into one row per order
    sql = """
      SELECT
        o.order_id,
        o.customer_id,
        o.order_datetime,
        o.order_total,
        o.shipping_fee,
        o.promo_used,
        c.birthdate,
        s.late_delivery
      FROM orders o
      JOIN customers c ON c.customer_id = o.customer_id
      JOIN shipments s ON s.order_id = o.order_id;
    """
    with pg_conn(SUPABASE_DB_URL) as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        return 0

    # Parse datetimes
    df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce")

    # Feature engineering
    now_year = datetime.now().year
    df["customer_age_years"] = now_year - df["birthdate"].dt.year
    df["order_dow"] = df["order_datetime"].dt.dayofweek
    df["order_month"] = df["order_datetime"].dt.month

    modeling_cols = [
        "order_id",
        "customer_id",
        "order_total",
        "shipping_fee",
        "promo_used",
        "customer_age_years",
        "order_dow",
        "order_month",
        "late_delivery",
    ]
    df_model = df[modeling_cols].dropna(subset=["late_delivery"])

    # Load into the warehouse table (replace each run)
    with pg_conn(SUPABASE_DB_URL) as conn:
        rows = [
            (
                int(r.order_id),
                int(r.customer_id),
                float(r.order_total),
                float(r.shipping_fee),
                bool(r.promo_used),
                int(r.customer_age_years),
                int(r.order_dow),
                int(r.order_month),
                bool(r.late_delivery),
            )
            for r in df_model.itertuples(index=False)
        ]
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE fact_orders_ml;")
            cur.executemany(
                """
                INSERT INTO fact_orders_ml (
                  order_id, customer_id, order_total, shipping_fee, promo_used,
                  customer_age_years, order_dow, order_month, late_delivery
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows,
            )
        conn.commit()

    return len(df_model)


if __name__ == "__main__":
    count = build_modeling_table()
    print(f"fact_orders_ml updated. Rows: {count}")

