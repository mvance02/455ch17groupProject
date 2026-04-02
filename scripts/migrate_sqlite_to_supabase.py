import os
import sqlite3
from typing import Any, Dict, List

import requests


SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
    "SUPABASE_ANON_KEY", ""
)

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "shop.db")


def _post(table: str, rows: List[Dict[str, Any]]) -> None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_*KEY env vars must be set before running migration."
        )
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    # Chunk inserts to avoid large payloads
    import json

    batch_size = 500
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        resp = requests.post(url, headers=headers, data=json.dumps(chunk), timeout=30)
        resp.raise_for_status()


def migrate_customers(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT * FROM customers")
    cols = [c[0] for c in cur.description]
    rows = []
    for r in cur.fetchall():
        rec = dict(zip(cols, r))
        rows.append(
            {
                "customer_id": rec["customer_id"],
                "full_name": rec["full_name"],
                "email": rec["email"],
                "gender": rec["gender"],
                "birthdate": rec["birthdate"],
                "created_at": rec["created_at"],
                "city": rec["city"],
                "state": rec["state"],
                "zip_code": rec["zip_code"],
                "customer_segment": rec["customer_segment"],
                "loyalty_tier": rec["loyalty_tier"],
                "is_active": bool(rec["is_active"]),
            }
        )
    _post("customers", rows)


def migrate_products(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT * FROM products")
    cols = [c[0] for c in cur.description]
    rows = []
    for r in cur.fetchall():
        rec = dict(zip(cols, r))
        rows.append(
            {
                "product_id": rec["product_id"],
                "sku": rec["sku"],
                "product_name": rec["product_name"],
                "category": rec["category"],
                "price": rec["price"],
                "cost": rec["cost"],
                "is_active": bool(rec["is_active"]),
            }
        )
    _post("products", rows)


def migrate_orders(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT * FROM orders")
    cols = [c[0] for c in cur.description]
    rows = []
    for r in cur.fetchall():
        rec = dict(zip(cols, r))
        rows.append(
            {
                "order_id": rec["order_id"],
                "customer_id": rec["customer_id"],
                "order_datetime": rec["order_datetime"],
                "billing_zip": rec["billing_zip"],
                "shipping_zip": rec["shipping_zip"],
                "shipping_state": rec["shipping_state"],
                "payment_method": rec["payment_method"],
                "device_type": rec["device_type"],
                "ip_country": rec["ip_country"],
                "promo_used": bool(rec["promo_used"]),
                "promo_code": rec["promo_code"],
                "order_subtotal": rec["order_subtotal"],
                "shipping_fee": rec["shipping_fee"],
                "tax_amount": rec["tax_amount"],
                "order_total": rec["order_total"],
                "risk_score": rec["risk_score"],
                "is_fraud": bool(rec["is_fraud"]),
            }
        )
    _post("orders", rows)


def migrate_order_items(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT * FROM order_items")
    cols = [c[0] for c in cur.description]
    rows = []
    for r in cur.fetchall():
        rec = dict(zip(cols, r))
        rows.append(
            {
                "order_item_id": rec["order_item_id"],
                "order_id": rec["order_id"],
                "product_id": rec["product_id"],
                "quantity": rec["quantity"],
                "unit_price": rec["unit_price"],
                "line_total": rec["line_total"],
            }
        )
    _post("order_items", rows)


def migrate_shipments(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT * FROM shipments")
    cols = [c[0] for c in cur.description]
    rows = []
    for r in cur.fetchall():
        rec = dict(zip(cols, r))
        rows.append(
            {
                "shipment_id": rec["shipment_id"],
                "order_id": rec["order_id"],
                "ship_datetime": rec["ship_datetime"],
                "carrier": rec["carrier"],
                "shipping_method": rec["shipping_method"],
                "distance_band": rec["distance_band"],
                "promised_days": rec["promised_days"],
                "actual_days": rec["actual_days"],
                "late_delivery": bool(rec["late_delivery"]),
            }
        )
    _post("shipments", rows)


def migrate_product_reviews(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT * FROM product_reviews")
    cols = [c[0] for c in cur.description]
    rows = []
    for r in cur.fetchall():
        rec = dict(zip(cols, r))
        rows.append(
            {
                "review_id": rec["review_id"],
                "customer_id": rec["customer_id"],
                "product_id": rec["product_id"],
                "rating": rec["rating"],
                "review_datetime": rec["review_datetime"],
                "review_text": rec["review_text"],
            }
        )
    _post("product_reviews", rows)


def main() -> None:
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(f"SQLite file not found at {SQLITE_PATH}")

    conn = sqlite3.connect(SQLITE_PATH)
    try:
        cur = conn.cursor()
        migrate_customers(cur)
        migrate_products(cur)
        migrate_orders(cur)
        migrate_order_items(cur)
        migrate_shipments(cur)
        migrate_product_reviews(cur)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

