# IS455 Chapter 17 — Supabase + Vercel ML Pipeline

This repo implements the Chapter 17 pattern:

**Operational DB (Supabase Postgres)** → **ETL warehouse table** → **train model artifact** → **run inference** → **write predictions back** → **web app reads priority queue**

## 1) Supabase setup

### Create tables

Run the SQL in:

- `supabase_schema.sql`

in the Supabase SQL editor.

### Environment variables (local)

Set these in your shell for the Python jobs:

- `SUPABASE_DB_URL`: Supabase Postgres connection string (use the pooler/SSL string)
- `SUPABASE_URL`: your project URL (for the migration script)
- `SUPABASE_SERVICE_ROLE_KEY`: service role key (for the migration script)

## 2) Seed Supabase from `shop.db`

Install Python deps:

```bash
python -m pip install -r requirements.txt
```

Run the migration:

```bash
python scripts/migrate_sqlite_to_supabase.py
```

## 3) ML pipeline jobs

### ETL (build warehouse table)

Build/refresh `fact_orders_ml` from operational tables:

```bash
python jobs/etl_build_warehouse.py
```

### Train (save artifacts)

Trains a baseline logistic regression pipeline and saves:

- `artifacts/late_delivery_model.sav`
- `artifacts/model_metadata.json`
- `artifacts/metrics.json`

```bash
python jobs/train_model.py
```

### Inference (write predictions back)

Scores **unfulfilled orders** (orders with **no row in `shipments`**) and upserts into:

- `order_predictions`

```bash
python jobs/run_inference.py
```

## 4) Web app (Next.js on Vercel)

The Next.js app lives in `web/` and connects directly to Postgres for SQL + transactions.

### Web env vars

Set:

- `SUPABASE_DB_URL`: same Postgres connection string used by Python

### Run locally

```bash
cd web
npm install
npm run dev
```

Open:

- `/select-customer`
- `/dashboard`
- `/place-order`
- `/orders`
- `/warehouse/priority`
- `/scoring`
- `/debug/schema`

## 5) Manual QA checklist (grading)

- Select customer (`/select-customer`)
- Place order (`/place-order`)
- Confirm order appears in history (`/orders`)
- Run scoring locally (`python jobs/run_inference.py`)
- Confirm the priority queue shows the new order (after refresh) (`/warehouse/priority`)

## Notes / constraints

- The deployed web app does **not** run Python (Vercel). The `/scoring` page is instructional for the `local_manual` inference requirement.

