-- Supabase/Postgres schema mirroring local SQLite `shop.db`
-- Adjust schema/owners as needed in the Supabase SQL editor.

-- Customers
CREATE TABLE IF NOT EXISTS public.customers (
  customer_id      BIGSERIAL PRIMARY KEY,
  full_name        TEXT NOT NULL,
  email            TEXT NOT NULL UNIQUE,
  gender           TEXT NOT NULL,
  birthdate        DATE NOT NULL,
  created_at       TIMESTAMPTZ NOT NULL,
  city             TEXT,
  state            TEXT,
  zip_code         TEXT,
  customer_segment TEXT,
  loyalty_tier     TEXT,
  is_active        BOOLEAN NOT NULL DEFAULT TRUE
);

-- Products
CREATE TABLE IF NOT EXISTS public.products (
  product_id   BIGSERIAL PRIMARY KEY,
  sku          TEXT NOT NULL UNIQUE,
  product_name TEXT NOT NULL,
  category     TEXT NOT NULL,
  price        DOUBLE PRECISION NOT NULL,
  cost         DOUBLE PRECISION NOT NULL,
  is_active    BOOLEAN NOT NULL DEFAULT TRUE
);

-- Orders
CREATE TABLE IF NOT EXISTS public.orders (
  order_id       BIGSERIAL PRIMARY KEY,
  customer_id    BIGINT NOT NULL REFERENCES public.customers(customer_id),
  order_datetime TIMESTAMPTZ NOT NULL,
  billing_zip    TEXT,
  shipping_zip   TEXT,
  shipping_state TEXT,
  payment_method TEXT NOT NULL,
  device_type    TEXT NOT NULL,
  ip_country     TEXT NOT NULL,
  promo_used     BOOLEAN NOT NULL DEFAULT FALSE,
  promo_code     TEXT,
  order_subtotal DOUBLE PRECISION NOT NULL,
  shipping_fee   DOUBLE PRECISION NOT NULL,
  tax_amount     DOUBLE PRECISION NOT NULL,
  order_total    DOUBLE PRECISION NOT NULL,
  risk_score     DOUBLE PRECISION NOT NULL,
  is_fraud       BOOLEAN NOT NULL DEFAULT FALSE
);

-- Order items
CREATE TABLE IF NOT EXISTS public.order_items (
  order_item_id BIGSERIAL PRIMARY KEY,
  order_id      BIGINT NOT NULL REFERENCES public.orders(order_id),
  product_id    BIGINT NOT NULL REFERENCES public.products(product_id),
  quantity      INTEGER NOT NULL,
  unit_price    DOUBLE PRECISION NOT NULL,
  line_total    DOUBLE PRECISION NOT NULL
);

-- Shipments
CREATE TABLE IF NOT EXISTS public.shipments (
  shipment_id     BIGSERIAL PRIMARY KEY,
  order_id        BIGINT NOT NULL UNIQUE REFERENCES public.orders(order_id),
  ship_datetime   TIMESTAMPTZ NOT NULL,
  carrier         TEXT NOT NULL,
  shipping_method TEXT NOT NULL,
  distance_band   TEXT NOT NULL,
  promised_days   INTEGER NOT NULL,
  actual_days     INTEGER NOT NULL,
  late_delivery   BOOLEAN NOT NULL DEFAULT FALSE
);

-- Product reviews
CREATE TABLE IF NOT EXISTS public.product_reviews (
  review_id       BIGSERIAL PRIMARY KEY,
  customer_id     BIGINT NOT NULL REFERENCES public.customers(customer_id),
  product_id      BIGINT NOT NULL REFERENCES public.products(product_id),
  rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  review_datetime TIMESTAMPTZ NOT NULL,
  review_text     TEXT,
  CONSTRAINT uq_product_reviews_customer_product UNIQUE (customer_id, product_id)
);

-- Predictions table created by inference
CREATE TABLE IF NOT EXISTS public.order_predictions (
  order_id                BIGINT PRIMARY KEY REFERENCES public.orders(order_id),
  late_delivery_probability DOUBLE PRECISION NOT NULL,
  predicted_late_delivery   BOOLEAN NOT NULL,
  prediction_timestamp      TIMESTAMPTZ NOT NULL
);

-- Optional: analytics/warehouse table used for training
CREATE TABLE IF NOT EXISTS public.fact_orders_ml (
  order_id           BIGINT PRIMARY KEY,
  customer_id        BIGINT NOT NULL,
  num_items          INTEGER NOT NULL,
  order_total        DOUBLE PRECISION NOT NULL,
  shipping_fee       DOUBLE PRECISION NOT NULL,
  promo_used         BOOLEAN NOT NULL,
  customer_age_years INTEGER NOT NULL,
  order_dow          INTEGER NOT NULL,
  order_month        INTEGER NOT NULL,
  late_delivery      BOOLEAN NOT NULL
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_orders_customer ON public.orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_datetime ON public.orders(order_datetime);
CREATE INDEX IF NOT EXISTS idx_shipments_late ON public.shipments(late_delivery);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON public.order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON public.order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_product_reviews_product ON public.product_reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_product_reviews_customer ON public.product_reviews(customer_id);

