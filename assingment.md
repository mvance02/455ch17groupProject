Goal
In this chapter, you built a realistic ML pipeline that reads from a live operational database, creates an analytical “warehouse” table for modeling, trains a model, saves a model artifact, generates predictions, and writes those predictions back to the operational database.

In this section, you will vibe code a complete (but simple) web app on top of that database. You will use an AI coding agent (Cursor or Claude Code) to generate most of the application scaffolding. Your job is to (1) provide clear requirements, (2) keep the database contract stable, and (3) test the application until it matches expected behavior.

To keep scope manageable, this app intentionally ignores authentication. Instead, it lets the user select an existing customer to “act as” during testing.

What Your App Must Do
Use an existing SQLite database file named shop.db (operational DB).

Provide a “Select Customer” screen (no signup/login).

Allow placing a new order for the selected customer.

Save the order + line items into shop.db.

Show an order history page for that customer.

Show the warehouse “Late Delivery Priority Queue” page (top 50).

Provide a “Run Scoring” button that triggers the inference job and then refreshes the priority queue.

You will build the app in one of three stacks. The recommended default is JavaScript (Next.js) because it is widely supported by AI coding agents and has a straightforward developer experience.

Database Contract
Your AI agent must not invent new tables. It should only use the operational database tables you already have (for example, customers, orders, order_items, products, and order_predictions). If your database uses different table or column names, update the prompts below to match your schema.

The pipeline writes predictions into order_predictions keyed by order_id. The application should treat that table like any other application table.

Recommended Stack
For students with limited background, use: Next.js + SQLite for the web app, and a separate Python inference script that writes predictions into the database.

The rest of this section provides a complete sequence of copy/paste prompts. Paste them into Cursor (or Claude Code) in order. After each step, run the app and verify behavior before moving on.

Prompt 0: Project Setup (Next.js)
      You are generating a complete student project web app using Next.js (App Router) and SQLite.
      Constraints:
      - No authentication. Users select an existing customer to "act as".
      - Use a SQLite file named "shop.db" located at the project root (or /data/shop.db if you prefer).
      - Use better-sqlite3 for DB access.
      - Keep UI simple and clean.

      Tasks:
      1. Create a new Next.js app (App Router).
      2. Add a server-side DB helper module that opens shop.db and exposes helpers for SELECT and INSERT/UPDATE using prepared statements.
      3. Create a shared layout with navigation links:
        - Select Customer
        - Customer Dashboard
        - Place Order
        - Order History
        - Warehouse Priority Queue
        - Run Scoring
      4. Provide install/run instructions (npm) and any required scripts.

      Return:
      - All files to create/modify
      - Any commands to run
      
Prompt 0.5: Inspect the Database Schema
      Add a developer-only page at /debug/schema that prints:
      - All table names in shop.db
      - For each table, the column names and types (PRAGMA table_info)

      Purpose: Students can verify the real schema and adjust prompts if needed.
      Keep it simple and readable.
      
Prompt 1: Select Customer Screen
      Add a "Select Customer" page at /select-customer.

      Requirements:
      1. Query the database for customers:
        - customer_id
        - first_name
        - last_name
        - email
      2. Render a searchable dropdown or simple list. When a customer is selected, store customer_id in a cookie.
      3. Redirect to /dashboard after selection.
      4. Add a small banner showing the currently selected customer on every page (if set).

      Deliver:
      - Any new routes/components
      - DB query code using better-sqlite3
      - Notes on where customer_id is stored
      
Prompt 2: Customer Dashboard
      Create a /dashboard page that shows a summary for the selected customer.

      Requirements:
      1. If no customer is selected, redirect to /select-customer.
      2. Show:
        - Customer name and email
        - Total number of orders for the customer
        - Total spend across all orders (sum total_value)
        - A small table of the 5 most recent orders (order_id, order_timestamp, fulfilled, total_value)
      3. All data must come from shop.db.

      Deliver:
      - SQL queries used
      - Page UI implementation
      
Prompt 3: Place Order Page
      Create a /place-order page that allows creating a new order for the selected customer.

      Requirements:
      1. If no customer selected, redirect to /select-customer.
      2. Query products (product_id, product_name, price) and let the user add 1+ line items:
        - product
        - quantity
      3. On submit:
        - Insert a row into orders for this customer with fulfilled = 0 and order_timestamp = current time
        - Insert corresponding rows into order_items
        - Compute and store total_value in orders (sum price*quantity)
      4. After placing, redirect to /orders and show a success message.

      Constraints:
      - Use a transaction for inserts.
      - Keep the UI minimal (a table of line items is fine).

      Deliver:
      - SQL inserts
      - Next.js route handlers (server actions or API routes)
      - Any validation rules
      
Prompt 4: Order History Page
      Create a /orders page that shows order history for the selected customer.

      Requirements:
      1. If no customer selected, redirect to /select-customer.
      2. Render a table of the customer's orders:
        - order_id, order_timestamp, fulfilled, total_value
      3. Clicking an order shows /orders/[order_id] with line items:
        - product_name, quantity, unit_price, line_total
      4. Keep it clean and readable.

      Deliver:
      - The two pages
      - SQL queries
      
Prompt 5: Warehouse Priority Queue Page
      Create /warehouse/priority page that shows the "Late Delivery Priority Queue".

      Use this SQL query exactly (adjust table/column names only if they differ in shop.db):

      SELECT
        o.order_id,
        o.order_timestamp,
        o.total_value,
        o.fulfilled,
        c.customer_id,
        c.first_name || ' ' || c.last_name AS customer_name,
        p.late_delivery_probability,
        p.predicted_late_delivery,
        p.prediction_timestamp
      FROM orders o
      JOIN customers c ON c.customer_id = o.customer_id
      JOIN order_predictions p ON p.order_id = o.order_id
      WHERE o.fulfilled = 0
      ORDER BY p.late_delivery_probability DESC, o.order_timestamp ASC
      LIMIT 50;

      Requirements:
      - Render the results in a table.
      - Add a short explanation paragraph describing why this queue exists.

      Deliver:
      - Page code
      
Prompt 6: Run Scoring Button (Triggers Python Inference Job)
To keep the application simple, the web app will not run ML code. Instead, it triggers a Python inference script that writes predictions into order_predictions. The app then reloads the priority queue.

      Add a /scoring page with a "Run Scoring" button.

      Behavior:
      1. When clicked, the server runs:
        python jobs/run_inference.py
      2. The Python script writes predictions into order_predictions keyed by order_id.
      3. The UI shows:
        - Success/failure status
        - How many orders were scored (parse stdout if available)
        - Timestamp

      Constraints:
      - Provide safe execution: timeouts and capture stdout/stderr.
      - The app should not crash if Python fails; show an error message.
      - Do not require Docker.

      Deliver:
      - Next.js route/handler for triggering scoring
      - Implementation details for running Python from Node
      - Any UI components needed
      
Prompt 7: Polishing and Testing Checklist
      Polish the app for student usability and add a testing checklist.

      Tasks:
      1. Add a banner showing which customer is currently selected.
      2. Add basic form validation on /place-order.
      3. Add error handling for missing DB, missing tables, or empty results.
      4. Provide a manual QA checklist:
        - Select customer
        - Place order
        - View orders
        - Run scoring
        - View priority queue with the new order appearing (after scoring)

      Deliver:
      - Final code changes
      - A README.md with setup and run steps
      
Alternative Stack Prompts
If you prefer Python or C#, you can use the prompts below instead. These prompts generate the same app features but with different frameworks.

Option B: Python (FastAPI) Full-App Prompt

      Build a complete student web app using Python FastAPI, Jinja2 templates, and SQLite shop.db (at project root).
      No authentication: users select an existing customer to "act as".

      Pages:
      - GET /select-customer: list/search customers and store customer_id in a cookie
      - GET /dashboard: summary stats for selected customer
      - GET/POST /place-order: select products + quantities and insert orders + order_items
      - GET /orders: order history
      - GET /orders/{order_id}: order details with line items
      - GET /warehouse/priority: priority queue table using order_predictions
      - POST /scoring/run: runs python jobs/run_inference.py and then redirects to /warehouse/priority

      Constraints:
      - Use sqlite3 (no ORM).
      - Use transactions for writes.
      - Provide minimal CSS.
      - Include a README with setup and run instructions (uvicorn).

      Deliver all code files and commands.
      
Option C: ASP.NET/C# Full-App Prompt

      Build a complete student web app using ASP.NET Core and SQLite shop.db (at project root).
      No authentication: users select an existing customer to "act as" and store customer_id in a cookie.

      Pages/Endpoints:
      - /select-customer (GET + POST): choose customer
      - /dashboard (GET): customer summary + recent orders
      - /place-order (GET + POST): create an order and order_items using a DB transaction
      - /orders (GET): order history
      - /orders/{orderId} (GET): order detail with line items
      - /warehouse/priority (GET): late delivery priority queue (join orders/customers/order_predictions)
      - /scoring/run (POST): execute python jobs/run_inference.py and return status

      Constraints:
      - Use Microsoft.Data.Sqlite (no EF required).
      - Render simple HTML (Razor Pages or MVC ok).
      - Provide commands to run (dotnet run) and setup instructions.

      Deliver all code files, NuGet packages, and commands.
      
Key Idea
This app is intentionally simple, but it demonstrates a complete end-to-end pattern: operational data → analytics pipeline → trained model file → automated scoring → operational workflow improvement.