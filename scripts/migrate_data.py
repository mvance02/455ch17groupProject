#!/usr/bin/env python3
"""Migrate SQL data files to Supabase using the exec_sql RPC function."""

import os
import sys
import time
from supabase import create_client

SUPABASE_URL = "https://npandpamyupsinjarame.supabase.co"
# Using the anon key - the exec_sql function is SECURITY DEFINER so it runs as postgres
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wYW5kcGFteXVwc2luamFyYW1lIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ1NjA5NjQsImV4cCI6MjA5MDEzNjk2NH0.opuPPLLLlqmfc1IUxkgJXQKYkYwRquPsCIGPNpzdDIA"

SQL_DIR = "/Users/matthewvance/Desktop/455groupProject/scripts/sql_chunks"

# Files to execute in order
FILE_GROUPS: list[tuple[str, int]] = [
    ("orders", 20),
    ("order_items", 61),
    ("shipments", 20),
    ("product_reviews", 12),
]


def main() -> None:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    total_files = sum(count for _, count in FILE_GROUPS)
    completed = 0

    for prefix, count in FILE_GROUPS:
        print(f"\n=== Processing {prefix} ({count} files) ===")
        for i in range(count):
            fname = f"{prefix}_{i}.sql"
            fpath = os.path.join(SQL_DIR, fname)

            if not os.path.exists(fpath):
                print(f"  MISSING: {fname}")
                continue

            with open(fpath, "r") as f:
                sql = f.read().strip()

            if not sql:
                print(f"  EMPTY: {fname}")
                continue

            try:
                result = client.rpc("exec_sql", {"query": sql}).execute()
                completed += 1
                print(f"  [{completed}/{total_files}] {fname} - OK")
            except Exception as e:
                print(f"  [{completed}/{total_files}] {fname} - ERROR: {e}")
                # Continue with next file instead of stopping
                completed += 1

            # Small delay to avoid rate limiting
            if completed % 10 == 0:
                time.sleep(0.5)

    print(f"\n=== Migration complete: {completed}/{total_files} files processed ===")


if __name__ == "__main__":
    main()
