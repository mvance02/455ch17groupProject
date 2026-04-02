from contextlib import contextmanager
from typing import Iterator

import psycopg


@contextmanager
def pg_conn(db_url: str) -> Iterator[psycopg.Connection]:
    if not db_url:
        raise RuntimeError("Missing SUPABASE_DB_URL.")
    conn = psycopg.connect(db_url, autocommit=False)
    try:
        yield conn
    finally:
        conn.close()

