from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "./data/scholar_agent.sqlite3"


def load_db_path() -> Path:
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "SHOLAR_DB_PATH":
                return Path(value.strip()).expanduser().resolve()
    return Path(os.getenv("SHOLAR_DB_PATH", DEFAULT_DB_PATH)).expanduser().resolve()


def main() -> None:
    db_path = load_db_path()
    print(f"Database: {db_path}")

    if not db_path.exists():
        print("Database file does not exist.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        if not tables:
            print("No tables found.")
            return

        for table in tables:
            table_name = table["name"]
            rows = conn.execute(f'SELECT * FROM "{table_name}"').fetchall()
            print(f"\n=== {table_name} ({len(rows)} rows) ===")
            for row in rows:
                print(json.dumps(dict(row), ensure_ascii=False, indent=2, default=str))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
