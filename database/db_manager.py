"""SQLite database manager for EcoSort AI.

Logs classification results with timestamps and provides
query helpers for the history dashboard.
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DB_DIR, "classifications.db")


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database, creating it if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the classifications table if it does not already exist."""
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS classifications (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            image_name      TEXT,
            predicted_label TEXT    NOT NULL,
            waste_category  TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            bin_color       TEXT,
            carbon_saved_kg REAL    DEFAULT 0.0
        )
        """
    )
    conn.commit()
    conn.close()


def log_classification(
    image_name: str,
    predicted_label: str,
    waste_category: str,
    confidence: float,
    bin_color: str,
    carbon_saved_kg: float = 0.0,
) -> None:
    """Insert a new classification record."""
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO classifications
            (timestamp, image_name, predicted_label, waste_category,
             confidence, bin_color, carbon_saved_kg)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            image_name,
            predicted_label,
            waste_category,
            confidence,
            bin_color,
            carbon_saved_kg,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_classifications(limit: int = 20) -> list[dict]:
    """Return the most recent classification records."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM classifications ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_total_carbon_saved() -> float:
    """Return the sum of all carbon_saved_kg entries."""
    conn = _get_connection()
    result = conn.execute(
        "SELECT COALESCE(SUM(carbon_saved_kg), 0) FROM classifications"
    ).fetchone()
    conn.close()
    return float(result[0])


def get_category_counts() -> dict:
    """Return a dict of {waste_category: count}."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT waste_category, COUNT(*) as cnt "
        "FROM classifications GROUP BY waste_category"
    ).fetchall()
    conn.close()
    return {row["waste_category"]: row["cnt"] for row in rows}


def get_total_classifications() -> int:
    """Return the total number of classification records."""
    conn = _get_connection()
    result = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()
    conn.close()
    return int(result[0])
