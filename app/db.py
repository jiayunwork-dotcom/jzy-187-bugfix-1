"""SQLite persistence layer.

The database file lives on the mounted volume (path from ``TIDE_DB_PATH``,
default ``/data/tide.db``).  Three kinds of state are kept:

* station metadata,
* batches and every received sample (missing samples stored with NULL u/v),
* immutable result versions, one per *distinct* batch applied to a station.

All analysis versions are therefore recoverable after a service restart.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    station_id              TEXT PRIMARY KEY,
    longitude               REAL NOT NULL,
    latitude                REAL NOT NULL,
    sample_interval_seconds REAL NOT NULL,
    deployed_at             TEXT NOT NULL,
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id       TEXT NOT NULL,
    station_id     TEXT NOT NULL REFERENCES stations(station_id),
    received_at    TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    n_samples      INTEGER NOT NULL,
    n_missing      INTEGER NOT NULL,
    PRIMARY KEY (station_id, batch_id)
);

CREATE TABLE IF NOT EXISTS samples (
    station_id TEXT NOT NULL,
    batch_id   TEXT NOT NULL,
    t_hours    REAL NOT NULL,
    u          REAL,
    v          REAL,
    seq        INTEGER NOT NULL,
    PRIMARY KEY (station_id, t_hours)
);
CREATE INDEX IF NOT EXISTS idx_samples_batch
    ON samples(station_id, batch_id, seq);

CREATE TABLE IF NOT EXISTS versions (
    station_id         TEXT NOT NULL,
    version            INTEGER NOT NULL,
    created_at         TEXT NOT NULL,
    triggered_by_batch TEXT NOT NULL,
    n_batches          INTEGER NOT NULL,
    n_samples          INTEGER NOT NULL,
    n_valid            INTEGER NOT NULL,
    record_start_hours REAL,
    record_end_hours   REAL,
    record_span_hours  REAL,
    mean_u             REAL,
    mean_v             REAL,
    mean_speed         REAL,
    mean_direction     REAL,
    result_json        TEXT NOT NULL,
    PRIMARY KEY (station_id, version)
);

CREATE TABLE IF NOT EXISTS version_batches (
    station_id TEXT NOT NULL,
    version    INTEGER NOT NULL,
    batch_id   TEXT NOT NULL,
    PRIMARY KEY (station_id, version, batch_id)
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or os.environ.get("TIDE_DB_PATH", "/data/tide.db")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------
def create_station(conn: sqlite3.Connection, station: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO stations
           (station_id, longitude, latitude, sample_interval_seconds,
            deployed_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            station["station_id"],
            float(station["longitude"]),
            float(station["latitude"]),
            float(station["sample_interval_seconds"]),
            station["deployed_at"],
            utc_now_iso(),
        ),
    )
    conn.commit()


def get_station(conn: sqlite3.Connection, station_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM stations WHERE station_id = ?", (station_id,)
    ).fetchone()


# ---------------------------------------------------------------------------
# Batches / samples
# ---------------------------------------------------------------------------
def get_batch(conn: sqlite3.Connection, station_id: str, batch_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM batches WHERE station_id = ? AND batch_id = ?",
        (station_id, batch_id),
    ).fetchone()


def existing_times(conn: sqlite3.Connection, station_id: str) -> set[float]:
    rows = conn.execute(
        "SELECT t_hours FROM samples WHERE station_id = ?", (station_id,)
    ).fetchall()
    return {r["t_hours"] for r in rows}


def existing_grid_indices(
    conn: sqlite3.Connection, station_id: str, interval_seconds: float
) -> set[int]:
    rows = conn.execute(
        "SELECT t_hours FROM samples WHERE station_id = ?", (station_id,)
    ).fetchall()
    hours_per_step = interval_seconds / 3600.0
    return {round(r["t_hours"] / hours_per_step) for r in rows}


def insert_batch(
    conn: sqlite3.Connection,
    station_id: str,
    batch_id: str,
    content_sha256: str,
    rows: Iterable[tuple[float, float | None, float | None, int]],
    n_samples: int,
    n_missing: int,
) -> None:
    """Insert batch header and sample rows. Caller manages the transaction."""
    conn.execute(
        """INSERT INTO batches
           (batch_id, station_id, received_at, content_sha256, n_samples, n_missing)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (batch_id, station_id, utc_now_iso(), content_sha256, n_samples, n_missing),
    )
    conn.executemany(
        """INSERT INTO samples (station_id, batch_id, t_hours, u, v, seq)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(station_id, batch_id, t, u, v, seq) for (t, u, v, seq) in rows],
    )


def load_samples(
    conn: sqlite3.Connection, station_id: str
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Return t_hours, u, v (missing -> NaN) and batch ids, sorted by time."""
    cur = conn.execute(
        """SELECT t_hours, u, v, batch_id FROM samples
           WHERE station_id = ? ORDER BY t_hours""",
        (station_id,),
    )
    t: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    bids: list[str] = []
    nan = float("nan")
    for r in cur:
        t.append(r["t_hours"])
        us.append(nan if r["u"] is None else r["u"])
        vs.append(nan if r["v"] is None else r["v"])
        bids.append(r["batch_id"])
    return t, us, vs, bids


def list_batches(conn: sqlite3.Connection, station_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM batches WHERE station_id = ? ORDER BY received_at",
        (station_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
def next_version(conn: sqlite3.Connection, station_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM versions WHERE station_id = ?",
        (station_id,),
    ).fetchone()
    return int(row["v"])


def save_version(
    conn: sqlite3.Connection,
    station_id: str,
    version: int,
    triggered_by_batch: str,
    batch_ids: list[str],
    summary: dict[str, Any],
    result_json: str,
) -> None:
    conn.execute(
        """INSERT INTO versions
           (station_id, version, created_at, triggered_by_batch, n_batches,
            n_samples, n_valid, record_start_hours, record_end_hours,
            record_span_hours, mean_u, mean_v, mean_speed, mean_direction,
            result_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            station_id,
            version,
            utc_now_iso(),
            triggered_by_batch,
            len(batch_ids),
            summary["n_samples"],
            summary["n_valid"],
            summary["record_start_hours"],
            summary["record_end_hours"],
            summary["record_span_hours"],
            summary["mean_u"],
            summary["mean_v"],
            summary["mean_speed"],
            summary["mean_direction"],
            result_json,
        ),
    )
    conn.executemany(
        "INSERT INTO version_batches (station_id, version, batch_id) VALUES (?, ?, ?)",
        [(station_id, version, b) for b in batch_ids],
    )


def list_versions(conn: sqlite3.Connection, station_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM versions WHERE station_id = ? ORDER BY version",
        (station_id,),
    ).fetchall()


def get_version(
    conn: sqlite3.Connection, station_id: str, version: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM versions WHERE station_id = ? AND version = ?",
        (station_id, version),
    ).fetchone()


def dumps_result(obj: Any) -> str:
    return json.dumps(obj, allow_nan=False, sort_keys=True)
