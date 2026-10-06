"""Validation and analysis pipeline.

Every new batch is fully checked (all field problems are reported at once, not
just the first), persisted, and the station is then re-fitted from the complete
historical record.  Fitting from scratch each time is deliberate: see
docs/ALGORITHM.md for the trade-off.  Because the fit is a pure function of the
stored samples, the result is independent of upload order and bit-for-bit
reproducible across restarts.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import db
from .harmonics import (
    CONSTITUENT_PRIORITY,
    MEAN_NAME,
    ellipse_from_coefficients,
    least_squares_normal_equations,
    select_constituents,
)

# Timestamp must sit on the station sampling grid this tightly.
_REL_TOL = 1e-9
_ABS_TOL_SECONDS = 1e-3


class ValidationError(Exception):
    """422-style error carrying every offending field."""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("validation failed")
        self.errors = errors


class ConflictError(Exception):
    """A batch_id already exists for the station with different content."""

    def __init__(self, message: str, field: str = "batch_id"):
        super().__init__(message)
        self.message = message
        self.field = field


def parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; a trailing Z means UTC; naive means UTC."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def canonical_payload(batch_id: str, samples: list[dict[str, Any]]) -> bytes:
    """Deterministic serialization used for duplicate detection."""
    norm = {
        "batch_id": batch_id,
        "samples": sorted(
            (
                {"time": s["time"], "u": s["u"], "v": s["v"]}
                for s in samples
            ),
            key=lambda r: r["time"],
        ),
    }
    return json.dumps(norm, sort_keys=True, allow_nan=True).encode("utf-8")


def content_hash(batch_id: str, samples: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_payload(batch_id, samples)).hexdigest()


def validate_batch(
    station: sqlite3.Row,
    batch_id: str,
    samples: list[dict[str, Any]],
    used_grid: set[int],
) -> tuple[list[tuple[float, float | None, float | None]], int]:
    """Validate one batch against the station definition.

    Returns (rows, n_missing) where rows = (t_hours, u|None, v|None) sorted by
    time and referenced to the deployment instant.  Raises ValidationError with
    all collected field problems.
    """
    errors: list[dict[str, str]] = []
    deployed = parse_iso8601(station["deployed_at"])
    interval = float(station["sample_interval_seconds"])

    if not isinstance(batch_id, str) or not batch_id.strip():
        errors.append({"field": "batch_id", "message": "batch_id is required"})

    if not isinstance(samples, list) or len(samples) == 0:
        raise ValidationError(
            [{"field": "samples", "message": "samples must be a non-empty list"}]
        )

    parsed: list[tuple[datetime, float | None, float | None, int, int]] = []
    seen: dict[int, int] = {}
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append({"field": f"samples[{i}]", "message": "sample must be an object"})
            continue

        raw_time = sample.get("time")
        dt: datetime | None = None
        try:
            dt = parse_iso8601(raw_time)
        except (ValueError, TypeError):
            errors.append(
                {"field": f"samples[{i}].time",
                 "message": f"invalid ISO-8601 timestamp: {raw_time!r}"}
            )

        u = sample.get("u")
        v = sample.get("v")
        u_val: float | None = None
        v_val: float | None = None
        if u is not None:
            if not _is_number(u) or not math.isfinite(float(u)):
                errors.append(
                    {"field": f"samples[{i}].u",
                     "message": f"u must be a finite number, got {u!r}"}
                )
            else:
                u_val = float(u)
        if v is not None:
            if not _is_number(v) or not math.isfinite(float(v)):
                errors.append(
                    {"field": f"samples[{i}].v",
                     "message": f"v must be a finite number, got {v!r}"}
                )
            else:
                v_val = float(v)

        if dt is not None:
            if dt < deployed:
                errors.append(
                    {"field": f"samples[{i}].time",
                     "message": (
                         f"sample time {raw_time} is earlier than deployment "
                         f"time {station['deployed_at']}")}
                )
            # Grid alignment with the deployment instant.
            elapsed = (dt - deployed).total_seconds()
            nearest = round(elapsed / interval)
            resid = elapsed - nearest * interval
            if (
                nearest < 0
                or abs(resid) > max(_ABS_TOL_SECONDS, _REL_TOL * abs(nearest * interval))
            ):
                errors.append(
                    {"field": f"samples[{i}].time",
                     "message": (
                         f"time {raw_time} is not on the station sampling grid "
                         f"(interval {interval:g}s from deployment)")}
                )
            else:
                if nearest in seen:
                    errors.append(
                        {"field": f"samples[{i}].time",
                         "message": (
                             f"duplicate timestamp {raw_time} at "
                             f"samples[{seen[nearest]}] and samples[{i}]")}
                    )
                else:
                    seen[nearest] = i
                if nearest in used_grid:
                    errors.append(
                        {"field": f"samples[{i}].time",
                         "message": f"timestamp {raw_time} already exists in another batch"}
                    )
            parsed.append((dt, u_val, v_val, i, nearest))

    if errors:
        raise ValidationError(errors)

    # All timestamps are on-grid by construction; sorting makes the stored
    # order canonical.  A gap of more than one step is simply a missing
    # segment (samples for the skipped grid points may arrive in another
    # batch, or be reported as null within one).
    parsed.sort(key=lambda r: r[4])
    rows: list[tuple[float, float | None, float | None]] = []
    n_missing = 0
    for dt, u_val, v_val, i, grid in parsed:
        t_hours = grid * interval / 3600.0
        if u_val is None or v_val is None:
            n_missing += 1
        rows.append((t_hours, u_val, v_val))

    return rows, n_missing


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_station(
    t_hours: list[float], u: list[float], v: list[float]
) -> dict[str, Any]:
    """Run the full harmonic analysis on the station's complete record."""
    t = np.asarray(t_hours, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)

    finite = np.isfinite(t) & np.isfinite(uu) & np.isfinite(vv)
    n_valid = int(finite.sum())

    if n_valid > 0:
        span_hours = float(t[finite].max() - t[finite].min())
        start_hours = float(t[finite].min())
        end_hours = float(t[finite].max())
    else:
        span_hours = 0.0
        start_hours = None
        end_hours = None

    accepted, rejected = select_constituents(span_hours)

    mean = {"u": 0.0, "v": 0.0}
    included_out: list[dict[str, Any]] = []

    if n_valid > 0:
        x, _ = least_squares_normal_equations(t, uu, vv, accepted)
        mean["u"] = float(x[0, 0])
        mean["v"] = float(x[0, 1])

        for k, sel in enumerate(accepted[1:], start=1):
            au, bu = float(x[2 * k - 1, 0]), float(x[2 * k, 0])
            av, bv = float(x[2 * k - 1, 1]), float(x[2 * k, 1])
            ell = ellipse_from_coefficients(au, bu, av, bv)
            included_out.append(
                {
                    "name": sel.name,
                    "frequency_rad_per_hour": sel.frequency,
                    "u": {"amplitude": ell.u_amplitude, "phase": ell.u_phase},
                    "v": {"amplitude": ell.v_amplitude, "phase": ell.v_phase},
                    "ellipse": {
                        "semi_major": ell.semi_major,
                        "semi_minor": ell.semi_minor,
                        "inclination": ell.inclination,
                        "inclination_deg": math.degrees(ell.inclination),
                        "phase": ell.phase,
                    },
                }
            )

    accepted_names = {s.name for s in accepted}
    # Constituents never reached by the greedy loop also get a rejection entry
    # (e.g. a record too short to evaluate anything beyond the first blocker).
    excluded_out: list[dict[str, Any]] = [
        {
            "name": r.name,
            "masked_by": r.masked_by,
            "required_hours": r.required_hours,
            "missing_hours": max(0.0, r.required_hours - r.current_hours),
        }
        for r in rejected
    ]
    for name in CONSTITUENT_PRIORITY:
        if name not in accepted_names and not any(
            e["name"] == name for e in excluded_out
        ):
            # Cannot occur with the current greedy loop, kept for completeness.
            excluded_out.append(
                {
                    "name": name,
                    "masked_by": MEAN_NAME,
                    "required_hours": 0.0,
                    "missing_hours": 0.0,
                }
            )

    speed = math.hypot(mean["u"], mean["v"])
    direction = math.atan2(mean["v"], mean["u"]) if speed > 0 else 0.0

    return {
        "n_samples": int(t.shape[0]),
        "n_valid": n_valid,
        "record_start_hours": start_hours,
        "record_end_hours": end_hours,
        "record_span_hours": span_hours,
        "mean": {
            "u": mean["u"],
            "v": mean["v"],
            "speed": speed,
            "direction_rad": direction,
            "direction_deg": math.degrees(direction),
        },
        "included": included_out,
        "excluded": excluded_out,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def submit_batch(
    conn: sqlite3.Connection,
    station_id: str,
    batch_id: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate, persist and refit.  Idempotent on identical re-upload."""
    station = db.get_station(conn, station_id)
    if station is None:
        raise ValidationError(
            [{"field": "station_id", "message": f"station {station_id!r} does not exist"}]
        )

    existing = db.get_batch(conn, station_id, batch_id)
    if existing is not None:
        digest = content_hash(batch_id if isinstance(batch_id, str) else "", samples)
        if existing["content_sha256"] != digest:
            raise ConflictError(
                f"batch {batch_id!r} already exists for station {station_id!r} "
                "with different content"
            )
        # Identical upload: no new version, return the current latest version.
        return latest_result(conn, station_id, replayed=True, batch_id=batch_id)

    used_grid = db.existing_grid_indices(conn, station_id, float(station["sample_interval_seconds"]))
    rows, n_missing = validate_batch(station, batch_id, samples, used_grid)
    digest = content_hash(batch_id, samples)

    try:
        with conn:  # single transaction: batch rows + new version commit together
            db.insert_batch(
                conn,
                station_id,
                batch_id,
                digest,
                [(t, u, v, seq) for seq, (t, u, v) in enumerate(rows)],
                n_samples=len(rows),
                n_missing=n_missing,
            )
            result = _refit_and_version(conn, station, batch_id)
    except sqlite3.IntegrityError as exc:
        raise ConflictError(f"batch conflict: {exc}") from exc

    result["replayed"] = False
    return result


def _refit_and_version(
    conn: sqlite3.Connection, station: sqlite3.Row, triggered_by_batch: str
) -> dict[str, Any]:
    station_id = station["station_id"]
    t, u, v, _bids = db.load_samples(conn, station_id)
    fit = fit_station(t, u, v)

    batches = [r["batch_id"] for r in db.list_batches(conn, station_id)]
    deployed = parse_iso8601(station["deployed_at"])

    def hours_to_iso(h: float | None) -> str | None:
        if h is None:
            return None
        ts = deployed.timestamp() + h * 3600.0
        return (
            datetime.fromtimestamp(ts, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

    version_no = db.next_version(conn, station_id)
    result = {
        "station_id": station_id,
        "version": version_no,
        "triggered_by_batch": triggered_by_batch,
        "created_at": db.utc_now_iso(),
        "batches": sorted(batches),
        "record": {
            "n_samples": fit["n_samples"],
            "n_valid": fit["n_valid"],
            "n_missing": fit["n_samples"] - fit["n_valid"],
            "start_time": hours_to_iso(fit["record_start_hours"]),
            "end_time": hours_to_iso(fit["record_end_hours"]),
            "span_hours": fit["record_span_hours"],
        },
        "mean": fit["mean"],
        "constituents": fit["included"],
        "excluded": fit["excluded"],
    }

    summary = {
        "n_samples": fit["n_samples"],
        "n_valid": fit["n_valid"],
        "record_start_hours": fit["record_start_hours"],
        "record_end_hours": fit["record_end_hours"],
        "record_span_hours": fit["record_span_hours"],
        "mean_u": fit["mean"]["u"],
        "mean_v": fit["mean"]["v"],
        "mean_speed": fit["mean"]["speed"],
        "mean_direction": fit["mean"]["direction_rad"],
    }
    db.save_version(
        conn, station_id, version_no, triggered_by_batch, sorted(batches),
        summary, db.dumps_result(result),
    )
    return result


def latest_result(
    conn: sqlite3.Connection, station_id: str, replayed: bool = False,
    batch_id: str | None = None,
) -> dict[str, Any]:
    rows = db.list_versions(conn, station_id)
    if not rows:
        return {
            "station_id": station_id,
            "version": 0,
            "triggered_by_batch": batch_id,
            "replayed": replayed,
            "batches": [],
            "record": {
                "n_samples": 0, "n_valid": 0, "n_missing": 0,
                "start_time": None, "end_time": None, "span_hours": 0.0,
            },
            "mean": {"u": 0.0, "v": 0.0, "speed": 0.0,
                     "direction_rad": 0.0, "direction_deg": 0.0},
            "constituents": [],
            "excluded": [],
        }
    result = json.loads(rows[-1]["result_json"])
    result["replayed"] = replayed
    if batch_id is not None:
        result["triggered_by_batch"] = batch_id
    return result
