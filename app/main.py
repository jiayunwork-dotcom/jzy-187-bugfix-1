"""FastAPI interface: the only exposed surface of the service."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import Any
from pydantic import BaseModel, Field

from . import db, service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_path = os.environ.get("TIDE_DB_PATH", "/data/tide.db")
    conn = db.connect(app.state.db_path)
    db.init_db(conn)
    conn.close()
    yield


app = FastAPI(
    title="Tidal Harmonic Analysis Service",
    version="1.0.0",
    lifespan=lifespan,
)


def _conn():
    return db.connect(app.state.db_path)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    detail = []
    for err in exc.errors():
        parts = [str(p) for p in err["loc"] if p != "body"]
        field = ""
        for p in parts:
            field += f"[{p}]" if p.isdigit() else (f".{p}" if field else p)
        detail.append({"field": field or "body", "message": err["msg"]})
    return JSONResponse(status_code=422, content={"detail": detail})


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StationIn(BaseModel):
    station_id: str = Field(min_length=1)
    longitude: float = Field(ge=-180.0, le=180.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    sample_interval_seconds: float = Field(gt=0.0)
    deployed_at: str = Field(min_length=1)


class SampleIn(BaseModel):
    time: str
    u: float | None = None
    v: float | None = None


class BatchIn(BaseModel):
    batch_id: str = Field(min_length=1)
    # Kept permissive so every field problem (wrong types, non-finite numbers,
    # malformed timestamps ...) is collected in one place by the service layer
    # and reported with concrete samples[i].<field> names.
    samples: list[Any] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/stations", status_code=201)
def create_station(payload: StationIn) -> dict[str, Any]:
    # Validate the deployment timestamp up front.
    try:
        service.parse_iso8601(payload.deployed_at)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=[{"field": "deployed_at",
                     "message": "invalid ISO-8601 timestamp"}],
        )

    conn = _conn()
    try:
        if db.get_station(conn, payload.station_id) is not None:
            raise HTTPException(
                status_code=409,
                detail=[{"field": "station_id",
                         "message": "station already exists"}],
            )
        db.create_station(conn, payload.model_dump())
    finally:
        conn.close()
    return {"station_id": payload.station_id, "status": "created"}


@app.get("/stations/{station_id}")
def get_station(station_id: str) -> dict[str, Any]:
    conn = _conn()
    try:
        row = db.get_station(conn, station_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"station {station_id!r} not found")
        return dict(row)
    finally:
        conn.close()


@app.post("/stations/{station_id}/batches")
def submit_batch(station_id: str, payload: BatchIn) -> JSONResponse:
    conn = _conn()
    try:
        samples = payload.samples
        try:
            result = service.submit_batch(conn, station_id, payload.batch_id, samples)
        except service.ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": exc.errors})
        except service.ConflictError as exc:
            return JSONResponse(
                status_code=409,
                content={"detail": [{"field": exc.field, "message": exc.message}]},
            )
        return JSONResponse(status_code=200, content=result)
    finally:
        conn.close()


@app.get("/stations/{station_id}/versions")
def list_versions(station_id: str) -> dict[str, Any]:
    conn = _conn()
    try:
        if db.get_station(conn, station_id) is None:
            raise HTTPException(status_code=404, detail=f"station {station_id!r} not found")
        rows = db.list_versions(conn, station_id)
        return {
            "station_id": station_id,
            "versions": [
                {
                    "version": r["version"],
                    "created_at": r["created_at"],
                    "triggered_by_batch": r["triggered_by_batch"],
                    "n_batches": r["n_batches"],
                    "n_samples": r["n_samples"],
                    "n_valid": r["n_valid"],
                    "record_span_hours": r["record_span_hours"],
                }
                for r in rows
            ],
        }
    finally:
        conn.close()


@app.get("/stations/{station_id}/versions/{version}")
def get_version(station_id: str, version: int) -> dict[str, Any]:
    conn = _conn()
    try:
        if db.get_station(conn, station_id) is None:
            raise HTTPException(status_code=404, detail=f"station {station_id!r} not found")
        row = db.get_version(conn, station_id, version)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"version {version} not found for station {station_id!r}",
            )
        return json.loads(row["result_json"])
    finally:
        conn.close()


@app.get("/stations/{station_id}/latest")
def latest(station_id: str) -> dict[str, Any]:
    conn = _conn()
    try:
        if db.get_station(conn, station_id) is None:
            raise HTTPException(status_code=404, detail=f"station {station_id!r} not found")
        return service.latest_result(conn, station_id)
    finally:
        conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
