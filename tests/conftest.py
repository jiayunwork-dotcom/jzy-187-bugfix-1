import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("TIDE_DB_PATH", db_path)

    # Import lazily so the env var is picked up by the lifespan.
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def station_factory(client):
    def _create(station_id="ST01", interval=3600.0, deployed="2020-01-01T00:00:00Z",
                lon=122.0, lat=30.0):
        resp = client.post("/stations", json={
            "station_id": station_id,
            "longitude": lon,
            "latitude": lat,
            "sample_interval_seconds": interval,
            "deployed_at": deployed,
        })
        assert resp.status_code == 201, resp.text
        return resp.json()

    return _create


def iso_from_hour(h, origin="2020-01-01T00:00:00Z", interval=3600.0):
    """Build an ISO timestamp for grid step h (h may be fractional grid index
    in hours when interval == 3600 s)."""
    from datetime import datetime, timedelta, timezone

    dt = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=h * interval)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_sample(idx, u=None, v=None, interval=3600.0):
    return {"time": iso_from_hour(idx, interval=interval), "u": _f(u), "v": _f(v)}


def _f(x):
    if x is None:
        return None
    return float(x)


def upload_raw(client, station_id, batch_id, samples):
    """POST a batch, tolerating NaN/Infinity tokens in the raw JSON body
    (httpx's strict json= encoder rejects them)."""
    import json as _json

    body = _json.dumps({"batch_id": batch_id, "samples": samples}, allow_nan=True)
    return client.post(
        f"/stations/{station_id}/batches",
        content=body,
        headers={"Content-Type": "application/json"},
    )


def synth(idx, freq, au=100.0, bu=0.0, av=0.0, bv=100.0, mean=(0.0, 0.0)):
    """u = au cos wt + bu sin wt + mu ; v = av cos wt + bv sin wt + mv."""
    import math

    t = float(idx)
    th = freq * t
    u = au * math.cos(th) + bu * math.sin(th) + mean[0]
    v = av * math.cos(th) + bv * math.sin(th) + mean[1]
    return {"time": iso_from_hour(idx), "u": u, "v": v}


def get_constituent(result, name):
    for c in result["constituents"]:
        if c["name"] == name:
            return c
    return None


def excluded_names(result):
    return {e["name"]: e for e in result["excluded"]}
