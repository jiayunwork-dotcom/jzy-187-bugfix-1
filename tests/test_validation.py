"""Validation tests: every rejected field must be reported explicitly."""

from conftest import make_sample, upload_raw


def upload(client, station_id, batch_id, samples):
    return upload_raw(client, station_id, batch_id, samples)


def fields_of(resp):
    return {e["field"] for e in resp.json()["detail"]}


def test_station_not_found(client, station_factory):
    resp = upload(client, "GHOST", "B1", [make_sample(0, u=1.0, v=1.0)])
    assert resp.status_code == 422
    assert "station_id" in fields_of(resp)


def test_nonfinite_velocity_reported(client, station_factory):
    station_factory()
    samples = [
        make_sample(0, u=1.0, v=float("nan")),
        make_sample(1, u=float("inf"), v=1.0),
        make_sample(2, u=1.0, v=float("-inf")),
    ]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 422, resp.text
    f = fields_of(resp)
    assert "samples[0].v" in f
    assert "samples[1].u" in f
    assert "samples[2].v" in f
    # nothing persisted
    assert client.get("/stations/ST01/versions").json()["versions"] == []


def test_batch_earlier_than_deployment(client, station_factory):
    station_factory(deployed="2020-06-01T00:00:00Z")
    samples = [
        {"time": "2020-05-31T23:00:00Z", "u": 1.0, "v": 1.0},
        {"time": "2020-06-01T01:00:00Z", "u": 1.0, "v": 1.0},
    ]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 422
    f = fields_of(resp)
    assert "samples[0].time" in f
    assert "samples[1].time" not in f


def test_interval_multiple_steps_is_a_gap_not_an_error(client, station_factory):
    # Station expects 1800 s (30 min) cadence; hourly samples sit on the grid
    # with step 2 -> accepted, no samples are invented for the in-between gap.
    station_factory(interval=1800.0)
    samples = [
        {"time": "2020-01-01T00:00:00Z", "u": 1.0, "v": 1.0},
        {"time": "2020-01-01T01:00:00Z", "u": 1.0, "v": 1.0},
    ]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 200, resp.text
    assert resp.json()["record"]["n_samples"] == 2

    samples_bad = [
        {"time": "2020-01-01T00:00:00Z", "u": 1.0, "v": 1.0},
        {"time": "2020-01-01T01:15:00Z", "u": 1.0, "v": 1.0},
    ]
    resp = upload(client, "ST01", "B2", samples_bad)
    assert resp.status_code == 422
    assert "samples[1].time" in fields_of(resp)


def test_unequal_spacing_within_batch(client, station_factory):
    station_factory()
    samples = [
        {"time": "2020-01-01T00:00:00Z", "u": 1.0, "v": 1.0},
        {"time": "2020-01-01T01:00:00Z", "u": 1.0, "v": 1.0},
        {"time": "2020-01-01T02:30:00Z", "u": 1.0, "v": 1.0},
    ]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 422
    assert "samples[2].time" in fields_of(resp)


def test_duplicate_timestamp_within_batch(client, station_factory):
    station_factory()
    samples = [
        {"time": "2020-01-01T00:00:00Z", "u": 1.0, "v": 1.0},
        {"time": "2020-01-01T01:00:00Z", "u": 1.0, "v": 1.0},
        {"time": "2020-01-01T01:00:00Z", "u": 2.0, "v": 2.0},
    ]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 422
    assert "samples[2].time" in fields_of(resp)


def test_duplicate_timestamp_across_batches(client, station_factory):
    station_factory()
    first = [make_sample(i, u=1.0, v=0.0) for i in range(5)]
    assert upload(client, "ST01", "B1", first).status_code == 200
    overlap = [
        {"time": "2020-01-01T04:00:00Z", "u": 9.0, "v": 9.0},
        {"time": "2020-01-01T05:00:00Z", "u": 9.0, "v": 9.0},
    ]
    resp = upload(client, "ST01", "B2", overlap)
    assert resp.status_code == 422
    assert "samples[0].time" in fields_of(resp)


def test_multiple_field_errors_collected_together(client, station_factory):
    station_factory()
    # station exists; feed several independent problems in one request
    samples = [
        {"time": "not-a-time", "u": "abc", "v": 1.0},
        {"time": "2020-01-01T01:00:00Z", "u": float("nan"), "v": 1.0},
        {"time": "2019-01-01T00:00:00Z", "u": 1.0, "v": 1.0},
    ]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 422
    f = fields_of(resp)
    assert "samples[0].time" in f
    assert "samples[0].u" in f
    assert "samples[1].u" in f
    assert "samples[2].time" in f


def test_station_creation_validation(client):
    resp = client.post("/stations", json={
        "station_id": "X",
        "longitude": 999.0,
        "latitude": 30.0,
        "sample_interval_seconds": 3600,
        "deployed_at": "2020-01-01T00:00:00Z",
    })
    assert resp.status_code == 422
