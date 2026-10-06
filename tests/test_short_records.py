"""Edge cases for very short / incomplete records."""

from conftest import make_sample


def upload(client, station_id, batch_id, samples):
    return client.post(
        f"/stations/{station_id}/batches",
        json={"batch_id": batch_id, "samples": samples},
    )


def test_single_sample_returns_mean_only(client, station_factory):
    station_factory()
    resp = upload(client, "ST01", "B1", [make_sample(0, u=7.0, v=-2.0)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["record"]["span_hours"] == 0.0
    assert body["constituents"] == []
    assert abs(body["mean"]["u"] - 7.0) < 1e-12
    assert abs(body["mean"]["v"] + 2.0) < 1e-12
    excluded = {e["name"]: e for e in body["excluded"]}
    assert set(excluded) == {"M2", "S2", "N2", "K2", "K1", "O1"}
    # M2 needs one cycle (~12.42 h) to separate from the constant column.
    assert excluded["M2"]["masked_by"] == "mean"
    assert 12.0 < excluded["M2"]["required_hours"] < 12.8


def test_short_span_below_one_period_blocks_all_tides(client, station_factory):
    station_factory()
    # 3 hours of data: below the K1 (~23.93h) and every semidiurnal period.
    resp = upload(client, "ST01", "B1",
                  [make_sample(i, u=float(i), v=0.0) for i in range(4)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["constituents"] == []
    assert body["record"]["span_hours"] == 3.0
    assert abs(body["mean"]["u"] - 1.5) < 1e-9  # mean of 0,1,2,3
    for e in body["excluded"]:
        assert e["masked_by"] == "mean"
        assert e["missing_hours"] > 0.0


def test_all_missing_samples_keeps_empty_fit(client, station_factory):
    station_factory()
    samples = [make_sample(i, u=None, v=None) for i in range(5)]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["record"]["n_samples"] == 5
    assert body["record"]["n_valid"] == 0
    assert body["record"]["n_missing"] == 5
    assert body["constituents"] == []
    assert body["mean"]["speed"] == 0.0


def test_greedy_admission_below_beat_period(client, station_factory):
    station_factory()
    # Span 12h: one exact S2 cycle -> S2 orthogonal to the constant column and
    # admissible; M2's 12.42h period is not complete yet, so M2 is blocked by
    # the mean.  The weak K2 can never accompany S2 below the 697h beat.
    r1 = upload(client, "ST01", "B1",
                [make_sample(i, u=1.0, v=0.0) for i in range(13)])
    names1 = {c["name"] for c in r1.json()["constituents"]}
    assert "M2" not in names1
    assert "K2" not in names1
    excluded = {e["name"]: e for e in r1.json()["excluded"]}
    assert excluded["M2"]["masked_by"] in {"mean", "S2"}

    # One more hour (13h span): still far below every beat period.  Neither M2
    # nor K2 may appear alongside S2.
    r2 = client.post(
        "/stations/ST01/batches",
        json={"batch_id": "B2",
              "samples": [make_sample(i, u=1.0, v=0.0) for i in range(13, 14)]},
    )
    assert r2.status_code == 200, r2.text
    names2 = {c["name"] for c in r2.json()["constituents"]}
    assert "K2" not in names2
    assert not ({"S2", "K2"} <= names2)
