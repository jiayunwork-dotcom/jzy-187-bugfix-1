"""Edge cases for very short / incomplete records."""

import math

from conftest import get_constituent, iso_from_hour, make_sample

from app.harmonics import FREQUENCIES_RPH

M2 = FREQUENCIES_RPH["M2"]
S2 = FREQUENCIES_RPH["S2"]


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
    # Span 12h: M2's 12.42h period is not complete, so M2 is blocked by the
    # mean; S2 covers one of its own cycles but stays blocked by M2's slot
    # (354h beat), and the weak K2 can never accompany S2 below the 4383h
    # beat.
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


# ---------------------------------------------------------------------------
# Regression: reported field cases where a rejected constituent stopped
# masking its neighbours and their energy was booked under a wrong name.
# ---------------------------------------------------------------------------
def test_pure_m2_short_record_not_attributed_to_s2(client, station_factory):
    """Pure M2 signal, 13 hourly points (12 h span) -> no tidal entries.

    M2 needs 12.42 h against the mean, so at 12 h nothing is resolvable; the
    old rule admitted S2 (exactly one S2 cycle) and booked M2's energy under
    it (semi-major ~27.8).  At 60 h M2 must appear with its true ellipse, and
    once admitted it must survive every later version.
    """
    station_factory()

    def sample(i):
        th = M2 * i
        return {"time": iso_from_hour(i),
                "u": 25.0 * math.cos(th) + 8.0 * math.sin(th),
                "v": 4.0 * math.cos(th) + 15.0 * math.sin(th)}

    r1 = upload(client, "ST01", "B1", [sample(i) for i in range(13)])
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["record"]["span_hours"] == 12.0
    assert {c["name"] for c in body1["constituents"]} == set()
    ex1 = {e["name"]: e for e in body1["excluded"]}
    assert ex1["M2"]["masked_by"] == "mean"
    assert ex1["S2"]["masked_by"] == "M2"

    r2 = upload(client, "ST01", "B2", [sample(i) for i in range(13, 61)])
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    names2 = {c["name"] for c in body2["constituents"]}
    assert "M2" in names2
    assert "S2" not in names2
    m2 = get_constituent(body2, "M2")
    # True ellipse: |W+| = |20 - 2i|, |W-| = |5 + 6i| -> major 27.91000.
    assert abs(m2["ellipse"]["semi_major"] - 27.91) < 1e-3
    assert abs(m2["u"]["amplitude"] - math.hypot(25.0, 8.0)) < 1e-6
    assert abs(m2["v"]["amplitude"] - math.hypot(4.0, 15.0)) < 1e-6

    # Grow past the M2/S2 beat: M2 is never pushed back out.
    r3 = upload(client, "ST01", "B3", [sample(i) for i in range(61, 401)])
    assert r3.status_code == 200, r3.text
    names3 = {c["name"] for c in r3.json()["constituents"]}
    assert "M2" in names3
    assert abs(get_constituent(r3.json(), "M2")["ellipse"]["semi_major"]
               - 27.91) < 1e-3


def test_m2_s2_record_does_not_produce_phantom_k2(client, station_factory):
    """14 days of M2+S2 (no K2 in the signal) -> K2 must not appear.

    At 336 h S2 is 18 h short of its M2 beat and stays excluded; the old rule
    then let K2 in (327.86 h from M2) where it soaked up S2's energy
    (u amplitude ~9.9).  K2 needs the full S2 beat (4382.9 h).  Past 354 h S2
    enters with its true amplitudes and M2 stays put.
    """
    station_factory()

    def sample(i):
        return {"time": iso_from_hour(i),
                "u": (30.0 * math.cos(M2 * i) + 10.0 * math.sin(M2 * i)
                      + 10.0 * math.cos(S2 * i)),
                "v": (5.0 * math.cos(M2 * i) + 20.0 * math.sin(M2 * i)
                      + 8.0 * math.sin(S2 * i))}

    r1 = upload(client, "ST01", "B1", [sample(i) for i in range(337)])
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["record"]["span_hours"] == 336.0
    names1 = {c["name"] for c in body1["constituents"]}
    assert "M2" in names1
    assert "S2" not in names1
    assert "K2" not in names1
    ex1 = {e["name"]: e for e in body1["excluded"]}
    assert ex1["S2"]["masked_by"] == "M2"
    assert 354.0 < ex1["S2"]["required_hours"] < 355.0
    assert 17.0 < ex1["S2"]["missing_hours"] < 19.0
    assert ex1["K2"]["masked_by"] == "S2"
    assert 4382.0 < ex1["K2"]["required_hours"] < 4383.5

    # Extend past the M2/S2 beat: S2 enters with its true amplitudes, M2
    # stays, K2 remains excluded.
    r2 = upload(client, "ST01", "B2", [sample(i) for i in range(337, 401)])
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    names2 = {c["name"] for c in body2["constituents"]}
    assert {"M2", "S2"} <= names2
    assert "K2" not in names2
    s2 = get_constituent(body2, "S2")
    assert abs(s2["u"]["amplitude"] - 10.0) < 1e-6
    assert abs(s2["v"]["amplitude"] - 8.0) < 1e-6
    assert "K2" in {e["name"] for e in body2["excluded"]}

    # Admission is monotone across archived versions.
    v1 = client.get("/stations/ST01/versions/1").json()
    v2 = client.get("/stations/ST01/versions/2").json()
    names_v1 = {c["name"] for c in v1["constituents"]}
    names_v2 = {c["name"] for c in v2["constituents"]}
    assert names_v1 <= names_v2
