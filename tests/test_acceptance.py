"""End-to-end acceptance tests through the HTTP interface."""

import math

from conftest import get_constituent, make_sample, synth, upload_raw
from app.harmonics import FREQUENCIES_RPH

M2 = FREQUENCIES_RPH["M2"]
S2 = FREQUENCIES_RPH["S2"]
K2 = FREQUENCIES_RPH["K2"]

# 200 days of hourly data: every candidate pair is separable.
LONG_HOURS = 200 * 24
# ~5 days: enough for M2 vs mean but far short of the S2 beat.
SHORT_HOURS = 5 * 24


def upload(client, station_id, batch_id, samples):
    return upload_raw(client, station_id, batch_id, samples)


# ---------------------------------------------------------------------------
# 1. Single M2, noiseless -> amplitudes / phases / ellipse within 1e-6
# ---------------------------------------------------------------------------
def test_single_m2_noiseless_recovery(client, station_factory):
    station_factory()
    # Rectilinear along an arbitrary direction: u = A cos wt, v = q*u.
    A = 120.0
    q = 0.6
    samples = [
        synth(i, M2, au=A, bu=0.0, av=q * A, bv=0.0)
        for i in range(LONG_HOURS + 1)
    ]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 200, resp.text
    result = resp.json()

    m2 = get_constituent(result, "M2")
    assert m2 is not None
    expected_major = A * math.hypot(1.0, q)
    ell = m2["ellipse"]
    assert abs(ell["semi_major"] - expected_major) < 1e-6
    assert abs(ell["semi_minor"]) < 1e-6
    assert abs(m2["u"]["amplitude"] - A) < 1e-6
    assert _angle_close(m2["u"]["phase"], 0.0)
    assert abs(m2["v"]["amplitude"] - q * A) < 1e-6
    assert _angle_close(m2["v"]["phase"], 0.0)
    assert abs(ell["inclination"] - math.atan2(q, 1.0)) < 1e-6
    assert _angle_close(ell["phase"], 0.0)

    assert abs(result["mean"]["u"]) < 1e-6
    assert abs(result["mean"]["v"]) < 1e-6


def test_ccw_circle_minor_equals_major_positive(client, station_factory):
    station_factory()
    samples = [synth(i, M2, au=80.0, bu=0.0, av=0.0, bv=80.0)
               for i in range(LONG_HOURS + 1)]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 200, resp.text
    ell = get_constituent(resp.json(), "M2")["ellipse"]
    assert abs(ell["semi_major"] - 80.0) < 1e-6
    assert abs(ell["semi_minor"] - 80.0) < 1e-6
    assert ell["semi_minor"] > 0


def test_cw_circle_minor_negative(client, station_factory):
    station_factory()
    samples = [synth(i, M2, au=80.0, bu=0.0, av=0.0, bv=-80.0)
               for i in range(LONG_HOURS + 1)]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 200, resp.text
    ell = get_constituent(resp.json(), "M2")["ellipse"]
    assert abs(ell["semi_major"] - 80.0) < 1e-6
    assert abs(ell["semi_minor"] + 80.0) < 1e-6
    assert ell["semi_minor"] < 0


def test_coordinate_rotation_keeps_axes_shifts_inclination(client, station_factory):
    station_factory()
    au, bu, av, bv = 70.0, -30.0, 20.0, 50.0

    def build(c, s):
        out = []
        for i in range(LONG_HOURS + 1):
            base = synth(i, M2, au=au, bu=bu, av=av, bv=bv, mean=(15.0, -8.0))
            u0, v0 = base["u"], base["v"]
            out.append({"time": base["time"],
                        "u": c * u0 - s * v0,
                        "v": s * u0 + c * v0})
        return out

    r0 = upload(client, "ST01", "B1", build(1.0, 0.0)).json()
    # Second station with rotated frame.
    station_factory(station_id="ST02")
    alpha = 0.4321
    r1 = upload(client, "ST02", "B1",
                build(math.cos(alpha), math.sin(alpha))).json()

    e0 = get_constituent(r0, "M2")["ellipse"]
    e1 = get_constituent(r1, "M2")["ellipse"]
    assert abs(e1["semi_major"] - e0["semi_major"]) < 1e-7
    assert abs(e1["semi_minor"] - e0["semi_minor"]) < 1e-7
    dtheta = (e1["inclination"] - e0["inclination"]) % math.pi
    assert abs(dtheta - alpha) < 1e-7, dtheta

    # The mean flow vector rotates with the frame as well.
    assert abs(r0["mean"]["speed"] - r1["mean"]["speed"]) < 1e-7
    ddir = (r1["mean"]["direction_rad"] - r0["mean"]["direction_rad"]) % (2 * math.pi)
    assert abs(ddir - alpha) < 1e-7


def test_constant_velocity_only_changes_mean(client, station_factory):
    station_factory()
    base = [synth(i, M2, au=55.0, bu=25.0, av=-15.0, bv=40.0)
            for i in range(LONG_HOURS + 1)]
    shifted = [
        {"time": s["time"], "u": s["u"] + 33.0, "v": s["v"] - 17.0}
        for s in base
    ]
    station_factory(station_id="ST02")
    r0 = upload(client, "ST01", "B1", base).json()
    r1 = upload(client, "ST02", "B1", shifted).json()

    e0 = get_constituent(r0, "M2")
    e1 = get_constituent(r1, "M2")
    for key in ("semi_major", "semi_minor", "inclination", "phase"):
        assert abs(e0["ellipse"][key] - e1["ellipse"][key]) < 1e-7, key
    assert abs(r1["mean"]["u"] - 33.0) < 1e-6
    assert abs(r1["mean"]["v"] + 17.0) < 1e-6
    assert abs(r0["mean"]["u"]) < 1e-6
    assert abs(r0["mean"]["v"]) < 1e-6


# ---------------------------------------------------------------------------
# 2. Record too short -> S2 and K2 never fitted together
# ---------------------------------------------------------------------------
def test_short_record_s2_k2_not_together(client, station_factory):
    station_factory()
    samples = [make_sample(i, u=1.0, v=0.0) for i in range(SHORT_HOURS + 1)]
    resp = upload(client, "ST01", "B1", samples)
    assert resp.status_code == 200, resp.text
    result = resp.json()
    names = {c["name"] for c in result["constituents"]}
    assert not ({"S2", "K2"} <= names)
    assert "M2" in names  # M2 vs mean: only 0.69h needed
    assert result["record"]["span_hours"] == SHORT_HOURS


def test_masking_report_fields(client, station_factory):
    station_factory()
    samples = [make_sample(i, u=0.0, v=0.0) for i in range(SHORT_HOURS + 1)]
    result = upload(client, "ST01", "B1", samples).json()
    excluded = {e["name"]: e for e in result["excluded"]}
    assert "S2" in excluded
    assert excluded["S2"]["masked_by"] == "M2"
    # Beat period M2/S2: 1/|dw| ~= 354.37h
    assert 353.0 < excluded["S2"]["required_hours"] < 356.0
    assert excluded["S2"]["missing_hours"] > SHORT_HOURS
    assert "K2" in excluded


# ---------------------------------------------------------------------------
# 3. Duplicate upload, out-of-order arrival, version archive
# ---------------------------------------------------------------------------
def test_duplicate_upload_is_idempotent(client, station_factory):
    station_factory()
    samples_a = [make_sample(i, u=10.0, v=1.0) for i in range(48)]
    r1 = upload(client, "ST01", "B-A", samples_a)
    assert r1.status_code == 200
    assert r1.json()["version"] == 1
    assert r1.json()["replayed"] is False

    r2 = upload(client, "ST01", "B-A", samples_a)
    assert r2.status_code == 200
    body = r2.json()
    assert body["version"] == 1
    assert body["replayed"] is True

    versions = client.get("/stations/ST01/versions").json()["versions"]
    assert [v["version"] for v in versions] == [1]


def test_duplicate_batch_id_with_other_content_conflicts(client, station_factory):
    station_factory()
    s1 = [make_sample(i, u=1.0, v=0.0) for i in range(10)]
    s2 = [make_sample(i, u=2.0, v=0.0) for i in range(10)]
    assert upload(client, "ST01", "B1", s1).status_code == 200
    resp = upload(client, "ST01", "B1", s2)
    assert resp.status_code == 409
    assert resp.json()["detail"][0]["field"] == "batch_id"


def test_out_of_order_arrival_matches_sequential(client, station_factory):
    """Three disjoint windows; final result identical whatever the order."""

    def windows_split():
        return {
            "B1": (0, 1000),
            "B2": (1000, 2000),
            "B3": (2000, 3000),
        }

    def make_batch(lo, hi):
        return [
            synth(i, M2, au=60.0, bu=20.0, av=10.0, bv=40.0, mean=(5.0, -3.0))
            for i in range(lo, hi)
        ]

    # Station 1: chronological
    station_factory(station_id="SEQ")
    for bid, (lo, hi) in windows_split().items():
        assert upload(client, "SEQ", bid, make_batch(lo, hi)).status_code == 200
    seq = client.get("/stations/SEQ/latest").json()

    # Station 2: reverse order
    station_factory(station_id="REV")
    for bid, (lo, hi) in reversed(list(windows_split().items())):
        assert upload(client, "REV", bid, make_batch(lo, hi)).status_code == 200
    rev = client.get("/stations/REV/latest").json()

    assert seq["record"]["n_samples"] == rev["record"]["n_samples"] == 3000
    _assert_same_analysis(seq, rev, amp_tol=1e-9, phase_tol=1e-7)
    assert sorted(seq["batches"]) == sorted(rev["batches"])


def test_incremental_equals_one_shot_fit(client, station_factory):
    """Batch-by-batch update equals a single fit over the identical samples."""
    station_factory(station_id="INC")
    # Gaps included (missing segments) between 4 windows.
    windows = [(0, 300), (500, 800), (1000, 1300), (1500, 1801)]
    all_samples: list[dict] = []
    for j, (lo, hi) in enumerate(windows):
        batch = [
            synth(i, M2, au=65.0, bu=-18.0, av=22.0, bv=37.0, mean=(4.0, 2.0))
            for i in range(lo, hi)
        ]
        all_samples.extend(batch)
        assert upload(client, "INC", f"W{j}", batch).status_code == 200

    incremental = client.get("/stations/INC/latest").json()

    station_factory(station_id="ONE")
    oneshot = upload(client, "ONE", "ALL", all_samples).json()

    _assert_same_analysis(incremental, oneshot, amp_tol=1e-9, phase_tol=1e-7)
    # Missing segment count: total grid span coverage vs provided samples.
    assert incremental["record"]["n_samples"] == len(all_samples)


def test_every_version_is_archived(client, station_factory):
    station_factory()
    upload(client, "ST01", "B1", [make_sample(i, u=1.0, v=0.0) for i in range(0, 50)])
    upload(client, "ST01", "B2", [make_sample(i, u=1.0, v=0.0) for i in range(50, 100)])
    upload(client, "ST01", "B3", [make_sample(i, u=1.0, v=0.0) for i in range(100, 150)])

    listing = client.get("/stations/ST01/versions").json()["versions"]
    assert [v["version"] for v in listing] == [1, 2, 3]
    v1 = client.get("/stations/ST01/versions/1").json()
    v3 = client.get("/stations/ST01/versions/3").json()
    assert v1["batches"] == ["B1"]
    assert v3["batches"] == ["B1", "B2", "B3"]
    assert v1["record"]["n_samples"] == 50
    assert v3["record"]["n_samples"] == 150
    assert client.get("/stations/ST01/versions/99").status_code == 404


def test_missing_samples_in_gaps(client, station_factory):
    station_factory()
    # One batch with a missing middle section represented as null u/v.
    samples = []
    for i in range(300):
        if 100 <= i < 150:
            samples.append(make_sample(i, u=None, v=None))
        else:
            samples.append(synth(i, M2, au=40.0, bu=0.0, av=0.0, bv=40.0))
    result = upload(client, "ST01", "B1", samples).json()
    assert result["record"]["n_missing"] == 50
    assert result["record"]["n_valid"] == 250
    ell = get_constituent(result, "M2")["ellipse"]
    assert abs(ell["semi_major"] - 40.0) < 1e-5
    assert ell["semi_minor"] > 40.0 - 1e-5


# ---------------------------------------------------------------------------
# 4. Persistence across restart
# ---------------------------------------------------------------------------
def test_versions_survive_restart(client, station_factory, tmp_path, monkeypatch):
    station_factory()
    upload(client, "ST01", "B1", [make_sample(i, u=3.0, v=1.0) for i in range(60)])
    before = client.get("/stations/ST01/versions/1").json()

    # Simulate a restart: drop the app/client and rebuild against the same file.
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client2:
        r = client2.get("/stations/ST01/versions/1")
        assert r.status_code == 200
        after = r.json()
        assert after["record"] == before["record"]
        # Batch can still be replayed idempotently after restart.
        dup = upload(client2, "ST01", "B1",
                     [make_sample(i, u=3.0, v=1.0) for i in range(60)])
        assert dup.status_code == 200
        assert dup.json()["version"] == 1
        assert dup.json()["replayed"] is True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _assert_same_analysis(a, b, amp_tol, phase_tol):
    assert abs(a["mean"]["u"] - b["mean"]["u"]) < amp_tol
    assert abs(a["mean"]["v"] - b["mean"]["v"]) < amp_tol
    ca = {c["name"]: c for c in a["constituents"]}
    cb = {c["name"]: c for c in b["constituents"]}
    assert set(ca) == set(cb)
    for name in ca:
        for comp in ("u", "v"):
            # Relative difference of amplitudes, with an absolute floor for
            # components that are genuinely zero.
            xa, xb = ca[name][comp]["amplitude"], cb[name][comp]["amplitude"]
            assert abs(xa - xb) <= max(amp_tol * max(abs(xa), abs(xb)), 1e-12), name
            assert _phase_diff(ca[name][comp]["phase"],
                               cb[name][comp]["phase"]) < phase_tol
        for key in ("semi_major", "semi_minor", "inclination", "phase"):
            x, y = ca[name]["ellipse"][key], cb[name]["ellipse"][key]
            if key in ("inclination", "phase"):
                assert _phase_diff(x, y) < phase_tol, (name, key)
            else:
                assert abs(x - y) <= max(amp_tol * max(abs(x), abs(y)), 1e-12), name


def _phase_diff(p, q):
    d = abs(p - q) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _angle_close(a, b, tol=1e-6):
    return _phase_diff(a, b) < tol
