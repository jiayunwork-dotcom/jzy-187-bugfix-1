"""Regression tests for constituent "name stealing" on short records.

Two cases reported against the advisor's hand calculation:

1. A record too short to contain one full M2 cycle published a pure-M2 signal
   as S2 (S2 happened to complete an exact cycle); once extended, the same
   flow was republished as M2.  A rejected constituent must keep occupying its
   frequency slot, so a lower-priority neighbour can never absorb its energy.

2. A 14-day M2+S2 record (M2/S2 beat = 354.37 h not yet complete) dropped S2
   and published its energy under K2, which had only been checked against the
   admitted set -- and S2 was not in it.  Unresolvable pairs must produce
   NEITHER name as an independent entry.

See docs/ALGORITHM.md, section 3 ("asymmetric occupancy" rule).
"""

import math

from conftest import get_constituent, make_sample

from app.harmonics import FREQUENCIES_CPH, FREQUENCIES_RPH, select_constituents

M2 = FREQUENCIES_RPH["M2"]
S2 = FREQUENCIES_RPH["S2"]

# Hand-derived ellipse for the pure-M2 signal below:
#   W+ = (au+bv)/2 + i(av-bu)/2 ; W- = (au-bv)/2 + i(av+bu)/2
_AU, _BU, _AV, _BV = 25.0, 8.0, 4.0, 15.0
_WP = math.hypot(_AU + _BV, _AV - _BU) / 2.0
_WM = math.hypot(_AU - _BV, _AV + _BU) / 2.0
EXPECTED_M2_MAJOR = _WP + _WM  # ~27.910


def _m2_sample(i):
    t = float(i)
    return make_sample(
        i,
        u=_AU * math.cos(M2 * t) + _BU * math.sin(M2 * t),
        v=_AV * math.cos(M2 * t) + _BV * math.sin(M2 * t),
    )


def _m2_s2_sample(i):
    t = float(i)
    u = (30.0 * math.cos(M2 * t) + 10.0 * math.sin(M2 * t)
         + 10.0 * math.cos(S2 * t))
    v = (5.0 * math.cos(M2 * t) + 20.0 * math.sin(M2 * t)
         + 8.0 * math.sin(S2 * t))
    return make_sample(i, u=u, v=v)


def upload(client, station_id, batch_id, samples):
    return client.post(
        f"/stations/{station_id}/batches",
        json={"batch_id": batch_id, "samples": samples},
    )


# ---------------------------------------------------------------------------
# Example 1: pure M2, 13 points (0..12 h) then extended to 0..60 h
# ---------------------------------------------------------------------------
def test_short_pure_m2_is_not_published_as_s2(client, station_factory):
    station_factory()
    resp = upload(client, "ST01", "B1", [_m2_sample(i) for i in range(13)])
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # 12 h < one M2 cycle (12.42 h): M2 cannot be named, and S2 -- which is not
    # separable from M2 -- must not be named either.  No phantom constituent.
    assert body["constituents"] == []
    excluded = {e["name"]: e for e in body["excluded"]}
    assert excluded["M2"]["masked_by"] == "mean"
    assert excluded["S2"]["masked_by"] == "M2"
    beat_m2_s2 = 1.0 / abs(FREQUENCIES_CPH["M2"] - FREQUENCIES_CPH["S2"])
    assert abs(excluded["S2"]["required_hours"] - beat_m2_s2) < 1e-6
    # missing_hours is consistent with the stated gate and the current span.
    assert abs(excluded["S2"]["missing_hours"]
               - (beat_m2_s2 - body["record"]["span_hours"])) < 1e-9


def test_pure_m2_keeps_its_name_once_the_record_grows(client, station_factory):
    station_factory()
    upload(client, "ST01", "B1", [_m2_sample(i) for i in range(13)])
    resp = upload(client, "ST01", "B2",
                  [_m2_sample(i) for i in range(13, 61)])
    assert resp.status_code == 200, resp.text
    body = resp.json()

    names = {c["name"] for c in body["constituents"]}
    assert "M2" in names
    assert "S2" not in names  # M2/S2 beat (354 h) still far away
    m2 = get_constituent(body, "M2")
    assert abs(m2["ellipse"]["semi_major"] - EXPECTED_M2_MAJOR) < 1e-6
    # The mean must absorb no tidal energy at this length either.
    assert abs(body["mean"]["u"]) < 1e-6
    assert abs(body["mean"]["v"]) < 1e-6


def test_admitted_constituent_is_never_evicted_as_record_grows(
    client, station_factory
):
    """Same station, record only extended: a name once in must stay in."""
    station_factory()
    upload(client, "ST01", "B1", [_m2_sample(i) for i in range(13)])
    seen: set[str] = set()
    # Extend in 7-hour steps well past every tabulated beat threshold.
    lo, hi, step = 13, 4500, 7
    for n, end in enumerate(range(lo, hi + 1, step)):
        r = upload(client, "ST01", f"B{n+2}",
                   [_m2_sample(i) for i in range(lo, end + 1)])
        assert r.status_code == 200, r.text
        names = {c["name"] for c in r.json()["constituents"]}
        assert seen <= names, f"eviction at span ending {end}h: {seen} -> {names}"
        seen = names
        lo = end + 1


# ---------------------------------------------------------------------------
# Example 2: M2 + S2 over 14 days; the signal contains no K2
# ---------------------------------------------------------------------------
def test_m2_s2_fourteen_days_does_not_create_k2(client, station_factory):
    station_factory()
    hours = 14 * 24
    resp = upload(client, "ST01", "B1",
                  [_m2_s2_sample(i) for i in range(hours + 1)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["record"]["span_hours"] == float(hours)

    names = {c["name"] for c in body["constituents"]}
    # The M2/S2 beat needs 354.37 h; with only 336 h the pair is unresolved,
    # so neither S2 nor its near-twin K2 may appear as an independent entry.
    assert "S2" not in names
    assert "K2" not in names
    assert "M2" in names

    excluded = {e["name"]: e for e in body["excluded"]}
    # S2 is blocked by the stronger, higher-priority M2 (18.4 h short).
    assert excluded["S2"]["masked_by"] == "M2"
    beat_m2_s2 = 1.0 / abs(FREQUENCIES_CPH["M2"] - FREQUENCIES_CPH["S2"])
    assert abs(excluded["S2"]["required_hours"] - beat_m2_s2) < 1e-6
    assert abs(excluded["S2"]["missing_hours"] - (beat_m2_s2 - hours)) < 1e-9
    # K2 is blocked by the rejected S2 still occupying its slot; the quoted
    # gate is the full S2/K2 beat period, 4382.9 h (~182.6 d).
    assert excluded["K2"]["masked_by"] == "S2"
    beat_s2_k2 = 1.0 / abs(FREQUENCIES_CPH["S2"] - FREQUENCIES_CPH["K2"])
    assert abs(excluded["K2"]["required_hours"] - beat_s2_k2) < 1e-6
    assert abs(excluded["K2"]["missing_hours"]
               - (beat_s2_k2 - hours)) < 1e-9


# ---------------------------------------------------------------------------
# Unit-level: the rule is a fixed, span-monotone gate table
# ---------------------------------------------------------------------------
def test_admission_thresholds_are_the_priority_prefix_beats():
    """First admissible span for each constituent = max over the mean and the
    constituents ahead of it of R / |df|, regardless of whether those are
    themselves admitted."""
    priority = ["M2", "S2", "N2", "K2", "K1", "O1"]
    expected = {}
    for i, name in enumerate(priority):
        gates = [1.0 / FREQUENCIES_CPH[name]]  # one own cycle vs the mean
        for other in priority[:i]:
            gates.append(1.0 / abs(FREQUENCIES_CPH[name]
                                   - FREQUENCIES_CPH[other]))
        expected[name] = max(gates)

    # Just below the threshold the constituent is still out; at the threshold
    # (within the span tolerance) it is in.
    for name, threshold in expected.items():
        before = {s.name for s in select_constituents(threshold - 1e-6)[0]}
        at = {s.name for s in select_constituents(threshold)[0]}
        assert name not in before
        assert name in at

    # Spot values quoted in the docs.
    assert abs(expected["M2"] - 12.4206) < 1e-3
    assert abs(expected["S2"] - 354.367) < 1e-2
    assert abs(expected["N2"] - 661.309) < 1e-2
    assert abs(expected["K2"] - 4382.905) < 1e-2
    assert abs(expected["K1"] - 26.869) < 1e-2
    assert abs(expected["O1"] - 327.859) < 1e-2


def test_admitted_set_is_monotone_over_all_hourly_spans():
    seen: set[str] = set()
    for hours in range(0, 4500):
        current = {s.name for s in select_constituents(float(hours))[0]}
        assert seen <= current, f"lost constituents at {hours}h: {seen - current}"
        seen = current
    assert seen == {"mean", "M2", "S2", "N2", "K2", "K1", "O1"}
