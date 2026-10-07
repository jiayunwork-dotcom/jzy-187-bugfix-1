"""Harmonic analysis core: constituent selection, least-squares fit, ellipse math.

Conventions (see docs/ALGORITHM.md):
  * Frequencies are tabulated in degrees per mean solar hour and converted to
    radians per hour at import time.
  * Time origin for every fit is the station deployment instant (t = 0).
  * A single constituent k is represented by
        u(t) = a_u cos(w t) + b_u sin(w t)
        v(t) = a_v cos(w t) + b_v sin(w t)
    Amplitude U = hypot(a_u, b_u); phase phi_u = atan2(b_u, a_u) in [0, 2pi).
    The ellipse phase below is defined so that at t = 0 (deployment) the flow
    is on the major axis when phi = 0.
  * Signed minor axis: positive = counter-clockwise (cyclonic),
    negative = clockwise (anticyclonic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constituent table.  Frequencies in degrees per mean solar hour.
# Sources: standard Doodson tabulation (e.g. T_TIDE constituent list).
# ---------------------------------------------------------------------------
_FREQ_DEG_PER_HOUR = {
    "M2": 28.9841042,
    "S2": 30.0000000,
    "N2": 28.4397295,
    "K2": 30.0821373,
    "K1": 15.0410686,
    "O1": 13.9430356,
}

# Angular frequencies in radians per hour (used by the fitter).
FREQUENCIES_RPH: dict[str, float] = {
    name: math.radians(deg) for name, deg in _FREQ_DEG_PER_HOUR.items()
}

# Frequencies in cycles per hour (one cycle = 360 degrees); the Rayleigh
# separation test is expressed with these.
FREQUENCIES_CPH: dict[str, float] = {
    name: deg / 360.0 for name, deg in _FREQ_DEG_PER_HOUR.items()
}

# Mean flow has frequency zero and is always part of the model.
MEAN_NAME = "mean"

# Admission priority, strongest / most important first within each species
# (semidiurnal: M2 > S2 > N2 > K2; diurnal: K1 > O1).
#
# The priority list is also the *frequency ownership* order: a constituent is
# admitted only if it is Rayleigh-separable from the mean and from EVERY
# constituent ahead of it in this list -- whether or not that higher-priority
# constituent was itself admitted.  A rejected constituent still "occupies" its
# frequency slot: a lower-priority candidate that cannot be told apart from it
# is refused as well, so the rejected constituent's energy can never be
# published under the lower-priority name.  Blocking is deliberately
# asymmetric (a rejected weak constituent never blocks a stronger one): the
# higher-priority member of an unresolvable pair is treated as the owner of the
# pair's energy, which keeps the classical admission thresholds (e.g. S2 at
# ~14.8 days) instead of forcing both members to wait for their mutual beat.
CONSTITUENT_PRIORITY = ["M2", "S2", "N2", "K2", "K1", "O1"]

# Rayleigh separation factor: require record_span >= R / |df| (in hours).
RAYLEIGH_FACTOR = 1.0

# Tolerance (hours) on the span comparison, so a span sitting exactly on a
# tabulated threshold counts as satisfying it.
_SPAN_TOL = 1e-9

_TWO_PI = 2.0 * math.pi


@dataclass
class SelectedConstituent:
    name: str
    frequency: float  # rad/hour; 0.0 for the mean flow


@dataclass
class RejectedConstituent:
    name: str
    frequency: float
    masked_by: str
    required_hours: float  # record span needed to separate from masked_by
    current_hours: float


@dataclass
class Ellipse:
    semi_major: float       # cm/s, non-negative
    semi_minor: float       # cm/s, signed (CCW > 0)
    inclination: float      # radians, major axis CCW from east, [0, pi)
    phase: float            # radians, [0, 2pi)
    u_amplitude: float      # cm/s
    u_phase: float          # radians, [0, 2pi)
    v_amplitude: float      # cm/s
    v_phase: float          # radians, [0, 2pi)


def _wrap_positive(angle: float) -> float:
    """Wrap angle into [0, 2*pi); an exact multiple wraps to 0."""
    a = angle % _TWO_PI
    return 0.0 if a == _TWO_PI else a


def _wrap_inclination(angle: float) -> float:
    """Wrap an inclination into [0, pi)."""
    a = angle % math.pi
    return 0.0 if a == math.pi else a


def select_constituents(record_hours: float) -> tuple[
    list[SelectedConstituent], list[RejectedConstituent]
]:
    """Select the constituents resolvable by a record of ``record_hours``.

    Classical Rayleigh criterion, f in cycles per hour, factor R = 1:

        record_hours >= R / |f_c - f_j|

    A candidate c is admitted iff it is separable from

      1. the mean (f = 0) -- one full constituent cycle must have elapsed,
         otherwise its cos/sin columns are indistinguishable from the
         constant column; and
      2. every constituent j that precedes c in ``CONSTITUENT_PRIORITY``,
         **including ones that were themselves rejected**.

    The second point is the fix for the "borrowed name" failure of a plain
    greedy pass that only compares against the admitted set: a rejected
    constituent keeps occupying its frequency, so a near-by lower-priority
    candidate cannot slip in and absorb the rejected constituent's energy
    (short pure-M2 record published as S2; S2 energy published as K2 once S2
    itself is blocked by M2).  Blocking is one-way -- a candidate is never
    tested against constituents behind it -- so the stronger member of an
    unresolved pair, not the weaker, holds the slot.

    Because every comparison is a fixed function of the span, the admitted set
    is monotone in ``record_hours``: once a constituent enters it is never
    pushed out again as the record grows.

    For a rejected candidate the reported ``masked_by`` / ``required_hours``
    are the gate that is currently failing with the SMALLEST required span
    (the closest blocker, what an analyst would quote first):

      * if the span does not yet cover one full cycle of the candidate, the
        mean is reported;
      * otherwise the higher-priority constituent whose beat period is the
        nearest unmet threshold.

    ``missing_hours`` is relative to that gate; a longer gate may exist and
    become the binding one later (the caller derives the deficit from the
    reported required span and the current span).
    """
    accepted: list[SelectedConstituent] = [
        SelectedConstituent(MEAN_NAME, 0.0)
    ]
    rejected: list[RejectedConstituent] = []

    # Candidates are tested against every HIGHER-priority constituent, whether
    # admitted or not; this list is therefore the prefix of the priority list,
    # not the accepted list.
    higher: list[str] = []

    for name in CONSTITUENT_PRIORITY:
        w = FREQUENCIES_RPH[name]
        f_cycles = FREQUENCIES_CPH[name]

        # Gate 0: separation from the constant column = one constituent cycle.
        period = RAYLEIGH_FACTOR / f_cycles
        short_of_one_cycle = record_hours + _SPAN_TOL < period

        # Gates 1..k: one full beat period against each higher-priority
        # constituent, admitted or rejected.
        gates: list[tuple[str, float]] = []
        for other in higher:
            required = RAYLEIGH_FACTOR / abs(f_cycles - FREQUENCIES_CPH[other])
            if record_hours + _SPAN_TOL < required:
                gates.append((other, required))

        if not short_of_one_cycle and not gates:
            accepted.append(SelectedConstituent(name, w))
        else:
            if short_of_one_cycle:
                # The record does not even contain one cycle of the candidate:
                # the constant column is the meaningful blocker, regardless of
                # beat gates with numerically similar thresholds.
                blocker_name, required = MEAN_NAME, period
            else:
                # Nearest failing beat gate = the smallest still-unmet required
                # span.  Ties (numerically identical beat periods) resolve by
                # priority order, which is the order of ``higher``.
                order = {n: i for i, n in enumerate(higher)}

                def _gate_key(gate: tuple[str, float]) -> tuple[float, int]:
                    label, req = gate
                    return (req, order[label])

                blocker_name, required = min(gates, key=_gate_key)

            rejected.append(
                RejectedConstituent(
                    name=name,
                    frequency=w,
                    masked_by=blocker_name,
                    required_hours=required,
                    current_hours=record_hours,
                )
            )

        higher.append(name)

    return accepted, rejected


def build_design_matrix(t_hours: np.ndarray,
                        accepted: list[SelectedConstituent]) -> np.ndarray:
    """Assemble A where columns are [1, cos w t, sin w t, ...] per constituent.

    Column order: the mean occupies column 0; constituent k (1-based) occupies
    columns 2*k-1 (cosine) and 2*k (sine).
    """
    t = np.asarray(t_hours, dtype=np.float64)
    n = t.shape[0]
    cols = [np.ones((n, 1), dtype=np.float64)]
    for sel in accepted[1:]:
        theta = sel.frequency * t
        cols.append(np.cos(theta)[:, None])
        cols.append(np.sin(theta)[:, None])
    return np.hstack(cols)


def least_squares_normal_equations(
    t_hours: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    accepted: list[SelectedConstituent],
) -> tuple[np.ndarray, np.ndarray]:
    """Hand-written least squares via the normal equations A^T A x = A^T y.

    The normal matrix and right-hand side are assembled explicitly (no calls
    to lstsq / pinv / QR); only the final symmetric linear system is handed to
    ``numpy.linalg.solve``.  Rows containing non-finite u/v (missing samples)
    are skipped.
    """
    t = np.asarray(t_hours, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    good = np.isfinite(t) & np.isfinite(u) & np.isfinite(v)
    t, u, v = t[good], u[good], v[good]

    a = build_design_matrix(t, accepted)
    # Explicit sums of outer products / products:
    #   ata_ij = sum_n a_ni a_nj ;  atb_i = sum_n a_ni y_n
    ata = np.einsum("ni,nj->ij", a, a, optimize=False)
    atb = np.empty((a.shape[1], 2), dtype=np.float64)
    atb[:, 0] = np.einsum("ni,n->i", a, u, optimize=False)
    atb[:, 1] = np.einsum("ni,n->i", a, v, optimize=False)

    # Solve AᵀA x = Aᵀy.  The matrix can be rank-deficient for very short
    # records (e.g. fewer observations than tidal columns).  In that case we
    # solve in eigen-space and zero out null-space directions, which preserves
    # the correct solution on the identifiable subspace (notably the constant
    # column -> true sample mean) instead of returning the SVD minimum-norm
    # solution, which is biased for rank-deficient *normal* equations.
    eigvals, eigvecs = np.linalg.eigh(ata)
    cutoff = 1e-12 * max(1.0, float(eigvals[-1]))
    inv = np.zeros_like(eigvals)
    nz = eigvals > cutoff
    inv[nz] = 1.0 / eigvals[nz]
    x = (eigvecs * inv) @ (eigvecs.T @ atb)

    return x, good


def ellipse_from_coefficients(au: float, bu: float, av: float, bv: float) -> Ellipse:
    """Convert cos/sin coefficients of u,v to tidal ellipse elements.

    Derivation (z = u + i v):
        Wp = (au + bv)/2 + i(av - bu)/2   # counter-clockwise rotary component
        Wm = (au - bv)/2 + i(av + bu)/2   # clockwise rotary component
        major = |Wp| + |Wm|
        minor = |Wp| - |Wm|   (positive => net CCW)
        inclination = (arg Wp + arg Wm) / 2  mod pi
        phase       = (arg Wm - arg Wp) / 2  mod 2pi
    """
    wp_re = 0.5 * (au + bv)
    wp_im = 0.5 * (av - bu)
    wm_re = 0.5 * (au - bv)
    wm_im = 0.5 * (av + bu)

    wp_abs = math.hypot(wp_re, wp_im)
    wm_abs = math.hypot(wm_re, wm_im)

    semi_major = wp_abs + wm_abs
    semi_minor = wp_abs - wm_abs
    if abs(semi_major) < 1e-300:
        semi_major = 0.0
    if abs(semi_minor) < 1e-300:
        semi_minor = 0.0

    wp_arg = math.atan2(wp_im, wp_re)
    wm_arg = math.atan2(wm_im, wm_re)

    inclination = _wrap_inclination(0.5 * (wp_arg + wm_arg))
    phase = _wrap_positive(0.5 * (wm_arg - wp_arg))

    u_amplitude = math.hypot(au, bu)
    v_amplitude = math.hypot(av, bv)
    u_phase = _wrap_positive(math.atan2(bu, au))
    v_phase = _wrap_positive(math.atan2(bv, av))

    return Ellipse(
        semi_major=semi_major,
        semi_minor=semi_minor,
        inclination=inclination,
        phase=phase,
        u_amplitude=u_amplitude,
        u_phase=u_phase,
        v_amplitude=v_amplitude,
        v_phase=v_phase,
    )


def rotate_velocity(u: np.ndarray, v: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate velocity vectors by alpha radians CCW (used in tests/clients)."""
    c, s = math.cos(alpha), math.sin(alpha)
    return c * u - s * v, s * u + c * v
