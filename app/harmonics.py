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

# Acceptance priority: constituents are considered in this order, so a rejected
# weak constituent is "masked" by the most important member of the
# already-accepted set that it cannot be separated from.  The order puts the
# strongest constituents first within each species (semidiurnal:
# M2 > S2 > N2 > K2; diurnal: K1 > O1).
CONSTITUENT_PRIORITY = ["M2", "S2", "N2", "K2", "K1", "O1"]

# Rayleigh separation factor: require record_span >= R / |dw| (in hours).
RAYLEIGH_FACTOR = 1.0

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
    """Greedily accept constituents that the observed record can resolve.

    Rule (classical Rayleigh criterion): candidate c is accepted iff for
    every frequency already in the model, including the mean at f = 0,
        record_hours >= RAYLEIGH_FACTOR / |f_c - f_j|
    with f in cycles per hour.  Against the mean this demands at least one
    constituent cycle; against another tidal constituent it demands one full
    beat period, at which point the cosine/sine basis pairs are (nearly)
    orthogonal.  When rejected, the closest in-model frequency is reported as
    the "masking" constituent, along with the span that is still missing.
    """
    accepted: list[SelectedConstituent] = [
        SelectedConstituent(MEAN_NAME, 0.0)
    ]
    rejected: list[RejectedConstituent] = []

    for name in CONSTITUENT_PRIORITY:
        w = FREQUENCIES_RPH[name]
        f_cycles = FREQUENCIES_CPH[name]
        # The constituent must be separable from *every* frequency already in
        # the model, including the mean (frequency 0): with no positive span a
        # tidal column is indistinguishable from the constant column.  Record
        # the closest neighbour that blocks it (what an analyst would quote as
        # the interfering constituent).
        blockers: list[tuple[str, float]] = []
        for sel in accepted:
            dw = abs(f_cycles - (0.0 if sel.name == MEAN_NAME else FREQUENCIES_CPH[sel.name]))
            required = RAYLEIGH_FACTOR / dw
            if record_hours + 1e-9 < required:
                blockers.append((sel.name, required))

        if not blockers:
            accepted.append(SelectedConstituent(name, w))
        else:
            def _freq_of(label: str) -> float:
                return 0.0 if label == MEAN_NAME else FREQUENCIES_CPH[label]

            blocker_name, required = min(
                blockers, key=lambda item: abs(f_cycles - _freq_of(item[0]))
            )
            rejected.append(
                RejectedConstituent(
                    name=name,
                    frequency=w,
                    masked_by=blocker_name,
                    required_hours=required,
                    current_hours=record_hours,
                )
            )

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
