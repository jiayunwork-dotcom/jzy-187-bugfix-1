"""Unit tests for the hand-written least-squares and ellipse mathematics."""

import math

import numpy as np

from app.harmonics import (
    FREQUENCIES_RPH,
    ellipse_from_coefficients,
    least_squares_normal_equations,
    rotate_velocity,
    select_constituents,
)

M2 = FREQUENCIES_RPH["M2"]


def test_ellipse_rectilinear_u_direction():
    # u only, along east: a_u = 100, all v coefficients zero.
    e = ellipse_from_coefficients(100.0, 0.0, 0.0, 0.0)
    assert abs(e.semi_major - 100.0) < 1e-12
    assert abs(e.semi_minor) < 1e-12
    assert abs(e.inclination) < 1e-12
    assert abs(e.phase) < 1e-12
    assert abs(e.u_amplitude - 100.0) < 1e-12


def test_ellipse_counterclockwise_circle():
    # u = A cos wt ; v = A sin wt -> CCW circle.
    e = ellipse_from_coefficients(100.0, 0.0, 0.0, 100.0)
    assert abs(e.semi_major - 100.0) < 1e-12
    assert abs(e.semi_minor - 100.0) < 1e-12
    assert e.semi_minor > 0


def test_ellipse_clockwise_circle():
    # u = A cos wt ; v = -A sin wt -> CW circle.
    e = ellipse_from_coefficients(100.0, 0.0, 0.0, -100.0)
    assert abs(e.semi_major - 100.0) < 1e-12
    assert abs(e.semi_minor + 100.0) < 1e-12
    assert e.semi_minor < 0


def test_ellipse_general_rotation_invariance():
    # Rotate the physical vectors by alpha: major/minor unchanged,
    # inclination shifts by exactly alpha.
    au, bu, av, bv = 70.0, -30.0, 20.0, 50.0
    e0 = ellipse_from_coefficients(au, bu, av, bv)
    alpha = 0.37
    # Coefficients rotate exactly the same as instantaneous vectors.
    c, s = math.cos(alpha), math.sin(alpha)
    au2 = c * au - s * av
    bu2 = c * bu - s * bv
    av2 = s * au + c * av
    bv2 = s * bu + c * bv
    e1 = ellipse_from_coefficients(au2, bu2, av2, bv2)

    assert abs(e1.semi_major - e0.semi_major) < 1e-9
    assert abs(e1.semi_minor - e0.semi_minor) < 1e-9
    dtheta = (e1.inclination - e0.inclination) % math.pi
    assert abs(dtheta - alpha) < 1e-9 or abs(dtheta - alpha - math.pi) < 1e-9


def test_least_squares_recovers_coefficients():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 200.0 * 24.0, 1.0)  # hourly, 200 days
    au, bu, av, bv = 83.0, -41.0, 27.0, 55.0
    mu, mv = 12.0, -7.0
    u = au * np.cos(M2 * t) + bu * np.sin(M2 * t) + mu
    v = av * np.cos(M2 * t) + bv * np.sin(M2 * t) + mv

    accepted, _ = select_constituents(float(t[-1]))
    names = [a.name for a in accepted]
    assert "M2" in names
    x, good = least_squares_normal_equations(t, u, v, accepted)
    assert good.all()
    k = names.index("M2")
    assert abs(x[2 * k - 1, 0] - au) < 1e-6
    assert abs(x[2 * k, 0] - bu) < 1e-6
    assert abs(x[2 * k - 1, 1] - av) < 1e-6
    assert abs(x[2 * k, 1] - bv) < 1e-6
    assert abs(x[0, 0] - mu) < 1e-7
    assert abs(x[0, 1] - mv) < 1e-7


def test_least_squares_ignores_missing_samples():
    t = np.arange(0.0, 200.0 * 24.0, 1.0)
    u = 50.0 * np.cos(M2 * t) + 10.0
    v = np.full_like(t, np.nan)
    v[::2] = 50.0 * np.sin(M2 * t)[::2] - 4.0
    accepted, _ = select_constituents(float(t[-1]))
    x, good = least_squares_normal_equations(t, u, v, accepted)
    assert not good.all()
    assert good.sum() == len(t) // 2
    names = [a.name for a in accepted]
    k = names.index("M2")
    assert abs(x[2 * k - 1, 0] - 50.0) < 1e-5
    assert abs(x[2 * k, 1] - 50.0) < 1e-5
    assert abs(x[0, 1] - (-4.0)) < 1e-6


def test_rotate_velocity_helper():
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    u2, v2 = rotate_velocity(u, v, math.pi / 2)
    assert np.allclose(u2, [0.0, -1.0], atol=1e-12)
    assert np.allclose(v2, [1.0, 0.0], atol=1e-12)
