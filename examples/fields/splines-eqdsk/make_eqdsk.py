#!/usr/bin/env python3
"""Writes solovev.eqdsk, the equilibrium file of the splines-eqdsk example.

The equilibrium is an analytic one, a member of the Solov'ev family, so the
example can compare what it computes from the file with the exact answer:

    psi(R, Z) = psi_axis + A (R^2 - R0^2)^2 + B Z^2

This solves the Grad-Shafranov equation for a constant p'(psi) and a constant
F F'(psi): with Delta* psi = R d/dR (R^-1 d psi/dR) + d^2 psi/dZ^2, one finds
Delta* psi = 8 A R^2 + 2 B. The magnetic axis, the minimum of psi, is at
(R0, 0). The last closed flux surface is where psi = psi_boundary; A and B are
chosen so that it crosses the midplane at R0 + a and the axis at height
kappa * a.

The file is in the G-EQDSK format: a header line, four lines of numbers, the
profiles on the normalized flux grid, psi on the (R, Z) grid with R varying
fastest, and the boundary and limiter points, with five numbers (E16.9) per
line. Only psi is needed by the example; the other entries are filled with
consistent values.

Usage: make_eqdsk.py [output file]
"""
import sys

import numpy as np

# the grid: nw points in R, nh points in Z
NW, NH = 25, 33
R_LEFT, R_DIM = 1.0, 1.4
Z_MID, Z_DIM = 0.0, 2.0

# the equilibrium
R0, A_MINOR, KAPPA = 1.7, 0.5, 1.6
PSI_AXIS, PSI_BOUNDARY = -0.4, 0.6
B_TOROIDAL = 2.0  # at R0, in tesla
CURRENT = 1.0e6  # in ampere


def coefficients():
    """The coefficients A and B of psi."""
    delta = PSI_BOUNDARY - PSI_AXIS
    a_coef = delta / ((R0 + A_MINOR) ** 2 - R0 ** 2) ** 2
    b_coef = delta / (KAPPA * A_MINOR) ** 2
    return a_coef, b_coef


def psi(r, z):
    """The poloidal flux in weber per radian."""
    a_coef, b_coef = coefficients()
    return PSI_AXIS + a_coef * (r ** 2 - R0 ** 2) ** 2 + b_coef * z ** 2


def boundary(num_points=33):
    """Points of the last closed flux surface, going around the axis."""
    a_coef, b_coef = coefficients()
    delta = PSI_BOUNDARY - PSI_AXIS
    r_points, z_points = [], []
    for theta in np.linspace(0.0, 2.0 * np.pi, num_points):
        # along the ray from the axis, psi - psi_axis grows monotonically
        lo, hi = 0.0, 1.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            value = (a_coef * ((R0 + mid * np.cos(theta)) ** 2 - R0 ** 2) ** 2
                     + b_coef * (mid * np.sin(theta)) ** 2)
            lo, hi = (mid, hi) if value < delta else (lo, mid)
        r_points.append(R0 + lo * np.cos(theta))
        z_points.append(lo * np.sin(theta))
    return np.array(r_points), np.array(z_points)


def numbers(values):
    """Five numbers per line, as the format has them."""
    values = np.asarray(values, dtype=float).ravel()
    lines = []
    for start in range(0, len(values), 5):
        lines.append("".join("%16.9E" % v for v in values[start:start + 5]))
    return "\n".join(lines) + "\n"


def write(path):
    r = R_LEFT + R_DIM * np.arange(NW) / (NW - 1)
    z = Z_MID - Z_DIM / 2.0 + Z_DIM * np.arange(NH) / (NH - 1)
    psirz = psi(r[:, None], z[None, :])  # [i_R, i_Z]
    psin = np.linspace(0.0, 1.0, NW)
    fpol = np.full(NW, B_TOROIDAL * R0)
    pressure = 1.0e4 * (1.0 - psin)
    ffprim = np.full(NW, 0.0)
    pprime = np.full(NW, -1.0e4 / (PSI_BOUNDARY - PSI_AXIS))
    qpsi = 1.0 + 2.0 * psin ** 2
    r_boundary, z_boundary = boundary()
    r_limiter = np.array([R_LEFT, R_LEFT + R_DIM, R_LEFT + R_DIM, R_LEFT,
                          R_LEFT])
    z_limiter = np.array([-Z_DIM / 2, -Z_DIM / 2, Z_DIM / 2, Z_DIM / 2,
                          -Z_DIM / 2])

    with open(path, "w") as out:
        # (a48, 3i4): a comment, then 0, nw, nh
        out.write("%-48s%4d%4d%4d\n" % ("pcms Solov'ev example", 0, NW, NH))
        out.write(numbers([R_DIM, Z_DIM, R0, R_LEFT, Z_MID]))
        out.write(numbers([R0, 0.0, PSI_AXIS, PSI_BOUNDARY, B_TOROIDAL]))
        out.write(numbers([CURRENT, PSI_AXIS, 0.0, R0, 0.0]))
        out.write(numbers([0.0, 0.0, PSI_BOUNDARY, 0.0, 0.0]))
        out.write(numbers(fpol))
        out.write(numbers(pressure))
        out.write(numbers(ffprim))
        out.write(numbers(pprime))
        out.write(numbers(psirz.T))  # R fastest
        out.write(numbers(qpsi))
        out.write("%5d%5d\n" % (len(r_boundary), len(r_limiter)))
        out.write(numbers(np.stack([r_boundary, z_boundary], axis=1)))
        out.write(numbers(np.stack([r_limiter, z_limiter], axis=1)))


if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else "solovev.eqdsk")
