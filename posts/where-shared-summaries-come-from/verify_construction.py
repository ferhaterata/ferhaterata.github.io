#!/usr/bin/env python3
"""Exact check of the four-coordinate construction in ETH Remark 4.4.

Run with Python 3; no third-party packages are needed. This checks the
illustrative construction, not the general algorithm or its runtime bound.
"""

from fractions import Fraction as Q
from itertools import product


def identity(n):
    return [[Q(i == j) for j in range(n)] for i in range(n)]


def permutation(indices):
    return [[Q(j == index) for j in range(len(indices))] for index in indices]


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(ar, bc)) for bc in zip(*b)] for ar in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    unit = identity(4)
    swap1 = permutation([1, 0, 3, 2])
    swap2 = permutation([2, 3, 0, 1])
    assert multiply(swap1, swap1) == unit
    assert multiply(swap2, swap2) == unit
    assert multiply(swap1, swap2) == multiply(swap2, swap1)

    # The paper's f(z)=1-2z, W=I-(Pi1+Pi2)/2, and M=Pi1*Pi2*f(W).
    w = add(unit, [[v / 2 for v in row] for row in add(swap1, swap2)], -1)
    f_w = add(unit, [[2 * v for v in row] for row in w], -1)
    matrix = multiply(multiply(swap1, swap2), f_w)
    assert matrix == add(add(swap1, swap2), multiply(swap1, swap2), -1)

    correction = add(unit, matrix, -1)
    measurement = [Q(1), Q(-1), Q(-1), Q(1)]
    assert correction == [[sign * v for v in measurement] for sign in measurement]

    # Derive row 0 via equation (4.10), with polynomials in ascending powers.
    # b is unavailable at t=2; c at t=1; d is available at both points.
    forbidden = {1: [2], 2: [1], 3: []}
    constant = correction[0].copy()
    linear = [Q(0)] * 4
    for coordinate, coefficient in enumerate(matrix[0]):
        if not coefficient:
            continue
        points = forbidden[coordinate]
        assert len(points) <= 1
        constant[coordinate] += coefficient
        if points:
            linear[coordinate] -= coefficient / points[0]
    assert constant == [1, 0, 0, 0]
    assert linear == [0, Q(-1, 2), -1, 0]

    fibers = {1: {}, 2: {}}
    remote_multiplicities = {}
    for a, c in product((0, 1), repeat=2):
        remote_multiplicities[a - c] = remote_multiplicities.get(a - c, 0) + 1
    assert remote_multiplicities == {-1: 1, 0: 2, 1: 1}

    for assignment in product((0, 1), repeat=4):
        a, b, c, d = assignment
        s = dot(measurement, assignment)
        for t in [Q(0), Q(1), Q(2), Q(3, 2), Q(-1)]:
            generated = dot(constant, assignment) + t * dot(linear, assignment)
            assert generated == a - t * b / 2 - t * c
            assert generated == s + (1 - t / 2) * b + (1 - t) * c - d
        at_one = a - Q(b, 2) - c
        at_two = a - b - 2 * c
        assert s == (a - c) + (d - b)
        assert s == (a - b) + (d - c)
        assert at_one == s + Q(b, 2) - d == (a - c) - Q(b, 2)
        assert at_two == s - c - d == (a - b) - 2 * c
        for t, key, value in [
            (1, (b, d, a - c), at_one),
            (2, (c, d, a - b), at_two),
        ]:
            if key in fibers[t]:
                assert fibers[t][key] == value
            fibers[t][key] = value

    print("Verified: prescribed matrix and one-dimensional correction.")
    print("Verified: cancellation rule gives L(t) = a - t*b/2 - t*c.")
    print("Verified: all 16 Boolean assignments and group-measurement fibers.")
    print("Remote measurement multiplicities: -1 -> 1, 0 -> 2, 1 -> 1.")


if __name__ == "__main__":
    main()
