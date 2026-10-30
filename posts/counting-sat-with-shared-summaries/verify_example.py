"""Check the blog's toy example, not the paper's general algorithm.

Run with Python 3; no third-party packages are required.
All calculations use integers, including the interpolation numerator.
"""

from itertools import product


BITS = (0, 1)


def phi(a, z):
    return a + (1 - a) * z


def valid_b(c, d):
    return c + d - c * d


def exhaustive_h(t):
    return sum(
        phi(a, c + t * d) * phi(b, d + t * c) * valid_b(c, d)
        for a, b, c, d in product(BITS, repeat=4)
    )


direct_count = sum(
    bool((a or c) and (b or d) and (c or d))
    for a, b, c, d in product(BITS, repeat=4)
)

# The local multiplicities in group B.
b_sum = [
    sum(valid_b(c, d) for c, d in product(BITS, repeat=2) if c + d == s)
    for s in range(3)
]
b_difference = [
    sum(valid_b(c, d) for c, d in product(BITS, repeat=2) if c - d == r)
    for r in (-1, 0, 1)
]

# Group A has already been summed symbolically.
h1 = sum(b_sum[s] * (1 + s) ** 2 for s in range(3))
hm1 = sum(n * (1 - r * r) for r, n in zip((-1, 0, 1), b_difference))
dh1 = sum(b_sum[s] * s * (1 + s) for s in range(3))
numerator = 3 * h1 + hm1 - 2 * dh1
assert numerator % 4 == 0
recovered = numerator // 4

assert b_sum == [0, 2, 1]
assert (h1, hm1, dh1) == (17, 1, 10)
assert direct_count == recovered == 8
assert h1 == exhaustive_h(1)
assert hm1 == exhaustive_h(-1)
for t in range(-5, 6):
    assert exhaustive_h(t) == 8 + 8 * t + t * t

print(f"Direct model count: {direct_count}")
print(f"H(1) = {h1}; H(-1) = {hm1}; H'(1) = {dh1}")
print(f"Recovered count: {recovered}")
print("All exact-arithmetic checks passed.")
