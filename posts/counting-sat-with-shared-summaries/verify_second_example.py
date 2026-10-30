"""Check the second blog example using two different exact calculations.

1. Enumerate Boolean models directly.
2. Recover H(1+z) by grouped polynomial sums and inclusion-exclusion.

The second calculation illustrates the restricted-sum identity used for Taylor
recovery. It does not implement the paper's general matrix construction or its
subexponential-time algorithm. Run with Python 3; no dependencies are required.
"""

from itertools import combinations, product


# Variables: a=1, b=2, c=3, d=4, e=5, f=6, g=7.
CLAUSES = (
    (1, -2, 3),
    (-1, 5, 4),
    (2, -5, -3),
    (-2, 6, -4),
    (1, 2, 5),
    (-1, -5, 6),
    (3, -4, 7),
)
A_VARS = (1, 2, 5, 6)
BITS = (0, 1)


# A polynomial is a list of integer coefficients in ascending degree.
def add(p, q):
    result = [0] * max(len(p), len(q))
    for i, value in enumerate(p):
        result[i] += value
    for i, value in enumerate(q):
        result[i] += value
    return result


def scale(p, scalar):
    return [scalar * value for value in p]


def multiply(p, q):
    result = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            result[i + j] += a * b
    return result


def evaluate(p, x):
    result = 0
    for coefficient in reversed(p):
        result = result * x + coefficient
    return result


def subsets(items):
    return [
        frozenset(subset)
        for size in range(len(items) + 1)
        for subset in combinations(items, size)
    ]


def sum_group_a(summary, formal_ones):
    """Sum A, freezing the formal correction inputs to the indicator of U.

    Around t=1, c+t*d = s+z*d and d+t*c = s+z*c.
    The formal inputs replace d and c in those corrections, so only the
    summary s, not an assignment to B, is needed for this local sum.
    """
    total = [0]
    for bits in product(BITS, repeat=len(A_VARS)):
        values = {variable: [bit] for variable, bit in zip(A_VARS, bits)}
        values[3] = [summary, int(4 in formal_ones)]
        values[4] = [summary, int(3 in formal_ones)]
        term = [1]
        for clause in CLAUSES[:-1]:
            failure = [1]
            for literal in clause:
                value = values[abs(literal)]
                false_literal = add([1], scale(value, -1)) if literal > 0 else value
                failure = multiply(failure, false_literal)
            term = multiply(term, add([1], scale(failure, -1)))
        total = add(total, term)
    return total


def count_group_b(summary, pinned_ones):
    """Count B assignments with c+d=s, the last clause true, and E pinned."""
    return sum(
        1
        for c, d, g in product(BITS, repeat=3)
        if c + d == summary
        and (c or not d or g)
        and all({3: c, 4: d}[variable] == 1 for variable in pinned_ones)
    )


def grouped_taylor_polynomial():
    # Boolean multilinearization needs only the subsets of {c,d}: these are
    # the only variables in the formal corrections. Sum over U subset E.
    total = [0]
    for pinned in subsets((3, 4)):
        for formal_ones in subsets(tuple(sorted(pinned))):
            sign = (-1) ** (len(pinned) - len(formal_ones))
            for summary in range(3):
                total = add(
                    total,
                    scale(
                        sum_group_a(summary, formal_ones),
                        sign * count_group_b(summary, pinned),
                    ),
                )
    return total


def direct_boolean_counts():
    counts = {(c, d): 0 for c, d in product(BITS, repeat=2)}
    for bits in product(BITS, repeat=7):
        if all(
            any(bits[abs(literal) - 1] == int(literal > 0) for literal in clause)
            for clause in CLAUSES
        ):
            counts[(bits[2], bits[3])] += 1
    return counts


def direct_algebraic_sum(t):
    """Independent scalar evaluation over all 128 complete assignments."""
    total = 0
    for bits in product(BITS, repeat=7):
        values = dict(enumerate(bits, 1))
        c, d = values[3], values[4]
        values[3] = c + t * d
        values[4] = d + t * c
        term = 1
        for clause in CLAUSES[:-1]:
            failure = 1
            for literal in clause:
                value = values[abs(literal)]
                failure *= (1 - value) if literal > 0 else value
            term *= 1 - failure
        total += term * int(bool(c or not d or bits[6]))
    return total


if __name__ == "__main__":
    counts = direct_boolean_counts()
    taylor = grouped_taylor_polynomial()
    assert taylor == [30, -15, -8, 0, 0]
    assert counts == {(0, 0): 8, (0, 1): 7, (1, 0): 10, (1, 1): 12}
    assert sum(counts.values()) == evaluate(taylor, -1) == 37
    for t in range(-3, 4):
        assert evaluate(taylor, t - 1) == direct_algebraic_sum(t)

    print("Counts by (c,d):", counts)
    print("Direct Boolean model count:", sum(counts.values()))
    print("H(1+z) coefficients through degree four:", taylor)
    print("Recovered H(0):", evaluate(taylor, -1))
    print("Grouped and direct algebraic evaluations agree at seven values of t.")
    print("All exact-arithmetic checks passed.")
