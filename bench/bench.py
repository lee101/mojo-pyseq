"""Reproducible Mojo-kernel timings against an independent Python DP reference."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojo_pyseq import NeedlemanWunsch, SmithWaterman, levenshtein


def reference_affine(a: str, b: str, local: bool = False) -> int:
    neg = -10**12
    previous_m = [0 if local else neg] * (len(b) + 1)
    previous_i = [0 if local else neg] * (len(b) + 1)
    previous_d = [0 if local else neg] * (len(b) + 1)
    if not local:
        previous_m[0] = 0
        previous_d = [neg] + [-4 - (j - 1) for j in range(1, len(b) + 1)]
    best_score = 0 if local else neg
    for i, ca in enumerate(a, 1):
        current_m = [0 if local else neg]
        current_i = [0 if local else -4 - (i - 1)]
        current_d = [0 if local else neg]
        for j, cb in enumerate(b, 1):
            sub = 1 if ca == cb else -2
            mm = max(previous_m[j - 1], previous_i[j - 1], previous_d[j - 1]) + sub
            ii = max(previous_m[j] - 4, previous_i[j] - 1, previous_d[j] - 4)
            dd = max(current_m[-1] - 4, current_i[-1] - 4, current_d[-1] - 1)
            if local:
                mm, ii, dd = max(0, mm), max(0, ii), max(0, dd)
            current_m.append(mm); current_i.append(ii); current_d.append(dd)
            best_score = max(best_score, mm, ii, dd)
        previous_m, previous_i, previous_d = current_m, current_i, current_d
    return best_score if local else max(previous_m[-1], previous_i[-1], previous_d[-1])


def reference_levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def best(fn, repeat=3):
    result = math.inf
    for _ in range(repeat):
        start = time.perf_counter(); fn(); result = min(result, time.perf_counter() - start)
    return result


def row(name, ours, reference):
    ratio = reference / ours
    status = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
    def duration(value):
        return f"{value * 1e6:.2f} us" if value < 1e-3 else f"{value * 1e3:.2f} ms"
    print(f"| {name} | {duration(ours)} | {duration(reference)} | {status} |")


def main():
    a = ("ACGTTGCA" * 113)[:900]
    b = ("ACGTCGCA" * 113)[:900]
    small_a, small_b = "GATTACA", "GCATGCU"
    assert NeedlemanWunsch().align(a, b).score == reference_affine(a, b)
    local = SmithWaterman().align(a, b)
    assert (local[0].score if local else 0) == reference_affine(a, b, True)
    assert levenshtein(a, b) == reference_levenshtein(a, b)
    print(f"Machine: {platform.platform()} ({platform.processor() or 'unknown CPU'})")
    print("| case | mojo-pyseq | Python reference | result |")
    print("| --- | ---: | ---: | --- |")
    row("NeedlemanWunsch.align (7 x 7)", best(lambda: NeedlemanWunsch().align(small_a, small_b), 100), best(lambda: reference_affine(small_a, small_b), 100))
    row("SmithWaterman.align (7 x 7)", best(lambda: SmithWaterman().align(small_a, small_b), 100), best(lambda: reference_affine(small_a, small_b, True), 100))
    row("levenshtein (7 x 7)", best(lambda: levenshtein(small_a, small_b), 100), best(lambda: reference_levenshtein(small_a, small_b), 100))
    row("NeedlemanWunsch.align (900 x 900)", best(lambda: NeedlemanWunsch().align(a, b)), best(lambda: reference_affine(a, b)))
    row("SmithWaterman.align (900 x 900)", best(lambda: SmithWaterman().align(a, b)), best(lambda: reference_affine(a, b, True)))
    row("levenshtein (900 x 900)", best(lambda: levenshtein(a, b)), best(lambda: reference_levenshtein(a, b)))


if __name__ == "__main__":
    main()
