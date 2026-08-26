"""Public alignment API compatible with the common ``pyseq-align`` surface."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Optional

import numpy as np

from ._lib import lib

# Leave room for arithmetic on unreachable DP states. _validate_problem keeps
# all reachable scores well inside Int64's range.
_NEG = -(1 << 50)
_MAX_TOTAL_SCORE = 1 << 40


def _integer(name: str, value: object) -> int:
    """Accept integer-like values without silently truncating floats."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Alignment:
    result_a: str
    result_b: str
    score: int
    pos_a: int
    pos_b: int
    len_a: int
    len_b: int

    @property
    def length(self) -> int:
        return len(self.result_a)


class Scoring:
    def __init__(
        self,
        match: int = 1,
        mismatch: int = -2,
        substitution_matrix: Optional[dict[str, dict[str, int]]] = None,
        gap_open: int = -4,
        gap_extend: int = -1,
        no_start_gap_penalty: bool = False,
        no_end_gap_penalty: bool = False,
        no_gaps_in_a: bool = False,
        no_gaps_in_b: bool = False,
        no_mismatches: bool = False,
        case_sensitive: bool = True,
    ):
        if no_mismatches and (no_gaps_in_a or no_gaps_in_b):
            raise ValueError("no_mismatches cannot be combined with no_gaps_in_a or no_gaps_in_b")
        if substitution_matrix:
            for x, row in substitution_matrix.items():
                if len(x) != 1 or any(len(y) != 1 for y in row):
                    raise ValueError("substitution_matrix keys must be one character")
        self.match = _integer("match", match)
        self.mismatch = _integer("mismatch", mismatch)
        if substitution_matrix is not None:
            substitution_matrix = {
                x: {y: _integer("substitution_matrix score", value) for y, value in row.items()}
                for x, row in substitution_matrix.items()
            }
        self.substitution_matrix = substitution_matrix
        self.gap_open = _integer("gap_open", gap_open)
        self.gap_extend = _integer("gap_extend", gap_extend)
        self.no_start_gap_penalty = bool(no_start_gap_penalty)
        self.no_end_gap_penalty = bool(no_end_gap_penalty)
        self.no_gaps_in_a, self.no_gaps_in_b = bool(no_gaps_in_a), bool(no_gaps_in_b)
        self.no_mismatches, self.case_sensitive = bool(no_mismatches), bool(case_sensitive)

    @property
    def use_match_mismatch(self) -> bool:
        return self.substitution_matrix is None

    def score_pair(self, a: str, b: str) -> int:
        if not self.case_sensitive:
            a, b = a.upper(), b.upper()
        if self.substitution_matrix is not None:
            return int(self.substitution_matrix.get(a, {}).get(b, self.mismatch))
        if a == b:
            return self.match
        return _NEG if self.no_mismatches else self.mismatch


def _bytes(sequence: str) -> np.ndarray:
    if not isinstance(sequence, str):
        raise TypeError("sequences must be str")
    try:
        # ctypes must never receive a null pointer, including for an empty
        # sequence.  The logical length remains zero, so this byte is not read.
        encoded = sequence.encode("ascii")
        return np.frombuffer(encoded if encoded else b"\0", dtype=np.uint8)
    except UnicodeEncodeError as exc:
        raise ValueError("Mojo kernels currently accept ASCII sequences") from exc


def _simple(scoring: Scoring) -> bool:
    return (scoring.substitution_matrix is None and scoring.case_sensitive and
            not scoring.no_start_gap_penalty and not scoring.no_end_gap_penalty and
            not scoring.no_gaps_in_a and not scoring.no_gaps_in_b and not scoring.no_mismatches)


def _validate_problem(a: str, b: str, scoring: Scoring, check_ascii: bool = True) -> None:
    """Reject scores that could overflow either DP implementation."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("sequences must be str")
    if check_ascii:
        try:
            a.encode("ascii")
            b.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("Mojo kernels currently accept ASCII sequences") from exc
    values = [scoring.match, scoring.mismatch, scoring.gap_open, scoring.gap_extend]
    if scoring.substitution_matrix is not None:
        values.extend(value for row in scoring.substitution_matrix.values() for value in row.values())
    largest = max(map(abs, values), default=0)
    if largest * (len(a) + len(b) + 1) > _MAX_TOTAL_SCORE:
        raise ValueError("scores and sequence lengths exceed the supported Int64 range")


def _use_i32(a: str, b: str, scoring: Scoring) -> bool:
    largest = max(abs(scoring.match), abs(scoring.mismatch),
                  abs(scoring.gap_open), abs(scoring.gap_extend))
    return largest * (len(a) + len(b) + 1) < 1 << 27


def _mojo_matrices(a: str, b: str, scoring: Scoring, local: bool):
    aa, bb = _bytes(a), _bytes(b)
    shape = (len(a) + 1, len(b) + 1)
    dtype = np.int32 if _use_i32(a, b, scoring) else np.int64
    storage = np.empty((3, *shape), dtype=dtype)
    mat, ins, dele = storage
    suffix = "_i32" if dtype == np.int32 else ""
    name = ("mps_local_affine" if local else "mps_global_affine") + suffix
    getattr(lib(), name)(int(aa.ctypes.data), int(bb.ctypes.data), len(a), len(b),
                         scoring.match, scoring.mismatch, scoring.gap_open, scoring.gap_extend,
                         int(mat.ctypes.data), int(ins.ctypes.data), int(dele.ctypes.data))
    return mat, ins, dele


def _python_matrices(a: str, b: str, s: Scoring, local: bool):
    n, m = len(a), len(b)
    mat = np.full((n + 1, m + 1), _NEG, dtype=np.int64)
    ins, dele = mat.copy(), mat.copy()
    if local:
        mat.fill(0); ins.fill(0); dele.fill(0)
    else:
        mat[0, 0] = 0
        if s.no_start_gap_penalty:
            mat[:, 0] = 0
            mat[0, :] = 0
        else:
            if not s.no_gaps_in_b:
                ins[1:, 0] = [s.gap_open + (i - 1) * s.gap_extend for i in range(1, n + 1)]
            if not s.no_gaps_in_a:
                dele[0, 1:] = [s.gap_open + (j - 1) * s.gap_extend for j in range(1, m + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = s.score_pair(a[i - 1], b[j - 1])
            mat[i, j] = max(mat[i - 1, j - 1], ins[i - 1, j - 1], dele[i - 1, j - 1]) + sub
            if not s.no_gaps_in_b:
                ins[i, j] = max(mat[i - 1, j] + s.gap_open,
                                ins[i - 1, j] + s.gap_extend,
                                dele[i - 1, j] + s.gap_open)
            if not s.no_gaps_in_a:
                dele[i, j] = max(mat[i, j - 1] + s.gap_open,
                                 ins[i, j - 1] + s.gap_open,
                                 dele[i, j - 1] + s.gap_extend)
            if local:
                mat[i, j] = max(0, mat[i, j]); ins[i, j] = max(0, ins[i, j]); dele[i, j] = max(0, dele[i, j])
    return mat, ins, dele


def _state_at(mat: np.ndarray, ins: np.ndarray, dele: np.ndarray, i: int, j: int):
    values = (int(mat[i, j]), int(ins[i, j]), int(dele[i, j]))
    state = max(range(3), key=values.__getitem__)
    return values[state], state


def _trace(a: str, b: str, s: Scoring, mat, ins, dele, local: bool, end: tuple[int, int]) -> Alignment:
    i, j = end
    score, state = _state_at(mat, ins, dele, i, j)
    end_a, end_b = i, j
    ra: list[str] = []
    rb: list[str] = []
    while i or j:
        value, state = _state_at(mat, ins, dele, i, j) if state is None else (int((mat, ins, dele)[state][i, j]), state)
        if local and value <= 0:
            break
        if not local and s.no_start_gap_penalty and (i == 0 or j == 0):
            break
        if state == 0:
            prev = (int(mat[i - 1, j - 1]), int(ins[i - 1, j - 1]), int(dele[i - 1, j - 1]))
            ra.append(a[i - 1]); rb.append(b[j - 1]); i -= 1; j -= 1
        elif state == 1:
            prev = (int(mat[i - 1, j]), int(ins[i - 1, j]), int(dele[i - 1, j]))
            ra.append(a[i - 1]); rb.append("-"); i -= 1
        else:
            prev = (int(mat[i, j - 1]), int(ins[i, j - 1]), int(dele[i, j - 1]))
            ra.append("-"); rb.append(b[j - 1]); j -= 1
        state = max(range(3), key=prev.__getitem__)
    return Alignment("".join(reversed(ra)), "".join(reversed(rb)), score, i, j, end_a - i, end_b - j)


class NeedlemanWunsch:
    def __init__(self, match: int = 1, mismatch: int = -2,
                 substitution_matrix: Optional[dict[str, dict[str, int]]] = None,
                 gap_open: int = -4, gap_extend: int = -1,
                 no_start_gap_penalty: bool = False, no_end_gap_penalty: bool = False,
                 no_gaps_in_a: bool = False, no_gaps_in_b: bool = False,
                 no_mismatches: bool = False, case_sensitive: bool = True):
        self.scoring = Scoring(match, mismatch, substitution_matrix, gap_open, gap_extend,
                               no_start_gap_penalty, no_end_gap_penalty, no_gaps_in_a,
                               no_gaps_in_b, no_mismatches, case_sensitive)

    def align(self, a: str, b: str) -> Alignment:
        s = self.scoring
        simple = _simple(s)
        _validate_problem(a, b, s, not simple)
        mat, ins, dele = _mojo_matrices(a, b, s, False) if simple else _python_matrices(a, b, s, False)
        n, m = len(a), len(b)
        if s.no_end_gap_penalty:
            candidates = [(i, m) for i in range(n + 1)] + [(n, j) for j in range(m + 1)]
            end = max(candidates, key=lambda ij: _state_at(mat, ins, dele, *ij)[0])
        else:
            end = (n, m)
        if _state_at(mat, ins, dele, *end)[0] <= _NEG // 2:
            raise ValueError("scoring constraints leave no valid alignment")
        return _trace(a, b, s, mat, ins, dele, False, end)


class SmithWaterman(NeedlemanWunsch):
    def align(self, a: str, b: str, n: int = 0) -> list[Alignment]:
        n = _integer("n", n)
        if n not in (0, 1):
            raise NotImplementedError("only the best Smith-Waterman alignment is supported")
        s = self.scoring
        simple = _simple(s)
        _validate_problem(a, b, s, not simple)
        mat, ins, dele = _mojo_matrices(a, b, s, True) if simple else _python_matrices(a, b, s, True)
        flat = np.maximum(np.maximum(mat, ins), dele)
        i, j = np.unravel_index(int(np.argmax(flat)), flat.shape)
        if int(flat[i, j]) == 0:
            return []
        return [_trace(a, b, s, mat, ins, dele, True, (int(i), int(j)))]


def levenshtein(a: str, b: str) -> int:
    """Return the unit-cost edit distance between two ASCII strings."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("sequences must be str")
    if min(len(a), len(b)) <= 64:
        aa, bb = _bytes(a), _bytes(b)
        return int(lib().mps_levenshtein_word_u8(
            int(aa.ctypes.data), len(a), int(bb.ctypes.data), len(b)))
    result = NeedlemanWunsch(match=0, mismatch=-1, gap_open=-1, gap_extend=-1).align(a, b)
    return -result.score
