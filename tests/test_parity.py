"""Independent reference and published-vector tests for pairwise alignment."""

from __future__ import annotations

import random

import numpy as np
import pytest

from mojo_pyseq import NeedlemanWunsch, SmithWaterman, levenshtein
from mojo_pyseq.alignment import Scoring, _bytes, _mojo_matrices


def reference_score(a, b, match=1, mismatch=-2, gap_open=-4, gap_extend=-1, local=False):
    neg = -10**12
    mtx = np.full((len(a) + 1, len(b) + 1), neg, dtype=np.int64)
    ins = mtx.copy(); dele = mtx.copy()
    if local:
        mtx.fill(0); ins.fill(0); dele.fill(0)
    else:
        mtx[0, 0] = 0
        for i in range(1, len(a) + 1): ins[i, 0] = gap_open + (i - 1) * gap_extend
        for j in range(1, len(b) + 1): dele[0, j] = gap_open + (j - 1) * gap_extend
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            sub = match if a[i - 1] == b[j - 1] else mismatch
            mtx[i, j] = max(mtx[i - 1, j - 1], ins[i - 1, j - 1], dele[i - 1, j - 1]) + sub
            ins[i, j] = max(mtx[i - 1, j] + gap_open, ins[i - 1, j] + gap_extend, dele[i - 1, j] + gap_open)
            dele[i, j] = max(mtx[i, j - 1] + gap_open, ins[i, j - 1] + gap_open, dele[i, j - 1] + gap_extend)
            if local:
                mtx[i, j] = max(0, mtx[i, j]); ins[i, j] = max(0, ins[i, j]); dele[i, j] = max(0, dele[i, j])
    return int(max(mtx[-1, -1], ins[-1, -1], dele[-1, -1])) if not local else int(max(mtx.max(), ins.max(), dele.max()))


@pytest.mark.parametrize("a,b", [("", ""), ("A", ""), ("", "AC"), ("GATTACA", "GCATGCU")])
def test_global_known_scores(a, b):
    result = NeedlemanWunsch().align(a, b)
    assert result.score == reference_score(a, b)
    assert result.result_a.replace("-", "") == a
    assert result.result_b.replace("-", "") == b


def test_alignment_metadata_and_ascii_input_contract():
    result = NeedlemanWunsch().align("A", "")
    assert (result.pos_a, result.pos_b, result.len_a, result.len_b, result.length) == (0, 0, 1, 0, 1)
    with pytest.raises(ValueError, match="ASCII"):
        NeedlemanWunsch().align("é", "e")
    with pytest.raises(TypeError, match="str"):
        NeedlemanWunsch().align(["A"], "A")
    empty = _bytes("")
    assert empty.dtype == np.uint8 and empty.flags.c_contiguous and empty.ctypes.data != 0


def test_global_random_parity():
    rng = random.Random(3)
    for _ in range(80):
        a = "".join(rng.choice("ACGT") for _ in range(rng.randrange(12)))
        b = "".join(rng.choice("ACGT") for _ in range(rng.randrange(12)))
        ours = NeedlemanWunsch(match=3, mismatch=-2, gap_open=-5, gap_extend=-1).align(a, b)
        assert ours.score == reference_score(a, b, 3, -2, -5, -1)


def test_mojo_i32_fast_path_and_i64_overflow_fallback():
    matrices = _mojo_matrices("ACGT", "AGT", Scoring(), False)
    assert all(matrix.dtype == np.int32 for matrix in matrices)
    matrices = _mojo_matrices("AAA", "AAA", Scoring(match=500_000_000), False)
    assert all(matrix.dtype == np.int64 for matrix in matrices)
    assert NeedlemanWunsch(match=500_000_000).align("AAA", "AAA").score == 1_500_000_000


def test_scores_are_not_silently_narrowed_or_allowed_to_overflow():
    with pytest.raises(TypeError, match="integer"):
        NeedlemanWunsch(match=1.5)
    with pytest.raises(TypeError, match="integer"):
        NeedlemanWunsch(substitution_matrix={"A": {"A": 1.5}})
    with pytest.raises(ValueError, match="Int64"):
        NeedlemanWunsch(match=1 << 50).align("AA", "AA")


def test_local_published_smith_waterman_vector():
    score = SmithWaterman(match=2, mismatch=-1, gap_open=-1, gap_extend=-1).align("ACACACTA", "AGCACACA")[0]
    assert score.score == 12
    assert score.result_a.replace("-", "") == "ACACACTA"
    assert score.result_b.replace("-", "") == "AGCACACA"
    with pytest.raises(NotImplementedError, match="best"):
        SmithWaterman().align("A", "A", n=2)


def test_local_random_parity():
    rng = random.Random(4)
    aligner = SmithWaterman(match=2, mismatch=-3, gap_open=-3, gap_extend=-1)
    for _ in range(80):
        a = "".join(rng.choice("ACGT") for _ in range(rng.randrange(12)))
        b = "".join(rng.choice("ACGT") for _ in range(rng.randrange(12)))
        got = aligner.align(a, b)
        value = got[0].score if got else 0
        assert value == reference_score(a, b, 2, -3, -3, -1, local=True)


def test_affine_gap_and_substitution_matrix_fallback():
    matrix = {"A": {"A": 5, "C": -4}, "C": {"A": -4, "C": 5}}
    result = NeedlemanWunsch(substitution_matrix=matrix, gap_open=-3, gap_extend=-1).align("AAC", "AC")
    assert result.score == 7
    assert result.result_a.replace("-", "") == "AAC"
    assert result.result_b.replace("-", "") == "AC"


def test_no_end_gap_penalty_and_case_insensitive():
    assert NeedlemanWunsch(no_end_gap_penalty=True).align("ACGT", "AC").score == 2
    assert NeedlemanWunsch(no_start_gap_penalty=True).align("GGAC", "AC").score == 2
    assert NeedlemanWunsch(case_sensitive=False).align("ac", "AC").score == 2


def test_gap_and_mismatch_constraints_are_applied():
    assert NeedlemanWunsch(no_gaps_in_a=True).align("GGAC", "AC").score == -3
    assert NeedlemanWunsch(no_mismatches=True, gap_open=-1, gap_extend=-1).align("A", "C").score == -2


def test_impossible_gap_constraint_is_explicit():
    with pytest.raises(ValueError, match="no valid alignment"):
        NeedlemanWunsch(no_gaps_in_b=True).align("GGAC", "AC")


@pytest.mark.parametrize(("a", "b", "distance"), [("kitten", "sitting", 3), ("", "abc", 3), ("GATTACA", "GCATGCU", 4)])
def test_levenshtein(a, b, distance):
    assert levenshtein(a, b) == distance


def test_levenshtein_word_tail_and_boundary_parity():
    rng = random.Random(9)
    for pattern_len in (1, 7, 15, 31, 63, 64):
        a = "".join(rng.choice("ACGT") for _ in range(pattern_len))
        b = "".join(rng.choice("ACGT") for _ in range(137))
        assert levenshtein(a, b) == -NeedlemanWunsch(
            match=0, mismatch=-1, gap_open=-1, gap_extend=-1).align(a, b).score
        assert levenshtein(b, a) == levenshtein(a, b)


def test_levenshtein_long_pattern_fallback_parity():
    a = "ACGT" * 17
    b = "AGGT" * 18
    assert min(len(a), len(b)) > 64
    assert levenshtein(a, b) == -NeedlemanWunsch(
        match=0, mismatch=-1, gap_open=-1, gap_extend=-1).align(a, b).score


def test_upstream_pyseq_is_installed_but_is_not_an_alignment_api():
    pyseq = pytest.importorskip("pyseq")
    assert hasattr(pyseq, "Sequence")
    assert not hasattr(pyseq, "NeedlemanWunsch")
