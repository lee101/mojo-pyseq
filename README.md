# mojo-pyseq

`mojo-pyseq` is a pairwise sequence-alignment library whose dynamic-programming
core is compiled with Mojo and called from Python. It provides global
Needleman-Wunsch and local Smith-Waterman alignment with affine gap costs,
plus unit-cost Levenshtein distance.

There is an important naming collision: the PyPI project named
[`pyseq`](https://pypi.org/project/pyseq/) is installed for parity checks, but
it groups numbered file names such as `render.0001.exr`; it contains no
biological/text alignment API. This repository therefore implements the
sequence-alignment scope directly.

## Covered subset

| API | Coverage |
| --- | --- |
| `NeedlemanWunsch(...).align(a, b)` | Global affine-gap alignment; Mojo kernel for ordinary match/mismatch scoring. |
| `SmithWaterman(...).align(a, b, n=0 or 1)` | Best local affine-gap alignment in a one-element list. Other `n` values raise `NotImplementedError`. |
| `Scoring` | `match`, `mismatch`, substitution matrices, affine gaps, end/start-gap options, gap/mismatch constraints, and case sensitivity. Non-default scoring uses the compatible Python fallback. |
| `Alignment` | `result_a`, `result_b`, `score`, `pos_a`, `pos_b`, `len_a`, `len_b`, and `length`. |
| `levenshtein(a, b)` | Unit-cost edit distance. |

It does not implement file-sequence operations from the unrelated `pyseq`
project, multiple-sequence alignment, Unicode sequence input, or exhaustive
enumeration of every equally optimal Smith-Waterman traceback. The Mojo path
holds three full score matrices: `12 * (len(a) + 1) * (len(b) + 1)` bytes on
the `int32` fast path and `24 * (len(a) + 1) * (len(b) + 1)` bytes otherwise.
Use it for pairwise problems that fit in memory.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi` supplies the pinned Mojo nightly, NumPy, pytest, and the published
`pyseq` package. The Python binding will also rebuild `dist/libmojo-pyseq.so`
when `src/capi.mojo` is newer.

## Usage

```python
from mojo_pyseq import NeedlemanWunsch, SmithWaterman, levenshtein

global_alignment = NeedlemanWunsch().align("GATTACA", "GCATGCU")
print(global_alignment.result_a, global_alignment.result_b, global_alignment.score)
# GATTACA GCATGCU -5

local_alignment = SmithWaterman(
    match=2, mismatch=-1, gap_open=-1, gap_extend=-1
).align("ACACACTA", "AGCACACA")[0]
print(local_alignment.result_a, local_alignment.result_b, local_alignment.score)
# A-CACACTA AGCACAC-A 12

assert levenshtein("kitten", "sitting") == 3
```

## Benchmarks

Measured by `pixi run bench` on Linux 6.8.0-136-generic, x86_64. Times are the
best of three and compare the public API with an
independent pure-Python affine dynamic-programming recurrence on the same
900-by-900 ASCII sequences.

| case | mojo-pyseq | Python reference | result |
| --- | ---: | ---: | --- |
| NeedlemanWunsch.align (900 x 900) | 8.84 ms | 536.23 ms | 60.67x faster |
| SmithWaterman.align (900 x 900) | 14.18 ms | 583.87 ms | 41.17x faster |

Reproduce these numbers only through the flock-protected task:

```bash
pixi run bench
```

## How it works

`src/capi.mojo` is a single Mojo compilation unit containing the affine
recurrences. Python validates ASCII input and bounded integer scores, encodes
each sequence into a contiguous non-null `uint8` NumPy buffer, and allocates
three row-major score matrices (`match`, gap-in-B, and gap-in-A). Ordinary
scoring uses `int32` matrices; a conservative range check selects the `int64`
kernel for large scores. The ctypes boundary passes each buffer as an `Int`
address because
Mojo C exports cannot be parametric over pointer origin. The exported wrappers
rebuild `UnsafePointer[..., AnyOrigin[mut=True]]` values, write only to
caller-owned matrices, and make no allocations. Python performs the traceback,
which keeps the public objects idiomatic while leaving the quadratic,
compute-bound recurrence in Mojo.

The recurrence has a loop-carried dependency across each row. No SIMD,
parallel, or GPU path is included.

The test suite covers random-score parity against an independently written
reference, published Needleman-Wunsch/Smith-Waterman-style vectors, each
advertised scoring option, edit distance, input and integer-score validation,
and a check that the real PyPI `pyseq` package is present but has no alignment
API.
