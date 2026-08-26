"""Pairwise-alignment dynamic-programming kernels exposed to Python."""

from std.memory import stack_allocation
from std.sys import simd_width_of as simdwidthof

comptime BytePtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime ScorePtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime Score32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]


def levenshtein_word_u8(pattern: BytePtr, pattern_len: Int,
                        text: BytePtr, text_len: Int) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var masks = stack_allocation[256, UInt64]()
    var zeros = SIMD[DType.uint64, W]()
    var vector_end = 256 - 256 % W
    for i in range(0, vector_end, W):
        masks.store(i, zeros)
    for i in range(vector_end, 256):
        masks[i] = 0
    for i in range(pattern_len):
        masks[Int(pattern[i])] |= UInt64(1) << UInt64(i)

    var positive = ~UInt64(0)
    var negative = UInt64(0)
    var top = UInt64(1) << UInt64(pattern_len - 1)
    var score = pattern_len
    for i in range(text_len):
        var matches = masks[Int(text[i])]
        var changed = (((matches & positive) + positive) ^ positive) | matches | negative
        var positive_change = negative | ~(changed | positive)
        var negative_change = changed & positive
        var shifted_positive = (positive_change << 1) | UInt64(1)
        positive = (negative_change << 1) | ~(changed | shifted_positive)
        negative = changed & shifted_positive
        if (positive_change & top) != 0:
            score += 1
        elif (negative_change & top) != 0:
            score -= 1
    return score


def mx3(a: Int64, b: Int64, c: Int64) -> Int64:
    var best = a
    if b > best:
        best = b
    if c > best:
        best = c
    return best


def mx3_i32(a: Int32, b: Int32, c: Int32) -> Int32:
    var best = a
    if b > best:
        best = b
    if c > best:
        best = c
    return best


def global_affine_i32(a: BytePtr, b: BytePtr, n: Int, m: Int, match_score: Int32,
                      mismatch: Int32, gap_open: Int32, gap_extend: Int32,
                      mat: Score32Ptr, ins: Score32Ptr, dele: Score32Ptr):
    var width = m + 1
    var neg = Int32(-1000000000)
    mat[0] = 0
    ins[0] = neg
    dele[0] = neg
    for i in range(1, n + 1):
        var k = i * width
        mat[k] = neg
        ins[k] = gap_open + Int32(i - 1) * gap_extend
        dele[k] = neg
    for j in range(1, m + 1):
        mat[j] = neg
        ins[j] = neg
        dele[j] = gap_open + Int32(j - 1) * gap_extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            var k = i * width + j
            var diag = (i - 1) * width + j - 1
            var up = (i - 1) * width + j
            var left = i * width + j - 1
            var sub = match_score if a[i - 1] == b[j - 1] else mismatch
            mat[k] = mx3_i32(mat[diag], ins[diag], dele[diag]) + sub
            ins[k] = mx3_i32(mat[up] + gap_open, ins[up] + gap_extend,
                             dele[up] + gap_open)
            dele[k] = mx3_i32(mat[left] + gap_open, ins[left] + gap_open,
                              dele[left] + gap_extend)


def local_affine_i32(a: BytePtr, b: BytePtr, n: Int, m: Int, match_score: Int32,
                     mismatch: Int32, gap_open: Int32, gap_extend: Int32,
                     mat: Score32Ptr, ins: Score32Ptr, dele: Score32Ptr):
    var width = m + 1
    for i in range(n + 1):
        var k = i * width
        mat[k] = 0
        ins[k] = 0
        dele[k] = 0
    for j in range(1, m + 1):
        mat[j] = 0
        ins[j] = 0
        dele[j] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            var k = i * width + j
            var diag = (i - 1) * width + j - 1
            var up = (i - 1) * width + j
            var left = i * width + j - 1
            var sub = match_score if a[i - 1] == b[j - 1] else mismatch
            var v = mx3_i32(mat[diag], ins[diag], dele[diag]) + sub
            mat[k] = v if v > 0 else 0
            v = mx3_i32(mat[up] + gap_open, ins[up] + gap_extend,
                        dele[up] + gap_open)
            ins[k] = v if v > 0 else 0
            v = mx3_i32(mat[left] + gap_open, ins[left] + gap_open,
                         dele[left] + gap_extend)
            dele[k] = v if v > 0 else 0


def global_affine(a: BytePtr, b: BytePtr, n: Int, m: Int, match_score: Int64,
                  mismatch: Int64, gap_open: Int64, gap_extend: Int64,
                  mat: ScorePtr, ins: ScorePtr, dele: ScorePtr):
    var width = m + 1
    var neg = Int64(-1125899906842624)
    mat[0] = 0
    ins[0] = neg
    dele[0] = neg
    for i in range(1, n + 1):
        var k = i * width
        mat[k] = neg
        ins[k] = gap_open + Int64(i - 1) * gap_extend
        dele[k] = neg
    for j in range(1, m + 1):
        mat[j] = neg
        ins[j] = neg
        dele[j] = gap_open + Int64(j - 1) * gap_extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            var k = i * width + j
            var diag = (i - 1) * width + j - 1
            var up = (i - 1) * width + j
            var left = i * width + j - 1
            var sub = match_score if a[i - 1] == b[j - 1] else mismatch
            mat[k] = mx3(mat[diag], ins[diag], dele[diag]) + sub
            ins[k] = mx3(mat[up] + gap_open, ins[up] + gap_extend,
                         dele[up] + gap_open)
            dele[k] = mx3(mat[left] + gap_open, ins[left] + gap_open,
                          dele[left] + gap_extend)


def local_affine(a: BytePtr, b: BytePtr, n: Int, m: Int, match_score: Int64,
                 mismatch: Int64, gap_open: Int64, gap_extend: Int64,
                 mat: ScorePtr, ins: ScorePtr, dele: ScorePtr):
    var width = m + 1
    for i in range(n + 1):
        var k = i * width
        mat[k] = 0
        ins[k] = 0
        dele[k] = 0
    for j in range(1, m + 1):
        mat[j] = 0
        ins[j] = 0
        dele[j] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            var k = i * width + j
            var diag = (i - 1) * width + j - 1
            var up = (i - 1) * width + j
            var left = i * width + j - 1
            var sub = match_score if a[i - 1] == b[j - 1] else mismatch
            var v = mx3(mat[diag], ins[diag], dele[diag]) + sub
            mat[k] = v if v > 0 else 0
            v = mx3(mat[up] + gap_open, ins[up] + gap_extend,
                    dele[up] + gap_open)
            ins[k] = v if v > 0 else 0
            v = mx3(mat[left] + gap_open, ins[left] + gap_open,
                    dele[left] + gap_extend)
            dele[k] = v if v > 0 else 0


@export("mps_global_affine_i32")
def mps_global_affine_i32(a: Int, b: Int, n: Int, m: Int, match_score: Int64,
                          mismatch: Int64, gap_open: Int64, gap_extend: Int64,
                          mat: Int, ins: Int, dele: Int) abi("C"):
    global_affine_i32(BytePtr(unsafe_from_address=a), BytePtr(unsafe_from_address=b), n, m,
                      Int32(match_score), Int32(mismatch), Int32(gap_open), Int32(gap_extend),
                      Score32Ptr(unsafe_from_address=mat), Score32Ptr(unsafe_from_address=ins),
                      Score32Ptr(unsafe_from_address=dele))


@export("mps_local_affine_i32")
def mps_local_affine_i32(a: Int, b: Int, n: Int, m: Int, match_score: Int64,
                         mismatch: Int64, gap_open: Int64, gap_extend: Int64,
                         mat: Int, ins: Int, dele: Int) abi("C"):
    local_affine_i32(BytePtr(unsafe_from_address=a), BytePtr(unsafe_from_address=b), n, m,
                     Int32(match_score), Int32(mismatch), Int32(gap_open), Int32(gap_extend),
                     Score32Ptr(unsafe_from_address=mat), Score32Ptr(unsafe_from_address=ins),
                     Score32Ptr(unsafe_from_address=dele))


@export("mps_global_affine")
def mps_global_affine(a: Int, b: Int, n: Int, m: Int, match_score: Int64,
                      mismatch: Int64, gap_open: Int64, gap_extend: Int64,
                      mat: Int, ins: Int, dele: Int) abi("C"):
    global_affine(BytePtr(unsafe_from_address=a), BytePtr(unsafe_from_address=b), n, m,
                  match_score, mismatch, gap_open, gap_extend,
                  ScorePtr(unsafe_from_address=mat), ScorePtr(unsafe_from_address=ins),
                  ScorePtr(unsafe_from_address=dele))


@export("mps_local_affine")
def mps_local_affine(a: Int, b: Int, n: Int, m: Int, match_score: Int64,
                     mismatch: Int64, gap_open: Int64, gap_extend: Int64,
                     mat: Int, ins: Int, dele: Int) abi("C"):
    local_affine(BytePtr(unsafe_from_address=a), BytePtr(unsafe_from_address=b), n, m,
                 match_score, mismatch, gap_open, gap_extend,
                 ScorePtr(unsafe_from_address=mat), ScorePtr(unsafe_from_address=ins),
                 ScorePtr(unsafe_from_address=dele))


@export("mps_levenshtein_word_u8")
def mps_levenshtein_word_u8(a: Int, n: Int, b: Int, m: Int) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    if n <= m:
        return levenshtein_word_u8(BytePtr(unsafe_from_address=a), n,
                                   BytePtr(unsafe_from_address=b), m)
    return levenshtein_word_u8(BytePtr(unsafe_from_address=b), m,
                               BytePtr(unsafe_from_address=a), n)
