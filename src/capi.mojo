"""Pairwise-alignment dynamic-programming kernels exposed to Python."""

comptime BytePtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime ScorePtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime Score32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]


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
