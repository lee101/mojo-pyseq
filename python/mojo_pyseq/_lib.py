"""ctypes loader for the Mojo alignment kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PYSEQ_LIB") or os.path.join(ROOT, "dist", "libmojo-pyseq.so")
I = ctypes.c_int64


def build(force: bool = False) -> str:
    sources = [os.path.join(ROOT, "src", "capi.mojo")]
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise RuntimeError("mojo is not on PATH; run through pixi")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run([mojo, "build", "--emit", "shared-lib", sources[0], "-o", LIB],
                          cwd=ROOT, text=True, capture_output=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        loaded = ctypes.CDLL(build())
        for name in ("mps_global_affine", "mps_local_affine",
                     "mps_global_affine_i32", "mps_local_affine_i32"):
            fn = getattr(loaded, name)
            fn.argtypes = [I] * 11
            fn.restype = None
        loaded.mps_levenshtein_word_u8.argtypes = [I] * 4
        loaded.mps_levenshtein_word_u8.restype = I
        _lib = loaded
    return _lib
