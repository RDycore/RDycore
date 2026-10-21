#!/usr/bin/env python3
"""Scale a per-cell Manning field on selected land-cover classes only.

  scale_manning_classes.py <manning.bin> <class.bin> <alpha> <out.bin> <code> [code ...]

Both inputs are PETSc binary Vecs over the same cells (big-endian:
int32 classid 1211214, int32 n, then n float64); the class Vec holds
the NLCD code per cell. Cells whose code is in the given list are
scaled by alpha, the rest keep their prior value.

Purpose: the uniform-alpha scan bounds what a single global roughness
knob can do. This localizes that authority -- if scaling only the
developed classes reproduces the global result, the effective
dimensionality of the calibration is a handful of classes, not fifteen,
which is a structural fix for over-parameterized fitting rather than a
regularization-tuning one.
"""
import struct, sys
from collections import Counter

man_p, cls_p, alpha, out_p = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
codes = {int(c) for c in sys.argv[5:]}
assert codes, "give at least one NLCD code"


def readvec(p):
    b = open(p, "rb").read()
    cid, n = struct.unpack_from(">ii", b, 0)
    assert cid == 1211214, f"{p}: not a PETSc Vec (classid {cid})"
    assert 8 + 8 * n == len(b), f"{p}: length mismatch"
    return list(struct.unpack_from(f">{n}d", b, 8)), n


man, n1 = readvec(man_p)
cls, n2 = readvec(cls_p)
assert n1 == n2, f"cell-count mismatch: {n1} vs {n2}"

hit = 0
out = []
for v, c in zip(man, cls):
    if int(round(c)) in codes:
        out.append(v * alpha)
        hit += 1
    else:
        out.append(v)

open(out_p, "wb").write(struct.pack(">ii", 1211214, n1) + struct.pack(f">{n1}d", *out))
present = Counter(int(round(c)) for c in cls)
sel = sorted(codes & set(present))
print(f"{out_p}: alpha={alpha} on classes {sel} -> {hit}/{n1} cells "
      f"({100.0*hit/n1:.1f}% of domain); untouched classes keep the NLCD prior")
