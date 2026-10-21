#!/usr/bin/env python3
"""Write a uniformly scaled copy of a per-cell Manning field.

  scale_manning.py <in.bin> <alpha> <out.bin>

The field is a PETSc binary Vec (big-endian: int32 classid 1211214,
int32 n, then n float64). Scaling every cell by the same alpha moves
the field along the "global roughness scale" direction -- the most
identifiable direction of the calibration problem, and the natural
one-parameter baseline that a multi-class calibration has to beat.
"""
import struct, sys

src, alpha, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3]
b = open(src, "rb").read()
cid, n = struct.unpack_from(">ii", b, 0)
assert cid == 1211214, f"not a PETSc Vec (classid {cid})"
assert 8 + 8 * n == len(b), f"length mismatch: n={n}, file={len(b)}"
vals = list(struct.unpack_from(f">{n}d", b, 8))
out = struct.pack(">ii", cid, n) + struct.pack(f">{n}d", *[v * alpha for v in vals])
open(dst, "wb").write(out)
nz = [v for v in vals if v > 0]
print(f"{dst}: {n} cells, alpha={alpha}, n range "
      f"{min(nz)*alpha:.4f}..{max(nz)*alpha:.4f} (was {min(nz):.4f}..{max(nz):.4f})")
