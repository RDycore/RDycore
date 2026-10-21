#!/usr/bin/env python
"""Render the calibrated Manning map (.vtu from rdycore_adjoint) as a PNG.

Handles PETSc's raw-appended VTU layout: the XML header is parsed up to
<AppendedData>, and DataArrays are read from the raw blob by offset.

Usage: plot_manning_map.py <map.vtu> <out.png> [truth-split-x]
"""
import re
import struct
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

DTYPES = {"Float64": "f8", "Float32": "f4", "Int64": "i8", "Int32": "i4", "UInt8": "u1", "UInt64": "u8", "UInt32": "u4"}


def main():
    vtu, out_png = sys.argv[1], sys.argv[2]
    data = open(vtu, "rb").read()
    i = data.find(b"<AppendedData")
    header = data[:i].decode()
    blob = data[data.find(b"_", i) + 1 :]
    hsize = 8 if 'header_type="UInt64"' in header else 4
    hfmt = "<Q" if hsize == 8 else "<I"

    def arrays(section):
        sec = re.search(section + r".*?</", header, re.S).group(0)
        out = []
        for m in re.finditer(r'<DataArray type="(\w+)"(?:\s+Name="([^"]*)")?[^>]*offset="(\d+)"', sec):
            typ, name, off = m.group(1), m.group(2) or "", int(m.group(3))
            n = struct.unpack(hfmt, blob[off : off + hsize])[0]
            out.append((name, np.frombuffer(blob[off + hsize : off + hsize + n], dtype="<" + DTYPES[typ])))
        return out

    pts = arrays(r"<Points>")[0][1].reshape(-1, 3)
    cells = dict(arrays(r"<Cells>"))
    conn, offs = cells["connectivity"].astype(np.int64), cells["offsets"].astype(np.int64)
    field = next(a for name, a in arrays(r"<CellData>") if "manning" in name.lower())

    polys, start = [], 0
    for end in offs:
        polys.append(pts[conn[start:end], :2])
        start = end

    fig, ax = plt.subplots(figsize=(7.5, 6))
    pc = PolyCollection(polys, array=np.asarray(field, dtype=float), cmap="viridis", edgecolor="none")
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cb = fig.colorbar(pc, ax=ax, shrink=0.85)
    cb.set_label("calibrated Manning $n$")
    if len(sys.argv) > 3:
        ax.axvline(float(sys.argv[3]), color="w", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    n = np.asarray(field)
    print(f"cells={len(polys)}  n: min={n.min():.4f} max={n.max():.4f} mean={n.mean():.4f}")


if __name__ == "__main__":
    main()
