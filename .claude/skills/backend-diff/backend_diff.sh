#!/bin/bash
# Differential gate: the PETSc operator and the libCEED operator must agree on
# the same mesh. Runs one case twice, same mesh and steps, once with
# -ceed /cpu/self and once without, and compares the final depth field.
#
# Usage: backend_diff.sh <build-dir> [case.yaml] [mesh.msh] [nranks]
#
# Defaults to the channel dam-break case on 3 ranks, which is where an adapted
# mesh has ghost cells interleaved and so where indexing bugs show up.
set -u
BUILD=${1:?usage: backend_diff.sh <build-dir> [case.yaml] [mesh.msh] [nranks]}
CASE=${2:-channel_wave_amr.yaml}
MESH=${3:-channel_100x4_tri.msh}
NP=${4:-3}

SRC=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
DRIVER="$BUILD/driver/rdycore_amr"
[ -x "$DRIVER" ] || { echo "no rdycore_amr in $BUILD -- build it first"; exit 2; }

# the launcher PETSc was configured with, split the way the top-level
# CMakeLists does; fall back to plain mpiexec
MPIEXEC=$(grep -m1 "^MPIEXEC_EXECUTABLE" "$BUILD/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
[ -n "$MPIEXEC" ] || MPIEXEC=$(command -v mpiexec)
[ -n "$MPIEXEC" ] || { echo "no mpiexec found"; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/rdy-backend-diff.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

for backend in basic ceed; do
  d="$WORK/$backend"; mkdir -p "$d"
  cp "$SRC/driver/tests/amr/$CASE" "$SRC/share/meshes/$MESH" "$d/" || exit 2
  case $backend in ceed) extra="-ceed /cpu/self";; *) extra="";; esac
  (cd "$d" && "$MPIEXEC" -n "$NP" "$DRIVER" "$CASE" \
      -refine_criterion jump -refine_at_start -refine_allow_coarsening \
      $extra > run.log 2>&1) \
    || { echo "  the $backend run failed; tail of its log:"; tail -5 "$d/run.log"; exit 1; }
done

python3 -I - "$WORK/basic" "$WORK/ceed" << 'EOF'
import sys, glob, os, h5py, numpy as np, xml.etree.ElementTree as ET
def depth(d):
    fs = sorted(glob.glob(os.path.join(d, 'output', '*.xmf')),
                key=lambda p: int(p.split('-')[-1].split('.')[0]))
    if not fs: raise SystemExit(f"  {d}: no output written")
    root = ET.parse(fs[-1]).getroot(); base = os.path.dirname(os.path.abspath(fs[-1]))
    ref = [a for a in root.findall('.//Attribute') if a.get('Name') == 'Height'][0]
    fn, ds = ref.find('DataItem').text.strip().split(':')
    with h5py.File(os.path.join(base, fn), 'r') as f:
        return f[ds][...]
a, b = depth(sys.argv[1]), depth(sys.argv[2])
print(f"  cells: basic {len(a)}, ceed {len(b)}")
if len(a) != len(b):
    print("  FAIL: the two backends produced DIFFERENT MESHES -- cannot compare values")
    raise SystemExit(1)
d = np.abs(a - b); rel = d.max() / max(np.abs(a).max(), 1e-300)
print(f"  max|dh| = {d.max():.3e}   relative = {rel:.3e}")
if rel < 1e-10:
    print("  PASS (machine precision)")
else:
    print("  FAIL: the operators disagree beyond summation order")
    raise SystemExit(1)
EOF
