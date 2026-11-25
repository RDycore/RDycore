#!/usr/bin/env bash
# Runs the implicit-solver checks for one libCEED backend and prints one PASS/FAIL line
# per check, plus a summary. Intended for GPU nodes (e.g. Perlmutter), where the ctest
# "ceed" configurations (which use -ceed /cpu/self) do not exercise the GPU.
#
# Run from <build>/driver/tests/swe_roe after building (ctest generates the inputs there):
#
#   bash <repo>/driver/tests/swe_roe/run_implicit_checks.sh <ceed-resource> "<launcher>" [extra options]
#
# <launcher> must end with the option that takes the number of ranks, e.g.
#   Perlmutter GPU: "srun -N 1 -c 32 -G 2 --gpu-bind=single:1 -n"   (2 ranks are used at most)
#   laptop:         "mpiexec -n"
# Extra options are passed to every run, e.g. "-dm_vec_type cuda -pc_type bjacobi".
#
# Each run's output is kept in implicit_checks_<backend>/ for inspection.

set -u

if [ $# -lt 2 ]; then
  sed -n '2,15p' "$0"
  exit 1
fi

CEED="$1"
LAUNCH="$2"
shift 2
EXTRA="$*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE=../../../CMakeCache.txt
H5DUMP=$(grep "^H5DUMP:" $CACHE | cut -d= -f2)
H5DIFF=$(grep "^H5DIFF:" $CACHE | cut -d= -f2)
MMS=../../rdycore_mms
RDY=../../rdycore
for f in $MMS $RDY mms_implicit_jacobian_check.yaml swe_implicit_dam_break_dry_none_np_2_ceed.yaml; do
  if [ ! -e "$f" ]; then
    echo "missing $f: run this script from <build>/driver/tests/swe_roe after building"
    exit 1
  fi
done

LOGDIR=implicit_checks_$(echo "$CEED" | tr '/' '_' | sed 's/^_//')
mkdir -p "$LOGDIR"
npass=0
nfail=0

report() {  # report <name> <status 0|1> <detail>
  if [ "$2" -eq 0 ]; then
    printf "PASS  %-58s %s\n" "$1" "$3"
    npass=$((npass + 1))
  else
    printf "FAIL  %-58s %s (log: %s/%s.log)\n" "$1" "$3" "$LOGDIR" "$1"
    nfail=$((nfail + 1))
  fi
}

newton_per_step() {  # average Newton iterations per step in a log
  awk '/Nonlinear solve converged/ {s += $NF; c++} END {if (c) printf "%.1f Newton/step", s / c}' "$1"
}

echo "Implicit solver checks: -ceed $CEED, launcher '$LAUNCH', extra '$EXTRA'"
echo

# 1. libCEED Jacobian vs host Jacobian (errors above 1e-12 relative)
for np in 1 2; do
  for scheme in roe hr augroe; do
    cfg=mms_implicit_jacobian_check
    [ $scheme != roe ] && cfg=${cfg}_$scheme
    name=jacobian_${scheme}_np$np
    log=$LOGDIR/$name.log
    $LAUNCH $np $MMS $cfg.yaml -ceed $CEED -rdy_jacobian_compare -snes_converged_reason $EXTRA > $log 2>&1
    st=$?
    worst=$(awk '/Jacobian check/ {v = $(NF - 3) + 0; if (v > m) m = v} END {printf "%.1e", m}' $log)
    [ $st -eq 0 ] && grep -q "Jacobian check" $log
    report $name $? "max rel diff $worst"
  done
done

# 2. MMS convergence: time only (rates 1, 2, 2) and space-time (rate ~0.9)
for m in beuler cn bdf; do
  name=mms_time_$m
  log=$LOGDIR/$name.log
  $LAUNCH 1 $MMS mms_implicit_time_$m.yaml -ceed $CEED -snes_converged_reason $EXTRA > $log 2>&1
  grep -q "^PASS" $log
  report $name $? "$(grep -E '^ +h :' $log | tail -1 | sed 's/ *h : /h rates: /')"
done
for wb in "" _hr _augroe; do
  name=mms_spacetime_beuler${wb:-_roe}
  log=$LOGDIR/$name.log
  $LAUNCH 2 $MMS mms_implicit_spacetime_beuler$wb.yaml -ceed $CEED -snes_converged_reason $EXTRA > $log 2>&1
  grep -q "^PASS" $log
  report $name $? "$(grep -E '^ +h :' $log | tail -1 | sed 's/ *h : /h rates: /'), $(newton_per_step $log)"
done

# 3. Dam breaks at dt = 0.1 s (front CFL ~2.5) vs Ritter / Stoker (rel. L1 <= 0.02, h >= 0)
for case in dry wet; do
  h_right=0
  [ $case = wet ] && h_right=0.1
  for wb in none hydrostatic_reconstruction augmented_roe; do
    t=swe_implicit_dam_break_${case}_${wb}_np_2_ceed
    name=dam_break_${case}_${wb}
    log=$LOGDIR/$name.log
    rm -rf output_$t
    $LAUNCH 2 $RDY $t.yaml -ceed $CEED -snes_converged_reason $EXTRA > $log 2>&1
    python3 "$SCRIPT_DIR/check_dam_break.py" $H5DUMP output_$t/$t-grid.h5 output_$t/$t-60.h5 "/60 6.000000E+00 sec" 6 1 $h_right 50 0.02 \
      >> $log 2>&1
    report $name $? "$(grep 'relative L1' $log | sed 's/relative L1 depth error/rel L1/; s/ (max 0.02)//'), $(newton_per_step $log)"
  done
done

# 4. Lake at rest preserved to 1e-12 (HR and augmented Roe, submerged and island)
for case in submerged island; do
  for wb in hydrostatic_reconstruction augmented_roe; do
    t=swe_implicit_lake_at_rest_tris_${case}_${wb}_np_2_ceed
    name=lake_at_rest_${case}_${wb}
    log=$LOGDIR/$name.log
    rm -rf output_$t
    $LAUNCH 2 $RDY $t.yaml -ceed $CEED -snes_converged_reason $EXTRA > $log 2>&1
    cmake -DH5DIFF=$H5DIFF -DTOL=1e-12 -DINITIAL=output_$t/$t-0.h5 -DFINAL=output_$t/$t-4.h5 "-DINITIAL_GROUP=/0 0.000000E+00 sec" \
      "-DFINAL_GROUP=/4 1.000000E+00 sec" -P "$SCRIPT_DIR/compare_lake_at_rest.cmake" >> $log 2>&1
    report $name $? "$(newton_per_step $log)"
  done
done

echo
echo "$npass passed, $nfail failed (logs in $LOGDIR/)"
[ $nfail -eq 0 ]
