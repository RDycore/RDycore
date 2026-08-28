#!/bin/bash
# Arm o60: ship the script to Perlmutter and sbatch it, the moment PM is
# reachable again. Idempotent -- safe to run repeatedly; it refuses to
# submit twice.
#
#   bash plans/campaigns/o60_submit_when_up.sh
#
# Exit codes: 0 submitted (or already queued), 10 PM unreachable, 1 error.
set -u
PM=madams@perlmutter-p1.nersc.gov
KEY=$HOME/.ssh/nersc
SSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=25"
REMOTE=/pscratch/sd/m/madams/gpu-implicit
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. reachable? (255 = connection failure; empty output with rc 0 = dead cert)
OUT=$($SSH $PM 'echo ALIVE' 2>&1) || { echo "PM unreachable: $OUT"; exit 10; }
[ -n "$OUT" ] || { echo "PM returned nothing with rc 0 -- NERSC cert expired, renew it"; exit 10; }

# 2. already queued or already done?
if $SSH $PM "squeue -u madams -h -o '%j' | grep -q o60_ic_extend"; then
  echo "o60 already in the queue -- nothing to do"; exit 0
fi
if $SSH $PM "test -f $REMOTE/o60_ic0.6.log"; then
  echo "o60_ic0.6.log already exists on PM -- already ran"; exit 0
fi

# 3. ship and submit
scp -i $KEY -q "$HERE/o60_ic_extend.sh" $PM:$REMOTE/ || { echo "scp failed"; exit 1; }
$SSH $PM "cd $REMOTE && chmod +x o60_ic_extend.sh && sbatch o60_ic_extend.sh"
