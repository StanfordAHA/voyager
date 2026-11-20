#!/bin/bash
# Run one Mamba2 MU+CGRA test (no psum chain) through RTL sim with aha regress; if it passes, stage its
# bin/ and voyager collateral for chip runs and push them to khaki (stage_chip_collateral.sh).
# Usage: run_single_mu_test.sh <mu_test> <app key>_RV_E64_MB <halide app dir name> <log dir>
#   e.g. run_single_mu_test.sh mamba2_prefill-matmul_mx zircon_2d_nop_post_mamba2_prefill_CBt_RV_E64_MB \
#            zircon_2d_nop /aha/rerun_logs/2026-11-19/cbt
# Run in tmux with SSH_AUTH_SOCK passed through.
MU_TEST=$1; APP_TEST=$2; APP=$3; LOGDIR=$4
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha
config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"external_mu_tests_fp":["'"$MU_TEST -> $APP_TEST"'"]}'
echo "$APP_TEST START $(date '+%F %T')" >> $LOGDIR/run.status
TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/regress.log 2>&1
rc=$?
echo "$APP_TEST EXIT $rc $(date '+%F %T')" >> $LOGDIR/run.status
[ $rc -ne 0 ] && exit $rc
bash $TOOLS/stage_chip_collateral.sh "$MU_TEST" "$APP_TEST" "$APP" >> $LOGDIR/run.status 2>&1
echo "DONE $(date '+%F %T')" >> $LOGDIR/run.status
