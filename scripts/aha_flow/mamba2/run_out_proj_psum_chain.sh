#!/bin/bash
# Run Mamba2 out_proj (mamba2_prefill-submodule_9) psum kernels START..END through RTL sim with aha regress,
# one kernel at a time. After each kernel passes:
#   1. its bin/ and voyager collateral are staged for chip runs and pushed to khaki (stage_chip_collateral.sh),
#      before the next regress run deletes them;
#   2. its RTL output becomes the next kernel's partial-sum input: parse_dnnLayer_tensors.py reads
#      <model>-<layer>_gold/kernel_<i-1>_output.txt.
# Stops at the first failing kernel; a failed push is reported but does not stop the chain.
# Usage: run_out_proj_psum_chain.sh [START] [END]   (run in tmux with SSH_AUTH_SOCK passed through)
START=${1:-0}; END=${2:-15}; LAST=15
MU_TEST=mamba2_prefill-submodule_9
APP=zircon_2d_psum_reduction_fp
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=/aha/rerun_logs/2026-11-19/out_proj_chain
CHAIN=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/$APP/${MU_TEST}_gold
mkdir -p $LOGDIR $CHAIN
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha
for i in $(seq $START $END); do
    app_test=${APP}_post_mamba2_prefill_out_projection_kernel${i}_RV_E64_MB
    config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"external_mu_tests_fp":["'"$MU_TEST -> $app_test"'"]}'
    echo "kernel $i START $(date '+%F %T')" >> $LOGDIR/chain.status
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/kernel_$i.log 2>&1
    rc=$?
    echo "kernel $i EXIT $rc $(date '+%F %T')" >> $LOGDIR/chain.status
    [ $rc -ne 0 ] && exit $rc
    bash $TOOLS/stage_chip_collateral.sh "$MU_TEST" "$app_test" "$APP" >> $LOGDIR/chain.status 2>&1
    if [ $i -lt $LAST ]; then
        cp /aha/garnet/tests/test_app/hw_output.txt $CHAIN/kernel_${i}_output.txt
    fi
done
echo "DONE $START..$END $(date '+%F %T')" >> $LOGDIR/chain.status
