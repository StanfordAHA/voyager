#!/bin/bash
# Mamba2 B^T@X_end + decayed entering state epilogue (zircon_2d_psum_reduction_fp), chunk 1 (matmul_mx_261 + mul_13),
# through RTL sim. The add is too small for test.py's tolerance to see, so a GEMM-only run of K tile 1 is the reference
# for an exact check: fused == bf16(GEMM-only RTL output + residual), bit for bit.
#   1. host checks (check_btxend_epilogue.py): graph identities, gold hook, partial-sum inputs
#   2. step1_k1: GEMM only (zircon_2d_nop), K tile 1. A stepping stone, NOT chip collateral: kept in the logs only
#   3. step2_k0: fused, K tile 0 (full map + PnR); step2_k1: fused, K tile 1 (reuses step2_k0's mapping)
#   4. exact check on K tile 1
# Passing fused runs are staged for chip runs and pushed to khaki (stage_chip_collateral.sh). Every run's hw_output.txt
# is saved. On a failure: save the app's bin/, the voyager collateral and the RTL output of that run, then stop.
# Failures are not debugged.
# Usage: [RESUME=k0|k1|exact] run_btxend_epilogue.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/2026-11-20/btxend}
STATUS=$LOGDIR/btxend_epilogue.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
MU=mamba2_prefill-matmul_mx_261
NOP=zircon_2d_nop
PSUM=zircon_2d_psum_reduction_fp
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

save_run() {  # name halide_app start_epoch: keep this run's bin/, voyager collateral and RTL output in the logs
    local dest=$LOGDIR/$1
    mkdir -p $dest
    cp -a $APPS/$2/bin $dest/aha_bin 2>/dev/null && log "  saved aha bin -> $dest/aha_bin"
    [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $dest/voyager_compiled_collateral \
        && log "  saved voyager collateral -> $dest"
}

save_output() {  # name start_epoch
    mkdir -p $LOGDIR/$1
    if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $2 ]; then
        cp -a $TB/hw_output.txt $LOGDIR/$1/ && log "  saved hw_output.txt -> $LOGDIR/$1"
    else
        log "  no RTL output from this run (failed before RTL sim)"
    fi
}

run_test() {  # name halide_app app_test stage(0|1)
    local name=$1 happ=$2 app=$3 stage=$4 start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"external_mu_tests_fp":["'"$MU -> $app"'"]}'
    log "$name START $MU -> $app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    save_output $name $start
    if [ $rc -eq 0 ]; then
        if [ $stage -eq 1 ]; then
            bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$happ" >> $STATUS 2>&1
        else
            save_run $name $happ
        fi
    else
        save_run $name $happ
        log "STOPPED at $name"; exit $rc
    fi
}

log "BTXEND_EPILOGUE START${RESUME:+ (RESUME=$RESUME)}"
if [ -z "$RESUME" ]; then
    $FLOW_PY -I $TOOLS/check_btxend_epilogue.py > $LOGDIR/host_checks.log 2>&1 \
        || { log "host checks FAILED (host_checks.log)"; exit 1; }
    log "host checks PASS"
    run_test step1_k1 $NOP ${NOP}_post_mamba2_prefill_BtXend_kernel1_RV_E64_MB 0
fi
if [ -z "$RESUME" ] || [ "$RESUME" = k0 ]; then
    run_test step2_k0 $PSUM ${PSUM}_post_mamba2_prefill_BtXend_state_kernel0_RV_E64_MB 1
fi
if [ "$RESUME" != exact ]; then
    run_test step2_k1 $PSUM ${PSUM}_post_mamba2_prefill_BtXend_state_kernel1_RV_E64_MB 1
fi
$FLOW_PY -I $TOOLS/check_btxend_epilogue.py --exact $LOGDIR/step1_k1/hw_output.txt $LOGDIR/step2_k1/hw_output.txt 1 \
    > $LOGDIR/exact_check.log 2>&1 && log "exact check PASS" || log "exact check FAILED (exact_check.log)"
log "BTXEND_EPILOGUE DONE"
