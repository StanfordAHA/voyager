#!/bin/bash
# Mamba2 score@x + Y_off * exp(a_cum) epilogue (zircon_2d_psum_reduction_fp) through RTL sim: chunk 1, head 0
# (matmul_mx_68; the app key's YOFF_EPILOGUE_CHUNK / YOFF_EPILOGUE_HEAD). The added block is LIVE: head 0's columns of
# the C@states epilogue's RTL output (YOFF_EPILOGUE_LIVE_TXT, written by run_cstates_epilogue.sh CHUNK=1); the gold is
# bf16(score@x + the HOST block). No GEMM-only run (the add changes ~62% of the outputs, so test.py sees a missing add).
#   1. host checks (check_scorex_epilogue.py --require-live): graph identities, gold hook, inputs, live vs host block
#   2. the fused run (full map + PnR), staged for chip runs and pushed to khaki (stage_chip_collateral.sh)
# Its hw_output.txt is saved in the logs. On a failure: save the app's bin/ and the voyager collateral, then stop.
# Failures are not debugged.
# Usage: run_scorex_epilogue.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=/aha/rerun_logs/2026-11-20/scorex
STATUS=$LOGDIR/scorex_epilogue.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
MU=mamba2_prefill-matmul_mx_68
APP=zircon_2d_psum_reduction_fp
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

run_test() {  # name app_test
    local name=$1 app=$2 start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"external_mu_tests_fp":["'"$MU -> $app"'"]}'
    log "$name START $MU -> $app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    mkdir -p $LOGDIR/$name
    if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
        cp -a $TB/hw_output.txt $LOGDIR/$name/ && log "  saved hw_output.txt -> $LOGDIR/$name"
    else
        log "  no RTL output from this run (failed before RTL sim)"
    fi
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$APP" >> $STATUS 2>&1
    else
        cp -a $APPS/$APP/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "STOPPED at $name"; exit $rc
    fi
}

log "SCOREX_EPILOGUE START"
$FLOW_PY -I $TOOLS/check_scorex_epilogue.py --require-live > $LOGDIR/host_checks.log 2>&1 \
    || { log "host checks FAILED (host_checks.log)"; exit 1; }
log "host checks PASS"
run_test chunk1_head0 ${APP}_post_mamba2_prefill_scorex_yoff_kernel0_RV_E64_MB
log "SCOREX_EPILOGUE DONE"
