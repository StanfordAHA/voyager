#!/bin/bash
# Mamba2 C@states x exp(a_cum) epilogue (zircon_2d_broadcast_mul_fp), K tiles 0 and 1, through RTL sim. The chunk is the
# app keys' EXP_EPILOGUE_CHUNK and the tests.py entries' layer, matmul_mx_(264 + CHUNK). Chunk 0 is all zeros (zero
# entering state), so CHUNK defaults to 1.
# The GEMM-only C@states run (zircon_2d_nop_post_mamba2_prefill_Cstates_kernel0, chunk 0) must already have passed: its
# bin/design_meta.json is the reference for the broadcast pattern check (data-independent). It is NOT chip collateral
# (its gold is the plain GEMM; the chip runs the fused epilogue), so its copy is kept in the logs only.
#   1. host checks (check_cstates_epilogue.py): broadcast pattern vs the MU output order, gold vs the graph, inputs
#   2. K tile 0 (full map + PnR, path-balanced), then K tile 1 (reuses K tile 0's mapping: SKIP CGRA MAP / PNR)
#   3. with REPLACE_MU (default mamba2_prefill-matmul_mx_264 for CHUNK 1): once both new entries are pushed and verified
#      on khaki, delete REPLACE_MU's entries of the same app tests locally and on khaki
# Every passing run is staged for chip runs and pushed to khaki (stage_chip_collateral.sh); every run's hw_output.txt is
# saved in the logs. On a failure: save the app's bin/ and the voyager collateral of that run, then stop. Failures are
# not debugged.
# Usage: [CHUNK=1] [ONLY_K0=1] [LOGDIR=...] [RESUME=k1|replace] run_cstates_epilogue.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHUNK=${CHUNK:-1}
MU=mamba2_prefill-matmul_mx_$((264 + CHUNK))
[ "$CHUNK" = 1 ] && REPLACE_MU=${REPLACE_MU-mamba2_prefill-matmul_mx_264}
LOGDIR=${LOGDIR:-/aha/rerun_logs/2026-11-20/cstates_chunk$CHUNK}
STATUS=$LOGDIR/cstates_epilogue.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
APP=zircon_2d_broadcast_mul_fp
KEYS="${APP}_post_mamba2_prefill_Cstates_exp_kernel0_RV_E64_MB ${APP}_post_mamba2_prefill_Cstates_exp_kernel1_RV_E64_MB"
[ -n "$ONLY_K0" ] && KEYS="${APP}_post_mamba2_prefill_Cstates_exp_kernel0_RV_E64_MB"   # ONLY_K0=1: K tile 0 only
STEP1_META=/aha/rerun_logs/2026-11-20/cstates/step1_k0/gemm_only_collateral/aha_bin/design_meta.json
REMOTE=mcoduoza@khaki.stanford.edu; REMOTE_DIR=/nobackup/mcoduoza/zircon_aha_flow_bins
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

save_output() {  # name start_epoch
    mkdir -p $LOGDIR/$1
    if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $2 ]; then
        cp -a $TB/hw_output.txt $LOGDIR/$1/ && log "  saved hw_output.txt -> $LOGDIR/$1"
    else
        log "  no RTL output from this run (failed before RTL sim)"
    fi
}

save_failure() {  # name
    local dest=$LOGDIR/failed_$1
    mkdir -p $dest
    cp -a $APPS/$APP/bin $dest/aha_bin 2>/dev/null && log "  saved aha bin -> $dest/aha_bin"
    [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $dest/voyager_compiled_collateral \
        && log "  saved voyager collateral -> $dest"
}

run_test() {  # name app_test
    local name=$1 app=$2 start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"external_mu_tests_fp":["'"$MU -> $app"'"]}'
    log "$name START $MU -> $app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    save_output $name $start
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$APP" >> $STATUS 2>&1
    else
        save_failure $name
        log "STOPPED at $name"; exit $rc
    fi
}

replace_old_entries() {  # verify the new entries on khaki, then delete REPLACE_MU's entries locally and on khaki
    local new="" old=""
    for app in $KEYS; do new="$new $MU,,,$app"; old="$old $REPLACE_MU,,,$app"; done
    bash $TOOLS/push_staged_to_khaki.sh $new >> $STATUS 2>&1 || { log "new entries not verified on khaki: old entries KEPT"; exit 2; }
    for name in $old; do
        rm -rf "/aha/aha_src_mamba/$name" "/aha/voyager_src_mamba/$name"
        ssh -o BatchMode=yes -o ConnectTimeout=20 $REMOTE "rm -rf '$REMOTE_DIR/aha_src_mamba/$name' '$REMOTE_DIR/voyager_src_mamba/$name'"
        left=$(ssh -o BatchMode=yes -o ConnectTimeout=20 $REMOTE "ls -d '$REMOTE_DIR/aha_src_mamba/$name' '$REMOTE_DIR/voyager_src_mamba/$name' 2>/dev/null | wc -l")
        [ "$left" = 0 ] && [ ! -e "/aha/aha_src_mamba/$name" ] && log "REPLACED: removed $name (local and khaki)" \
            || log "REMOVE FAILED: $name (khaki copies left: $left)"
    done
}

log "CSTATES_EPILOGUE START chunk $CHUNK ($MU)${RESUME:+ (RESUME=$RESUME)}"
key_chunk=$($FLOW_PY -I -c "import json; print(json.load(open('/aha/aha/util/application_parameters.json'))['${APP}_post_mamba2_prefill_Cstates_exp_kernel0']['default']['EXP_EPILOGUE_CHUNK'])")
[ "$key_chunk" = "$CHUNK" ] || { log "app keys' EXP_EPILOGUE_CHUNK $key_chunk != CHUNK $CHUNK"; exit 1; }
if [ -z "$RESUME" ]; then
    [ -f "$STEP1_META" ] || { log "Step 1 design_meta missing: $STEP1_META"; exit 1; }
    $FLOW_PY -I $TOOLS/check_cstates_epilogue.py "$STEP1_META" > $LOGDIR/host_checks.log 2>&1 \
        || { log "host checks FAILED (host_checks.log)"; exit 1; }
    log "host checks PASS"
    run_test step2_k0 ${APP}_post_mamba2_prefill_Cstates_exp_kernel0_RV_E64_MB
fi
if [ "$RESUME" != replace ] && [ -z "$ONLY_K0" ]; then
    run_test step2_k1 ${APP}_post_mamba2_prefill_Cstates_exp_kernel1_RV_E64_MB
fi
[ -n "$REPLACE_MU" ] && replace_old_entries
log "CSTATES_EPILOGUE DONE"
