#!/bin/bash
# Mamba2 dt_proj + dt_bias (submodule_4 = linear_mx_default_2 + add_5) through RTL sim on zircon_2d_psum_reduction_fp.
# dt_bias [64] is added in the MU by kernel 0 (zircon_2d_nop + CONV1_BIAS_HACK: the fused add_5's dt_bias becomes the GEMM's
# bias), so the softplus that follows can run bias-free. Kernels 1..3 are zircon_2d_psum_reduction_fp.
#   1. host golds (make_psum_golds.py submodule_4): bf16 running sums of bias + slices 0..p, the last vs voyager's gold
#   2. kernel 0 (nop app, full map + PnR): slice 0 + dt_bias in the MU; gold = host kernel 0
#   3. kernel 3 (psum app, full map + PnR): reads the HOST running sum through slice 2 (kernels 1-2 are not run);
#      gold = voyager's submodule_4 (GEMM + dt_bias)
# Passing runs are staged for chip runs and pushed to khaki (stage_chip_collateral.sh); every run's hw_output.txt is
# saved in the logs. On a failure: save the app's bin/ and the voyager collateral, then stop. Failures are not debugged.
# Usage: [RESUME=k3] run_dtproj_bias.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=/aha/rerun_logs/2026-11-20/dtproj
STATUS=$LOGDIR/dtproj_bias.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
MU=mamba2_prefill-submodule_4
APP=zircon_2d_psum_reduction_fp
NOP=zircon_2d_nop
ROOT=${APP}_post_mamba2_prefill_dt_projection
NOP_ROOT=${NOP}_post_mamba2_prefill_dt_projection
HOST_GOLD=$APPS/$APP/${MU}_host_gold
CHAIN=$APPS/$APP/${MU}_gold
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR $CHAIN
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

run_test() {  # name halide_app app_test
    local name=$1 happ=$2 app=$3 start=$(date +%s)
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
        bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$happ" >> $STATUS 2>&1
    else
        cp -a $APPS/$happ/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "STOPPED at $name"; exit $rc
    fi
}

log "DTPROJ_BIAS START${RESUME:+ (RESUME=$RESUME)}"
$FLOW_PY -I $TOOLS/make_psum_golds.py submodule_4 ${NOP_ROOT}_kernel0 $HOST_GOLD > $LOGDIR/host_golds.log 2>&1 \
    || { log "host golds FAILED (host_golds.log)"; exit 1; }
log "host golds: $(grep -a 'vs voyager gold' $LOGDIR/host_golds.log | sed 's/\x1b\[[0-9;]*m//g')"
cp $HOST_GOLD/kernel_2_output.txt $CHAIN/kernel_2_output.txt
log "chain kernel_2_output.txt = HOST running sum through slice 2 (kernels 1-2 not run)"
if [ "$RESUME" != k3 ]; then
    run_test k0 $NOP ${NOP_ROOT}_kernel0_RV_E64_MB
    log "  k0 MU bias: $(grep -a -o 'Bias offset: [0-9]*' $LOGDIR/k0.log | tail -1); tensor_metadata $(grep -o '"has_bias": [a-z]*\|"has_residual": [a-z]*' /aha/voyager_src_mamba/$MU,,,${NOP_ROOT}_kernel0_RV_E64_MB/tensor_metadata.json | tr '\n' ' ')"
    cp $LOGDIR/k0/hw_output.txt $CHAIN/kernel_0_output.txt
fi
run_test k3 $APP ${ROOT}_kernel3_RV_E64_MB
log "DTPROJ_BIAS DONE"
