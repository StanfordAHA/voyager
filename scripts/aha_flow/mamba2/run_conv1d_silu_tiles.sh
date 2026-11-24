#!/bin/bash
# Mamba2 causal depthwise conv1d (4 taps) + bias + SiLU (silu_1) on depthwise_conv1d_silu_fp (strait
# depthwise_conv1d_silu_bf16, 8 lanes = 8 channels per run) through RTL sim (voyager_cgra tests).
# Inputs from hardcoded tensor metadata (voyager/hardcoded_tensor_metadata/mamba2_prefill-silu_1_tensor_metadata.json):
# xBC = linear_mx_default_1 [1,1024,4352], weight _param_constant4 [4352,4], bias _param_constant5 [4352]; gold silu_1.
# Channel tile k of 544 (IO_INNERMOST_DIM_TILING; IO_INNERMOST_TILE_PARAMS slices the weight and bias the same way).
# After each run, the inputs the app loaded are checked against the tensors. Every run's hw_output.txt and golds are
# saved; a passing run is staged for chip runs and pushed to khaki, a failing run's bin/ and collateral are saved and
# the next tile still runs. Failures are not debugged.
# Usage: [TILES="0 543"] run_conv1d_silu_tiles.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/conv1d_silu}
STATUS=$LOGDIR/conv1d_silu.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=depthwise_conv1d_silu_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-silu_1
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

check_inputs() {  # name tile
    $FLOW_PY -I -c "
import numpy as np
k, d = $2, '$APPS/$APP'
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.int64)
bits = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64).reshape(-1)
x = np.fromfile('$BASE/linear_mx_default_1.bin', dtype=np.float32).reshape(1024, 4352)[:, 8 * k:8 * k + 8]
w = np.fromfile('$BASE/_param_constant4.bin', dtype=np.float32).reshape(4352, 4)[8 * k:8 * k + 8]
b = np.fromfile('$BASE/_param_constant5.bin', dtype=np.float32)[8 * k:8 * k + 8]
res = []
for name, want in (('input', x), ('weight', w), ('bias', b)):
    a, c = raw(f'{d}/hw_{name}_stencil.raw'), raw(f'{d}/bin/{name}_host_stencil.raw')
    res.append(f'{name} (app dir, bin/) == tile {k} ({want.size} values): {np.array_equal(a, bits(want))}, {np.array_equal(c, bits(want))}')
print('  $1 inputs: ' + '; '.join(res))
" >> $STATUS 2>&1
}

run_tile() {  # tile
    local k=$1 name=k$1 app=${APP}_mamba2_prefill_k$1_RV_E64_MB start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$MU::$app"'"]}'
    log "$name START $MU::$app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    log "  $(sed 's/\x1b\[[0-9;]*m//g' $LOGDIR/$name.log | grep -a -E 'Floating point comparison|comparison failed' | tail -1)"
    check_inputs $name $k
    mkdir -p $LOGDIR/$name
    if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
        cp -a $TB/hw_output.txt $LOGDIR/$name/ && log "  saved hw_output.txt -> $LOGDIR/$name"
    else
        log "  no RTL output from this run (failed before RTL sim)"
    fi
    [ -d /aha/voyager/compiled_collateral/$MU/compare ] && cp -a /aha/voyager/compiled_collateral/$MU/compare $LOGDIR/$name/
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$APP" </dev/null >> $STATUS 2>&1
    else
        cp -a $APPS/$APP/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name FAILED (not debugged); continuing"
    fi
}

TILES=${TILES:-"0 543"}
log "CONV1D_SILU START (tiles $TILES)"
for k in $TILES; do
    run_tile $k
done
log "CONV1D_SILU DONE"
