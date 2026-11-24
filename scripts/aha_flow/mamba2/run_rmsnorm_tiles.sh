#!/bin/bash
# Mamba2 RMSNorms on rms_norm_fp (strait rms_norm_bf16) through RTL sim, X tile 0 of each (voyager_cgra tests):
#   block norm  layer_norm_default    64 x 2048 of arg0_1,      gamma _param_constant0   (16 X tiles)
#   norm_f      layer_norm_default_1  64 x 2048 of submodule_9, gamma _param_constant10  (16 X tiles)
#   gated norm  submodule_8           32 x 4096 of reshape_18 (= g), gamma _param_constant8 handed over with
#               CGRA_WEIGHT_NODE: submodule_8's first op (mul_18) carries no weight   (32 X tiles)
# Gold = voyager's layer output, X-tile sliced. After each run, the inputs the app loaded (hw_input_stencil.raw,
# hw_weight_stencil.raw) are checked against the voyager tensors. Passing runs are staged for chip runs and pushed to
# khaki (stage_chip_collateral.sh); every run's hw_output.txt is saved. A failing layer's bin/, collateral and RTL
# output are saved and the next layer still runs (the layers are independent). Failures are not debugged.
# Usage: [ONLY="block norm_f gated"] run_rmsnorm_tiles.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/2026-11-20/rmsnorm}
STATUS=$LOGDIR/rmsnorm.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=rms_norm_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

check_inputs() {  # name input_node gamma_node rows width
    $FLOW_PY -I -c "
import numpy as np
n = $4 * $5
raw = lambda f: np.fromfile('$APPS/$APP/' + f, dtype='>u2').astype(np.uint16)
bits = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
x = np.fromfile('$BASE/$2.bin', dtype=np.float32)[:n]
g = np.fromfile('$BASE/$3.bin', dtype=np.float32)
xi, gi = raw('hw_input_stencil.raw'), raw('hw_weight_stencil.raw')
print('  $1 inputs: hw_input_stencil.raw == rows 0..$4 of $2 (%d values): %s; hw_weight_stencil.raw == $3 (%d values): %s'
      % (xi.size, xi.size == n and np.array_equal(xi, bits(x)), gi.size, gi.size == g.size and np.array_equal(gi, bits(g))))
" >> $STATUS 2>&1
}

run_test() {  # name layer key input_node gamma_node rows width
    local name=$1 mu=mamba2_prefill-$2 app=$3_RV_E64_MB start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$mu::$app"'"]}'
    log "$name START $mu::$app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    check_inputs $name $4 $5 $6 $7
    mkdir -p $LOGDIR/$name
    if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
        cp -a $TB/hw_output.txt $LOGDIR/$name/ && log "  saved hw_output.txt -> $LOGDIR/$name"
    else
        log "  no RTL output from this run (failed before RTL sim)"
    fi
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$mu" "$app" "$APP" >> $STATUS 2>&1
    else
        cp -a $APPS/$APP/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$mu ] && cp -a /aha/voyager/compiled_collateral/$mu $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name FAILED (not debugged); continuing with the next layer"
    fi
}

ONLY=${ONLY:-"block norm_f gated"}
log "RMSNORM_TILES START ($ONLY)"
for which in $ONLY; do
    case $which in
        block)  run_test block_norm layer_norm_default   rms_norm_fp_mamba2_prefill_block_norm_kernel0 arg0_1      _param_constant0  64 2048 ;;
        norm_f) run_test norm_f     layer_norm_default_1 rms_norm_fp_mamba2_prefill_norm_f_kernel0     submodule_9 _param_constant10 64 2048 ;;
        gated)  run_test gated_norm submodule_8          rms_norm_fp_mamba2_prefill_gated_norm_kernel0 reshape_18  _param_constant8  32 4096 ;;
    esac
done
log "RMSNORM_TILES DONE"
