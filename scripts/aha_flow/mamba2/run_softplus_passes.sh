#!/bin/bash
# Mamba2 softplus on add_softplus_fp (strait elementwise_broadcast_add_softplus_bf16, 8 lanes) as the 4 passes of
# yuchen's README: dt = submodule_4 [1,1024,64] (dt_proj + dt_bias, the bias added in the MU; ZERO_BIAS here) sits in the
# 32-lane GLB layout (IO_RESHAPE_LAST_DIM 32: rows of 32 lanes, lane = h mod 32); pass k reads lanes 8k..8k+7 = GLB tile
# 6+k (input x 12+2k, 13+2k; bias x 10+2k) and writes GLB tile 10+k (x 20+2k, 21+2k): a 64 x 256 run (IO_INNERMOST_DIM_TILING
# over 4). After each run: the inputs the app loaded and the gold vs the tensor slices, and the RTL output vs the earlier
# single-call RTL output re-sliced (elementwise, expected bit-exact) and vs voyager's softplus. A passing pass is staged for
# chip runs and pushed to khaki; a failing pass's bin/ and collateral are saved and the next pass still runs. Failures are
# not debugged.
# Usage: [PASSES="0 1 2 3"] run_softplus_passes.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/softplus_passes}
STATUS=$LOGDIR/softplus_passes.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=add_softplus_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-softplus
OLD_OUT=${OLD_OUT:-/aha/rerun_logs/2026-11-21/softplus/softplus/hw_output.txt}   # the single-call run (8,450 cycles)
FLOW_PY=/aha/bin/python
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

PYHEAD="
import numpy as np, os
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.int64)
hexf = lambda f: np.array([int(h, 16) for h in open(f).read().split()], dtype=np.int64)
bits = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64).reshape(-1)
f32 = lambda b: (np.asarray(b, dtype=np.int64).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
def lanes(a, k):     # pass k's slice of a [1024 x 64] tensor in the 32-lane layout: [2048 rows x lanes 8k..8k+7]
    return np.asarray(a).reshape(2048, 32)[:, 8 * k:8 * k + 8].reshape(-1)
x = np.fromfile('$BASE/submodule_4.bin', dtype=np.float32)
g = np.fromfile('$BASE/softplus.bin', dtype=np.float32)
"

check_inputs() {  # name pass
    $FLOW_PY -I -c "$PYHEAD
k, d = $2, '$APPS/$APP'
a, c, z = raw(d + '/hw_input_stencil.raw'), raw(d + '/bin/input_host_stencil.raw'), raw(d + '/bin/bias_host_stencil.raw')
gf = '/aha/voyager/compiled_collateral/$MU/compare/gold_activation.txt'
gold_ok = np.array_equal(hexf(gf), bits(lanes(g, k))) if os.path.exists(gf) else 'no gold file'
print(f'  $1 inputs (app dir, bin/) == submodule_4 lanes {8 * k}..{8 * k + 7} ({a.size} values): {np.array_equal(a, bits(lanes(x, k)))}, '
      f'{np.array_equal(c, bits(lanes(x, k)))}; bias all zero ({z.size} words): {bool(z.size == 64 and not z.any())}; '
      f'gold_activation.txt == softplus slice: {gold_ok}')
" >> $STATUS 2>&1
}

check_output() {  # name pass
    $FLOW_PY -I -c "$PYHEAD
k = $2
want = lanes(g, k).astype(np.float64)
out = f32(hexf('$LOGDIR/$1/hw_output.txt')[:want.size])
old = f32(lanes(hexf('$OLD_OUT')[:65536], k))
rel = np.abs(out - want) / np.abs(want)
print(f'  $1 RTL == single-call RTL re-sliced: {100 * np.mean(out == old):.2f}% of {want.size} values; '
      f'vs voyager softplus: {100 * np.mean(out == want):.1f}% bit-exact, mean rel err {100 * rel.mean():.2f}%, max {100 * rel.max():.2f}%')
" >> $STATUS 2>&1
}

run_pass() {  # pass
    local k=$1 name=pass$1 app=${APP}_mamba2_prefill_pass$1_RV_E64_MB start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$MU::$app"'"]}'
    log "$name START $MU::$app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    log "  $(sed 's/\x1b\[[0-9;]*m//g' $LOGDIR/$name.log | grep -a -E 'Floating point comparison|comparison failed|Interrupt wait timeout' | tail -1)"
    check_inputs $name $k
    mkdir -p $LOGDIR/$name
    if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
        cp -a $TB/hw_output.txt $LOGDIR/$name/ && log "  saved hw_output.txt -> $LOGDIR/$name"
        check_output $name $k
    else
        log "  no hw_output.txt from this run (no RTL output, or the simulation hung)"
    fi
    [ -d /aha/voyager/compiled_collateral/$MU/compare ] && cp -a /aha/voyager/compiled_collateral/$MU/compare $LOGDIR/$name/
    cp -a $APPS/$APP/bin/design_meta.json $LOGDIR/$name/ 2>/dev/null
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$APP" </dev/null >> $STATUS 2>&1
    else
        cp -a $APPS/$APP/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name FAILED (not debugged); continuing"
    fi
}

PASSES=${PASSES:-"0 1 2 3"}
log "SOFTPLUS_PASSES START (passes $PASSES)"
for k in $PASSES; do
    run_pass $k
done
log "SOFTPLUS_PASSES DONE"
