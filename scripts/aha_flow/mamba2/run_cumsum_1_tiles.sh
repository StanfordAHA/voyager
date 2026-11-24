#!/bin/bash
# Mamba2 segment sum cumsum_1 = cumsum(mul_8, dim -2) on col_prefix_sum_fp (strait col_prefix_sum_bf16, 32 lanes, one
# fp_add + Pond ring per lane) through RTL sim (voyager_cgra tests). mul_8 [1,4,1,64,256,256] is flattened to
# [65536 rows x 256] (IO_FLATTEN_TO_3D) and cut into 64 row tiles of 4 whole 256 x 256 matrices (IO_DIM1_TILING);
# each matrix is scanned as 4 column slices (COL_PREFIX_SUM_COL_SPLITS 4: G = 2, 512 tokens per slice).
# After each run: the inputs the app loaded and the gold vs the tensors, and the RTL output vs the host model of the
# kernel (sequential bf16 adds down each column, expected bit-exact) and vs voyager's cumsum_1. Every run's
# hw_output.txt and golds are saved; a passing run is staged for chip runs and pushed to khaki, a failing run's bin/ and
# collateral are saved and the next tile still runs. Failures are not debugged.
# Usage: [TILES="0 63"] [KEY_SUFFIX=_unsplit] [STAGE=0] run_cumsum_1_tiles.sh   (run in tmux with SSH_AUTH_SOCK passed through)
# KEY_SUFFIX picks the key variant (e.g. _unsplit: no column slices); STAGE=0 skips chip staging (stepping-stone runs).
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/cumsum_1}
STATUS=$LOGDIR/cumsum_1.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=col_prefix_sum_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-cumsum_1
KEY_SUFFIX=${KEY_SUFFIX:-}
STAGE=${STAGE:-1}
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

PYHEAD="
import numpy as np
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.int64)
hexf = lambda f: np.array([int(h, 16) for h in open(f).read().split()], dtype=np.int64)
bits = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64).reshape(-1)
f32 = lambda b: (np.asarray(b, dtype=np.int64).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
def bf_rne(a):
    m, e = np.frexp(np.asarray(a, dtype=np.float64))
    return np.ldexp(np.rint(m * 256.0) / 256.0, e)
def tile(name, k):   # matrices 4k .. 4k+3 of [256 x 256] (b = chunk * 64 + head)
    return np.fromfile('$BASE/' + name + '.bin', dtype=np.float32).reshape(256, 256, 256)[4 * k:4 * k + 4]
def model(x):        # the kernel: y = bf16_RNE(y_up + x) down each column, from 0
    out, acc = np.empty(x.shape), np.zeros((x.shape[0], x.shape[2]))
    for i in range(x.shape[1]):
        acc = bf_rne(acc + x[:, i, :]); out[:, i, :] = acc
    return out
"

check_inputs() {  # name tile
    $FLOW_PY -I -c "$PYHEAD
k, d = $2, '$APPS/$APP'
x, g = tile('mul_8', k), tile('cumsum_1', k)
a, c = raw(d + '/hw_input_stencil.raw'), raw(d + '/bin/input_host_stencil.raw')
gf = '/aha/voyager/compiled_collateral/$MU/compare/gold_activation.txt'
import os
gold_ok = np.array_equal(hexf(gf)[:g.size], bits(g)) if os.path.exists(gf) else 'no gold file'
print(f'  $1 inputs (app dir, bin/) == mul_8 tile {k} ({x.size} values): {np.array_equal(a, bits(x))}, {np.array_equal(c, bits(x))}; '
      f'gold_activation.txt == cumsum_1 tile: {gold_ok}')
" >> $STATUS 2>&1
}

check_output() {  # name tile
    $FLOW_PY -I -c "$PYHEAD
k = $2
x, g = tile('mul_8', k).astype(np.float64), tile('cumsum_1', k).astype(np.float64)
m = model(x)
out = f32(hexf('$LOGDIR/$1/hw_output.txt')[:x.size]).reshape(x.shape)
ii, jj = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
s = np.broadcast_to(ii > jj, x.shape)
rel = np.abs(out[s] - g[s]) / np.abs(g[s])
print(f'  $1 RTL == host model (sequential bf16): {100 * np.mean(out == m):.2f}% of {x.size} values; '
      f'vs voyager cumsum_1 (strict lower): {100 * np.mean(out[s] == g[s]):.1f}% bit-exact, mean rel err {100 * rel.mean():.2f}%, max {100 * rel.max():.2f}%; '
      f'upper triangle zero: {np.all(out[~s] == 0)}')
" >> $STATUS 2>&1
}

run_tile() {  # tile
    local k=$1 name=k$1$KEY_SUFFIX app=${APP}_mamba2_prefill_k$1${KEY_SUFFIX}_RV_E64_MB start=$(date +%s)
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
    if [ $rc -eq 0 ] && [ "$STAGE" = 1 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$APP" </dev/null >> $STATUS 2>&1
    elif [ $rc -eq 0 ]; then
        log "  $name PASS, not staged (STAGE=0)"
    else
        cp -a $APPS/$APP/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name FAILED (not debugged); continuing"
    fi
}

TILES=${TILES:-"0 63"}
log "CUMSUM_1 START (tiles $TILES)"
for k in $TILES; do
    run_tile $k
done
log "CUMSUM_1 DONE"
