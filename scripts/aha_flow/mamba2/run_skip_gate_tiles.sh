#!/bin/bash
# Mamba2 g = (Y + D * x) * silu(z) (submodule_7 + mul_17 fused) on mamba_skip_gate_fp (strait elementwise_skip_gate_bf16,
# 16 lanes) through RTL sim (voyager_cgra tests). Run k = heads 2k, 2k+1 x all 1024 positions = columns 128k .. 128k+127
# of x = reshape, silu(z) = silu and g = mul_17 (IO_INNERMOST_DIM_TILING over 32); Y = submodule_6 stays heads-first and
# the kernel's input DMA reads it in x's order (permute_5); D = _param_constant7 is loaded whole, read from head 2k.
# After each run: the 4 inputs the app loaded and the gold vs the tensors, and the RTL output vs the host model of the
# kernel (the PE's fp_mul rounding, exact RNE for the add) and vs voyager's mul_17. Every run's
# hw_output.txt and golds are saved; a passing run is staged for chip runs and pushed to khaki, a failing run's bin/ and
# collateral are saved and the next tile still runs. Failures are not debugged.
# Usage: [TILES="0 31"] run_skip_gate_tiles.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/skip_gate}
STATUS=$LOGDIR/skip_gate.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=mamba_skip_gate_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-mul_17
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

PYHEAD="
import numpy as np, os
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.int64) if os.path.exists(f) else None
hexf = lambda f: np.array([int(h, 16) for h in open(f).read().split()], dtype=np.int64)
bits = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64).reshape(-1)
f32 = lambda b: (np.asarray(b, dtype=np.int64).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
def bf_rne(a):
    m, e = np.frexp(np.asarray(a, dtype=np.float64))
    return np.ldexp(np.rint(m * 256.0) / 256.0, e)
def tensors(k):   # run k's x / silu slabs [1024 x 128], Y heads-first [4, 2, 256, 64], D [64], mul_17 slab
    cols = slice(128 * k, 128 * k + 128)
    x = np.fromfile('$BASE/reshape.bin', dtype=np.float32).reshape(1024, 4096)[:, cols]
    z = np.fromfile('$BASE/silu.bin', dtype=np.float32).reshape(1024, 4096)[:, cols]
    y = np.fromfile('$BASE/submodule_6.bin', dtype=np.float32).reshape(4, 64, 256, 64)[:, 2 * k:2 * k + 2]
    d = np.fromfile('$BASE/_param_constant7.bin', dtype=np.float32)
    g = np.fromfile('$BASE/mul_17.bin', dtype=np.float32).reshape(1024, 4096)[:, cols]
    return x, z, y, d, g
def pe_mul(a, b):  # Zircon PE fp_mul: round to nearest even from the product truncated to 3 bits below the bf16 LSB
    m, e = np.frexp(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))
    q = np.trunc(np.abs(m) * 256.0 * 8.0) / 8.0
    r = np.floor(q); f = q - r
    return np.ldexp(np.sign(m) * (r + ((f > 0.5) | ((f == 0.5) & (r % 2 == 1)))) / 256.0, e)
def model(k, x, z, y, d, mul):   # mul = pe_mul (the RTL) or exact RNE (voyager's rounding)
    yp = y.transpose(0, 2, 1, 3).reshape(1024, 128)            # permute_5 within the run's 2 heads
    dcol = np.repeat(d[2 * k:2 * k + 2], 64)[None, :]
    return mul(bf_rne(yp.astype(np.float64) + mul(x, dcol)), z)
"

check_inputs() {  # name tile
    $FLOW_PY -I -c "$PYHEAD
k, a = $2, '$APPS/$APP'
x, z, y, d, g = tensors(k)
ok = lambda f, t: (lambda r: r is not None and np.array_equal(r, bits(t)))(raw(f))
gf = '/aha/voyager/compiled_collateral/$MU/compare/gold_activation.txt'
print(f'  $1 inputs (app dir, bin/): x {ok(a + \"/hw_input_stencil.raw\", x)}, {ok(a + \"/bin/input_host_stencil.raw\", x)}; '
      f'D[2k:2k+2] {ok(a + \"/hw_weight_stencil.raw\", d[2 * k:2 * k + 2])}, {ok(a + \"/bin/weight_host_stencil.raw\", d[2 * k:2 * k + 2])}; '
      f'Y heads-first {ok(a + \"/hw_yheads_stencil.raw\", y)}, {ok(a + \"/bin/yheads_host_stencil.raw\", y)}; '
      f'silu(z) {ok(a + \"/hw_gate_stencil.raw\", z)}, {ok(a + \"/bin/gate_host_stencil.raw\", z)}; '
      f'gold_activation.txt == mul_17 slab: {np.array_equal(hexf(gf), bits(g)) if os.path.exists(gf) else \"no gold file\"}')
" >> $STATUS 2>&1
}

check_output() {  # name tile
    $FLOW_PY -I -c "$PYHEAD
k = $2
x, z, y, d, g = tensors(k)
m = model(k, x, z, y, d, pe_mul).reshape(-1)
m_rne = model(k, x, z, y, d, lambda a, b: bf_rne(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))).reshape(-1)
out = f32(hexf('$LOGDIR/$1/hw_output.txt')[:m.size])
gv = g.astype(np.float64).reshape(-1)
nz = gv != 0
rel = np.abs(out[nz] - gv[nz]) / np.abs(gv[nz])
print(f'  $1 RTL == host model (PE fp_mul rounding): {100 * np.mean(out == m):.3f}% of {m.size} (exact-RNE model {100 * np.mean(out == m_rne):.3f}%); vs voyager mul_17: '
      f'{100 * np.mean(out == gv):.3f}% bit-exact, mean rel err {100 * rel.mean():.4f}%, max {100 * rel.max():.4f}%')
" >> $STATUS 2>&1
}

run_tile() {  # tile
    local k=$1 name=k$1 app=${APP}_mamba2_prefill_k$1_RV_E64_MB start=$(date +%s)
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
        bash $TOOLS/push_staged_to_khaki.sh "$MU,,,$app" </dev/null >> $STATUS 2>&1
    else
        cp -a $APPS/$APP/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name FAILED (not debugged); continuing"
    fi
}

TILES=${TILES:-"0 31"}
log "SKIP_GATE START (tiles $TILES)"
for k in $TILES; do
    run_tile $k
done
log "SKIP_GATE DONE"
