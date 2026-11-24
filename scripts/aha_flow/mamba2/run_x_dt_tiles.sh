#!/bin/bash
# Mamba2 X = x * dt (mul_7, written heads-first = permute_1) on mamba_x_dt_fp (strait elementwise_mul_head_gather_bf16,
# 16 lanes) through RTL sim (voyager_cgra tests). Run k = 16 c + 2 hb + q: chunk c (256 positions), heads 8 hb .. 8 hb + 7
# (one dt GLB tile), lane half q of the 32-lane layout (channels p = 32 g + 16 q + j of each head, hold 2). dt is
# serialized by a get_interleave_mem gather tree; the output DMA writes the heads-first order.
# After each run: the inputs the app loaded and the gold vs the tensors, and the RTL output vs the host model (the PE's
# fp_mul rounding) and vs voyager's permute_1. Every run's hw_output.txt and gold are saved; a passing run is staged for
# chip runs and pushed to khaki, a failing run's bin/ and collateral are saved and the next run still goes. Failures
# are not debugged.
# Usage: [TILES="0 63"] run_x_dt_tiles.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/x_dt}
STATUS=$LOGDIR/x_dt.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=mamba_x_dt_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-permute_1
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
def tensors(k):   # run k's x half slab [256 x 256] (col 16 (2h + g) + j), dt [256 x 8], permute_1 half block [2048 x 32]
    c, hb, q = k // 16, (k // 2) % 8, k % 2
    x = np.fromfile('$BASE/reshape.bin', dtype=np.float32).reshape(1024, 4096)
    xh = x[256 * c:256 * c + 256, 512 * hb:512 * hb + 512].reshape(256, 16, 2, 16)[:, :, q, :].reshape(256, 256)
    dt = np.fromfile('$BASE/softplus.bin', dtype=np.float32).reshape(1024, 64)[256 * c:256 * c + 256, 8 * hb:8 * hb + 8]
    p1 = np.fromfile('$BASE/permute_1.bin', dtype=np.float32).reshape(4, 64, 256, 64)
    gold = p1[c, 8 * hb:8 * hb + 8].reshape(8, 256, 2, 2, 16)[:, :, :, q, :].reshape(2048, 32)
    return xh, dt, gold
def pe_mul(a, b):  # Zircon PE fp_mul: round to nearest even from the product truncated to 3 bits below the bf16 LSB
    m, e = np.frexp(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))
    q = np.trunc(np.abs(m) * 256.0 * 8.0) / 8.0
    r = np.floor(q); f = q - r
    return np.ldexp(np.sign(m) * (r + ((f > 0.5) | ((f == 0.5) & (r % 2 == 1)))) / 256.0, e)
def model(xh, dt, mul):   # heads-first X half: row h * 256 + i, col 16 g + j
    xv = xh.astype(np.float64).reshape(256, 8, 2, 16).transpose(1, 0, 2, 3)      # [h, i, g, j]
    dv = dt.astype(np.float64).T[:, :, None, None]                                # [h, i, 1, 1]
    return mul(xv, dv).reshape(2048, 32)
"

check_inputs() {  # name run
    $FLOW_PY -I -c "$PYHEAD
k, a = $2, '$APPS/$APP'
xh, dt, gold = tensors(k)
ok = lambda f, t: (lambda r: r is not None and np.array_equal(r, bits(t)))(raw(f))
gf = '/aha/voyager/compiled_collateral/$MU/compare/gold_activation.txt'
print(f'  $1 inputs (app dir, bin/): x half slab {ok(a + \"/hw_input_stencil.raw\", xh)}, {ok(a + \"/bin/input_host_stencil.raw\", xh)}; '
      f'dt {ok(a + \"/hw_dt_stencil.raw\", dt)}, {ok(a + \"/bin/dt_host_stencil.raw\", dt)}; '
      f'gold_activation.txt == permute_1 half block: {np.array_equal(hexf(gf), bits(gold)) if os.path.exists(gf) else \"no gold file\"}')
" >> $STATUS 2>&1
}

check_output() {  # name run
    $FLOW_PY -I -c "$PYHEAD
k = $2
xh, dt, gold = tensors(k)
m = model(xh, dt, pe_mul).reshape(-1)
m_rne = model(xh, dt, lambda a, b: bf_rne(a * b)).reshape(-1)
out = f32(hexf('$LOGDIR/$1/hw_output.txt')[:m.size])
gv = gold.astype(np.float64).reshape(-1)
nz = gv != 0
rel = np.abs(out[nz] - gv[nz]) / np.abs(gv[nz])
print(f'  $1 RTL == host model (PE fp_mul rounding): {100 * np.mean(out == m):.3f}% of {m.size} (exact-RNE model {100 * np.mean(out == m_rne):.3f}%); vs voyager permute_1: '
      f'{100 * np.mean(out == gv):.3f}% bit-exact, mean rel err {100 * rel.mean():.4f}%, max {100 * rel.max():.4f}%')
" >> $STATUS 2>&1
}

run_tile() {  # run
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

TILES=${TILES:-"0 63"}
log "X_DT START (runs $TILES)"
for k in $TILES; do
    run_tile $k
done
log "X_DT DONE"
