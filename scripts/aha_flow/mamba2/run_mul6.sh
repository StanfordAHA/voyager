#!/bin/bash
# Mamba2 a = dt * A (mul_6) on rms_norm_pass2_fp (strait elementwise_mul_bf16 input_x_weight_broadcast, 16 lanes: dt times
# a per-head weight A re-read for every row) through RTL sim (voyager_cgra test). dt = softplus [1,1024,64], A =
# _tensor_constant1 [64] (= -exp(A_log)), gold = voyager's mul_6; the whole layer is one run, with hardcoded tensor metadata.
# After the run: the inputs the app loaded and the gold vs the tensors, and the RTL output vs the host model bf16(dt * A)
# (one rounding, expected bit-exact) and vs voyager. The RTL output and golds are saved; a pass is staged for chip runs and
# pushed to khaki, a failure's bin/ and collateral are saved. Failures are not debugged.
# Usage: run_mul6.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/mul6}
STATUS=$LOGDIR/mul6.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=rms_norm_pass2_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-mul_6
KEY=rms_norm_pass2_fp_mamba2_prefill_mul6_RV_E64_MB
FLOW_PY=/aha/bin/python
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
dt = np.fromfile('$BASE/softplus.bin', dtype=np.float32).reshape(1024, 64)
A = np.fromfile('$BASE/_tensor_constant1.bin', dtype=np.float32)
g = np.fromfile('$BASE/mul_6.bin', dtype=np.float32)
"

start=$(date +%s)
config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$MU::$KEY"'"]}'
log "mul6 START $MU::$KEY"
TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/mul6.log 2>&1
rc=$?
log "mul6 EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/mul6.log | tail -1)"
log "  $(sed 's/\x1b\[[0-9;]*m//g' $LOGDIR/mul6.log | grep -a -E 'Floating point comparison|comparison failed|Interrupt wait timeout' | tail -1)"
$FLOW_PY -I -c "$PYHEAD
d = '$APPS/$APP'
a, b = raw(d + '/hw_input_stencil.raw'), raw(d + '/bin/input_host_stencil.raw')
w, v = raw(d + '/hw_weight_stencil.raw'), raw(d + '/bin/weight_host_stencil.raw')
gf = '/aha/voyager/compiled_collateral/$MU/compare/gold_activation.txt'
ok = lambda x, y: x is not None and np.array_equal(x, y)
print(f'  mul6 inputs: dt (app dir, bin/) == bits(softplus): {ok(a, bits(dt))}, {ok(b, bits(dt))}; '
      f'A (app dir, bin/) == bits(_tensor_constant1): {ok(w, bits(A))}, {ok(v, bits(A))}; '
      f'gold_activation.txt == bits(mul_6): {ok(hexf(gf), bits(g)) if os.path.exists(gf) else \"no gold file\"}')
" >> $STATUS 2>&1
mkdir -p $LOGDIR/mul6
if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
    cp -a $TB/hw_output.txt $LOGDIR/mul6/ && log "  saved hw_output.txt -> $LOGDIR/mul6"
    $FLOW_PY -I -c "$PYHEAD
model = bf_rne(dt.astype(np.float64) * A.astype(np.float64)[None, :]).reshape(-1)
out = f32(hexf('$LOGDIR/mul6/hw_output.txt')[:model.size])
gv = g.astype(np.float64)
nz = gv != 0
rel = np.abs(out[nz] - gv[nz]) / np.abs(gv[nz])
print(f'  mul6 RTL == host bf16(dt * A): {100 * np.mean(out == model):.2f}% of {model.size}; vs voyager mul_6: '
      f'{100 * np.mean(out == gv):.2f}% bit-exact, mean rel err {100 * rel.mean():.4f}%, max {100 * rel.max():.4f}%')
" >> $STATUS 2>&1
else
    log "  no hw_output.txt from this run (no RTL output, or the simulation hung)"
fi
[ -d /aha/voyager/compiled_collateral/$MU/compare ] && cp -a /aha/voyager/compiled_collateral/$MU/compare $LOGDIR/mul6/
cp -a $APPS/$APP/bin/design_meta.json $LOGDIR/mul6/ 2>/dev/null
if [ $rc -eq 0 ]; then
    bash $TOOLS/stage_chip_collateral.sh "$MU" "$KEY" "$APP" </dev/null >> $STATUS 2>&1
    bash $TOOLS/push_staged_to_khaki.sh "$MU,,,$KEY" </dev/null >> $STATUS 2>&1
else
    cp -a $APPS/$APP/bin $LOGDIR/mul6/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/mul6/aha_bin"
    [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/mul6/voyager_compiled_collateral \
        && log "  saved voyager collateral -> $LOGDIR/mul6"
    log "  mul6 FAILED (not debugged)"
fi
log "MUL6 DONE"
