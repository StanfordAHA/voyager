#!/bin/bash
# Step 2 of the split-row a_cum cumsum on Zircon: the carry add on the CGRA (add_softplus_fp in ADD_SOFTPLUS_OP=add mode,
# strait elementwise_broadcast_add_softplus_bf16 op="add", 32 lanes, 512 half-rows x 128).
# Step 1 (run_cumsum.sh with KEY=row_prefix_sum_fp_mamba2_prefill_half_RV_E64_MB) scans each 256-wide row as two
# 128-wide half-rows. Here every half-row gets a per-row carry: 0 on even half-rows, the first half's total
# y1[2r, 127] on odd ones, so the output is the full cumsum (gold = voyager's cumsum, the same byte order).
# Inputs: input = step 1's live RTL output; carry = gathered on the host from it (data movement only; on the chip
# this would be a strided DMA read of lane 31's bank). Usage: STEP1_DIR=<step-1 log dir> run_cumsum_carry.sh
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/cumsum_carry}
STEP1_DIR=${STEP1_DIR:-/aha/rerun_logs/$(date +%F)/cumsum_half}
STATUS=$LOGDIR/cumsum_carry.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=add_softplus_fp
LIVE=$APPS/$APP/mamba2_prefill-cumsum_carry_live
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-cumsum
KEY=add_softplus_fp_mamba2_prefill_cumsum_carry_RV_E64_MB
FLOW_PY=/aha/bin/python
mkdir -p $LOGDIR $LIVE
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

PYHEAD="
import numpy as np, torch
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.int64)
f32 = lambda b: (np.asarray(b, dtype=np.int64).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
bits = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64).reshape(-1)
bf = lambda a: torch.tensor(np.asarray(a, dtype=np.float32)).to(torch.bfloat16).to(torch.float64).numpy()
gold = np.fromfile('$BASE/cumsum.bin', dtype=np.float32).astype(np.float64)
x = np.fromfile('$BASE/permute.bin', dtype=np.float32).astype(np.float64).reshape(-1, 256)
exact = np.cumsum(x, axis=1).reshape(-1)
def stats(name, out):
    rel, rex = np.abs(out - gold) / np.abs(gold), np.abs(out - exact) / np.abs(exact)
    close = np.isclose(out, gold, atol=1.5e-4, rtol=0.2)
    return (f'{name}: vs voyager cumsum: {100 * (out == gold).mean():.1f}% bit-exact, mean rel err {100 * rel.mean():.3f}%, max {100 * rel.max():.2f}%, '
            f'{100 * (1 - close.mean()):.2f}% outside test.py tolerance; vs exact: mean {100 * rex.mean():.3f}%, max {100 * rex.max():.2f}%')
"

if [ ! -f $STEP1_DIR/cumsum/hw_output.txt ]; then
    log "no step-1 RTL output in $STEP1_DIR/cumsum/hw_output.txt: step 2 needs it, stopping"; log "CUMSUM_CARRY DONE"; exit 1
fi
# live inputs from step 1's RTL output
$FLOW_PY -c "import sys; sys.path.insert(0, '/aha/voyager/scripts/aha_flow')
from parse_dnnLayer_tensors import hw_output_txt_to_raw
hw_output_txt_to_raw('$STEP1_DIR/cumsum/hw_output.txt', '$LIVE/input.raw')" >> $LOGDIR/make_live.log 2>&1
$FLOW_PY -I -c "$PYHEAD
y1 = raw('$LIVE/input.raw')
assert y1.size == 65536, y1.size
half = y1.reshape(512, 128)
carry = np.zeros(512, dtype=np.int64); carry[1::2] = half[0::2, 127]          # 0 on even half-rows, T1 on odd ones
carry.astype('>u2').tofile('$LIVE/carry.raw')
pred = bf(f32(half) + f32(carry)[:, None]).reshape(-1)
print('  live inputs: input.raw = step-1 RTL output (%d values), carry.raw = %d carries (%d nonzero)' % (y1.size, carry.size, int((carry != 0).sum())))
print('  ' + stats('predicted step 2 (bf16(step-1 RTL + carry))', pred))
" >> $STATUS 2>&1

start=$(date +%s)
config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$MU::$KEY"'"]}'
log "cumsum_carry START $MU::$KEY"
TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/cumsum_carry.log 2>&1
rc=$?
log "cumsum_carry EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/cumsum_carry.log | tail -1)"
log "  $(sed 's/\x1b\[[0-9;]*m//g' $LOGDIR/cumsum_carry.log | grep -a -E 'Floating point comparison|comparison failed|Interrupt wait timeout' | tail -1)"
$FLOW_PY -I -c "$PYHEAD
d = '$APPS/$APP'
a, b = raw(d + '/bin/input_host_stencil.raw'), raw(d + '/bin/bias_host_stencil.raw')
print('  cumsum_carry inputs: bin/input_host_stencil.raw == input.raw: %s; bin/bias_host_stencil.raw == carry.raw: %s'
      % (np.array_equal(a, raw('$LIVE/input.raw')), np.array_equal(b, raw('$LIVE/carry.raw'))))
" >> $STATUS 2>&1
mkdir -p $LOGDIR/cumsum_carry
if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
    cp -a $TB/hw_output.txt $LOGDIR/cumsum_carry/ && log "  saved hw_output.txt -> $LOGDIR/cumsum_carry"
    $FLOW_PY -I -c "$PYHEAD
out = f32(np.array([int(h, 16) for h in open('$LOGDIR/cumsum_carry/hw_output.txt').read().split()], dtype=np.int64)[:65536])
print('  ' + stats('RTL two-step a_cum', out))
" >> $STATUS 2>&1
else
    log "  no hw_output.txt from this run (no RTL output, or the simulation hung)"
fi
[ -d /aha/voyager/compiled_collateral/$MU/compare ] && cp -a /aha/voyager/compiled_collateral/$MU/compare $LOGDIR/cumsum_carry/
if [ $rc -eq 0 ]; then
    bash $TOOLS/stage_chip_collateral.sh "$MU" "$KEY" "$APP" </dev/null >> $STATUS 2>&1
else
    cp -a $APPS/$APP/bin $LOGDIR/cumsum_carry/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/cumsum_carry/aha_bin"
    [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/cumsum_carry/voyager_compiled_collateral \
        && log "  saved voyager collateral -> $LOGDIR/cumsum_carry"
    log "  cumsum_carry FAILED (not debugged)"
fi
log "CUMSUM_CARRY DONE"
