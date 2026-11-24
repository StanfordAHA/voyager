#!/bin/bash
# Mamba2 quantize_mx_default (MX quantization of the block-norm output layer_norm_default [1, 1024, 2048], blocks of 64
# along the channels) on yuchen's one-pass fused_e8m0_quant_single_IO (strait fused_reduction_quant_pack_bf16), X tile 0
# of 16 (64 rows x 2048 = 131,072 values = 4,096 passes per lane, half the template's FIFO-Pond limit), through RTL sim
# as a voyager_cgra test. Gold = voyager's quantize_mx_default int8 data and e8m0 scales, packed (PACKED_OUTPUTS) and
# compared as integers (SOFT_INTEGER_COMPARISON), as in Llama's apply_e8m0_scale_single_IO test.
# After the run, the input the app used (hw_input_stencil.raw and its bin/ copy) is checked against layer_norm_default
# rows 0-63. A passing run is staged for chip runs, pushed to khaki and re-verified; every hw_*output*.txt is saved.
# On a failure: save bin/, the voyager collateral and the RTL outputs, then stop. Failures are not debugged.
# Usage: run_quantize_mx_tile.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/2026-11-20/quantize_mx}
STATUS=$LOGDIR/quantize_mx.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=fused_e8m0_quant_single_IO
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-quantize_mx_default
KEY=${APP}_mamba2_prefill_quantize_mx_default_kernel0_RV_E64_MB
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

name=quantize_mx_default start=$(date +%s)
config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$MU::$KEY"'"]}'
log "QUANTIZE_MX_TILE START $MU::$KEY"
TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
rc=$?
log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
grep -a -E 'comparison|exceptions in|mismatch' $LOGDIR/$name.log | grep -a -v -E 'rv_comparison|Mux in|width mismatch' | sed 's/\x1b\[[0-9;]*m//g' | tail -4 \
    | while read -r line; do log "  $line"; done
$FLOW_PY -I -c "
import numpy as np
x = np.fromfile('$BASE/layer_norm_default.bin', dtype=np.float32)[:64 * 2048]
want = (np.ascontiguousarray(x).view(np.uint32) >> 16).astype(np.uint16)
for f in ('$APPS/$APP/hw_input_stencil.raw', '$APPS/$APP/bin/input_bf_act_host_stencil.raw'):
    try:
        raw = np.fromfile(f, dtype='>u2').astype(np.uint16)
        print('  input check %s: %d values, == layer_norm_default rows 0..63: %s' % (f.split('/apps/')[1], raw.size, raw.size == want.size and np.array_equal(raw, want)))
    except FileNotFoundError:
        print('  input check %s: MISSING' % f)
" >> $STATUS 2>&1
mkdir -p $LOGDIR/$name
for f in $TB/hw_*output*.txt; do
    [ -f "$f" ] && [ $(stat -c %Y "$f") -ge $start ] && cp -a "$f" $LOGDIR/$name/ && log "  saved $(basename $f)"
done
if [ $rc -eq 0 ]; then
    bash $TOOLS/stage_chip_collateral.sh "$MU" "$KEY" "$APP" >> $STATUS 2>&1
    bash $TOOLS/push_staged_to_khaki.sh "$MU,,,$KEY" >> $STATUS 2>&1
else
    cp -a $APPS/$APP/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
    [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
        && log "  saved voyager collateral -> $LOGDIR/$name"
    log "  $name FAILED (not debugged)"
fi
log "QUANTIZE_MX_TILE DONE"
