#!/bin/bash
# Mamba2 a_cum = cumsum(a, dim -1) on row_prefix_sum_fp (strait row_prefix_sum_bf16, 32 lanes, G = 8 groups per row)
# through RTL sim (voyager_cgra test). Input permute = a = dt * A [1,4,1,64,256], gold = voyager's cumsum. The whole
# layer is one run of 256 rows x 256.
# After the run, the inputs the app loaded are checked. The RTL output and golds are saved; a pass is staged for chip
# runs and pushed to khaki, a failure's bin/ and collateral are saved. Failures are not debugged.
# Usage: run_cumsum.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/cumsum}
STATUS=$LOGDIR/cumsum.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
APP=row_prefix_sum_fp
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
MU=mamba2_prefill-cumsum
KEY=${KEY:-row_prefix_sum_fp_mamba2_prefill_RV_E64_MB}   # _half: the 512 x 128 (G = 4) step-1 run
FLOW_PY=/aha/bin/python
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

start=$(date +%s)
config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$MU::$KEY"'"]}'
log "cumsum START $MU::$KEY"
TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/cumsum.log 2>&1
rc=$?
log "cumsum EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/cumsum.log | tail -1)"
log "  $(sed 's/\x1b\[[0-9;]*m//g' $LOGDIR/cumsum.log | grep -a -E 'Floating point comparison|comparison failed' | tail -1)"
$FLOW_PY -I -c "
import numpy as np
d = '$APPS/$APP'
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.int64)
want = (np.fromfile('$BASE/permute.bin', dtype=np.float32).view(np.uint32) >> 16).astype(np.int64)
a, b = raw(d + '/hw_input_stencil.raw'), raw(d + '/bin/input_host_stencil.raw')
print(f'  cumsum inputs: hw_input_stencil.raw, bin/input_host_stencil.raw == bf16(permute) ({want.size} values): {np.array_equal(a, want)}, {np.array_equal(b, want)}')
" >> $STATUS 2>&1
mkdir -p $LOGDIR/cumsum
if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
    cp -a $TB/hw_output.txt $LOGDIR/cumsum/ && log "  saved hw_output.txt -> $LOGDIR/cumsum"
else
    log "  no RTL output from this run (failed before RTL sim)"
fi
[ -d /aha/voyager/compiled_collateral/$MU/compare ] && cp -a /aha/voyager/compiled_collateral/$MU/compare $LOGDIR/cumsum/
if [ $rc -eq 0 ]; then
    bash $TOOLS/stage_chip_collateral.sh "$MU" "$KEY" "$APP" </dev/null >> $STATUS 2>&1
else
    cp -a $APPS/$APP/bin $LOGDIR/cumsum/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/cumsum/aha_bin"
    [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/cumsum/voyager_compiled_collateral \
        && log "  saved voyager collateral -> $LOGDIR/cumsum"
    log "  cumsum FAILED (not debugged)"
fi
log "CUMSUM DONE"
