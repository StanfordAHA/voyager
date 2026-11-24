#!/bin/bash
# Mamba2 gated RMSNorm (submodule_8) in two passes through RTL sim, X tile 0 (rows 0..63 of 1024) (voyager_cgra tests).
# The single-pass rms_norm_fp hangs at 4096 wide; the norm reduces all 4096 channels, so only the gamma multiply is K-tiled:
#   pass 1  rms_norm_pass1_fp, 32 lanes: x * rstd of reshape_18 (= g), 64 x 4096 (16 X tiles).
#           Gold = host bf16(x / sqrt(mean x^2 + 1e-5)) (RMS_NORM_GOLD_INPUT without a weight).
#   pass 2  rms_norm_pass2_fp, 16 lanes: * gamma (_param_constant8 via CGRA_WEIGHT_NODE) on one 16-of-32 channel half
#           (GOLD_CHANNEL_TRIMMING k = 0, 1), 64 x 2048. Input = pass 1's LIVE RTL output, trimmed to the half.
#           Gold = voyager submodule_8, X-sliced and trimmed.
# After each run the inputs the app loaded are checked; passing runs are staged for chip runs and pushed to khaki
# (stage_chip_collateral.sh); every run's hw_output.txt is saved. A failing run's bin/, collateral and RTL output are
# saved. Pass 2 needs pass 1's output, so a pass-1 failure stops the driver; a pass-2 failure goes on to the other half.
# Failures are not debugged.
# Usage: [ONLY="pass1 k0 k1"] run_gated_norm_two_pass.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/gated_norm_2pass}
STATUS=$LOGDIR/gated_norm_2pass.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
BASE=/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files
LIVE=$APPS/rms_norm_pass2_fp/mamba2_prefill-submodule_8_pass1_live
PB=$APPS/rms_norm_pass1_fp/path_balancing_configs/mamba2_prefill-submodule_8_path_balancing.json
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
MU=mamba2_prefill-submodule_8
ROWS=64
mkdir -p $LOGDIR $LIVE
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

# Shared host helpers: big-endian bf16 raw files, bf16 RNE bits, the 16-of-32 channel trim, the pass-1 host gold.
PYHEAD="
import numpy as np
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.uint16)
def bits(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
f32 = lambda b: (b.astype(np.uint32) << 16).view(np.float32)
trim = lambda a, k: a.reshape(-1, 2, 16)[:, k, :].reshape(-1)
x = np.fromfile('$BASE/reshape_18.bin', dtype=np.float32)[:$ROWS * 4096].reshape($ROWS, 4096)
gamma = np.fromfile('$BASE/_param_constant8.bin', dtype=np.float32)
xd = x.astype(np.float64)
host1 = bits(xd / np.sqrt((xd * xd).mean(1, keepdims=True) + 1e-5))
"

check_pass1() {  # inputs, path balancing, design.packed IDs
    $FLOW_PY -I -c "$PYHEAD
import json, re
xi = raw('$APPS/rms_norm_pass1_fp/hw_input_stencil.raw')
print('  pass1 inputs: hw_input_stencil.raw == bf16(reshape_18 rows 0..$((ROWS - 1))) (%d values): %s'
      % (xi.size, xi.size == x.size and np.array_equal(xi, bits(x).reshape(-1))))
pb = json.load(open('$PB'))
binpb = json.load(open('$APPS/rms_norm_pass1_fp/bin/path_balancing.json'))
print('  pass1 bin/path_balancing.json == the mamba2 file:', binpb == pb)
packed = dict((m.group(2), m.group(1)) for m in re.finditer(r'^(p\d+): (\S+)$', open('$APPS/rms_norm_pass1_fp/bin/design.packed').read(), re.M))
bad = [(n, i, packed.get(n)) for n, i in pb['name_to_id'].items() if packed.get(n) != i]
print('  pass1 design.packed IDs match name_to_id: %d/%d %s' % (len(pb['name_to_id']) - len(bad), len(pb['name_to_id']), bad[:4]))
" >> $STATUS 2>&1
}

make_live() {  # pass 1's RTL output -> the two trimmed pass-2 inputs; informational compare with the host x * rstd
    for k in 0 1; do
        GOLD_CHANNEL_TRIMMING_WORKAROUND=1 GOLD_CHANNEL_TRIMMING_WORKAROUND_NUM_KERNELS=2 GOLD_CHANNEL_TRIMMING_WORKAROUND_KERNEL_IDX=$k \
            $FLOW_PY -c "import sys; sys.path.insert(0, '/aha/voyager/scripts/aha_flow')
from parse_dnnLayer_tensors import hw_output_txt_to_raw
hw_output_txt_to_raw('$LOGDIR/pass1/hw_output.txt', '$LIVE/pass1_x0_k$k.raw')" >> $LOGDIR/make_live.log 2>&1 \
            || { log "  make_live k$k FAILED (see make_live.log)"; return 1; }
    done
    $FLOW_PY -I -c "$PYHEAD
for k in (0, 1):
    live, h = raw('$LIVE/pass1_x0_k%d.raw' % k), trim(host1.reshape(-1), k)
    if live.size != h.size:
        print('  live pass-2 input k%d: %d values, expected %d' % (k, live.size, h.size)); continue
    d = np.abs(live.astype(np.int32) - h.astype(np.int32))
    rel = np.abs(f32(live).astype(np.float64) / f32(h) - 1)
    print('  live pass-2 input k%d (%d values) vs host bf16(x * rstd): %.1f%% bit-exact, %.1f%% within 1 ulp, max rel err %.4f'
          % (k, live.size, 100 * (d == 0).mean(), 100 * (d <= 1).mean(), np.nanmax(rel)))
" >> $STATUS 2>&1
}

check_pass2() {  # k: the loaded input == the live file, the loaded gamma == the trim k of _param_constant8
    $FLOW_PY -I -c "$PYHEAD
k = $1
xi, live = raw('$APPS/rms_norm_pass2_fp/bin/input_host_stencil.raw'), raw('$LIVE/pass1_x0_k%d.raw' % k)
gi, gw = raw('$APPS/rms_norm_pass2_fp/hw_weight_stencil.raw'), raw('$APPS/rms_norm_pass2_fp/bin/weight_host_stencil.raw')
g = trim(bits(gamma), k)
print('  pass2 k%d inputs: bin/input_host_stencil.raw == pass1_x0_k%d.raw (%d values): %s; hw_weight_stencil.raw and bin/weight_host_stencil.raw == gamma trim %d (%d values): %s'
      % (k, k, xi.size, np.array_equal(xi, live), k, gi.size, np.array_equal(gi, g) and np.array_equal(gw, g)))
" >> $STATUS 2>&1
}

run_test() {  # name app_dir key  -> returns regress's exit code
    local name=$1 app_dir=$2 app=$3_RV_E64_MB start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$MU::$app"'"]}'
    log "$name START $MU::$app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    mkdir -p $LOGDIR/$name
    if [ -f $TB/hw_output.txt ] && [ $(stat -c %Y $TB/hw_output.txt) -ge $start ]; then
        cp -a $TB/hw_output.txt $LOGDIR/$name/ && log "  saved hw_output.txt -> $LOGDIR/$name"
    else
        log "  no RTL output from this run (failed before RTL sim)"
    fi
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$MU" "$app" "$app_dir" >> $STATUS 2>&1
    else
        cp -a $APPS/$app_dir/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$MU ] && cp -a /aha/voyager/compiled_collateral/$MU $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name FAILED (not debugged)"
    fi
    return $rc
}

ONLY=${ONLY:-"pass1 k0 k1"}
log "GATED_NORM_2PASS START ($ONLY)"
for which in $ONLY; do
    case $which in
        pass1)
            run_test pass1 rms_norm_pass1_fp rms_norm_pass1_fp_mamba2_prefill_gated_norm
            rc=$?
            check_pass1
            if [ $rc -ne 0 ] || [ ! -f $LOGDIR/pass1/hw_output.txt ]; then log "pass 2 needs pass 1's output: stopping"; break; fi
            make_live || { log "stopping"; break; } ;;
        k0|k1)
            k=${which#k}
            [ -f $LIVE/pass1_x0_k$k.raw ] || { log "pass2_k$k: no live input $LIVE/pass1_x0_k$k.raw, skipped"; continue; }
            run_test pass2_k$k rms_norm_pass2_fp rms_norm_pass2_fp_mamba2_prefill_gated_norm_kernel$k
            check_pass2 $k ;;
    esac
done
log "GATED_NORM_2PASS DONE"
