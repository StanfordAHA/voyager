#!/bin/bash
# Mamba2 multiply kernels through RTL sim (voyager_cgra tests), two runs each:
#   x_end        #6 X_end = (x * dt) * decay_to_end (mul_11) on mamba_x_end_fp (16 lanes, dt + decay_to_end gather trees);
#                run k = 16 c + 2 hb + q (chunk, 8 heads, lane half), 64 runs per layer
#   state_decay  #7 decayed state = state * chunk decay (mul_13, the kernel of mul_12 .. mul_14) on mamba_state_decay_fp;
#                run k = 2 u + q (chunk-decay GLB tile = 16 heads, lane half), 8 runs per chunk boundary
#   mul10        #5 W = scores * decay (mul_10) on rms_norm_pass2_fp (scores re-read per head);
#                run k = 32 c + 2 hg + q (chunk, 4 heads, lane half), 128 runs per layer
#   decay_chain  #3 decay = exp(cumsum_rows(a * strict)) * lower (mul_8 -> cumsum_1 -> exp -> mul_9) on mamba_decay_chain_fp;
#                run k = 32 c + 16 t + 2 jl + q (chunk, mul_6 GLB tile + lane = 4 heads, lane half), 128 runs per layer
#   state_decay_lane  #7 with the chunk decays read through one lane of exp_2's output GLB tile 3 (no gather tree);
#                run k = 2 v + q (16 consecutive heads, lane half), 8 runs per chunk boundary
#   exp2         exp_2 = exp(a_cum) on mamba_exp_fp (32 lanes, z clamped at -127); one run per layer (the key ends in
#                "exp2": KEY "..._exp" + run "2")
#   transpose / permute / permute_2  the batched transposes on transpose_fp (strait transpose_bf16): decay_to_end
#                [4, 64, 256] -> [4, 256, 64] (32 passes), a [4, 256, 64] -> [4, 64, 256] from #1's 16 lanes (16 passes),
#                B [4, 256, 128] -> B^T (32 passes); pass k reads input lane k and writes the output rows y = k mod lanes
# After each run: the inputs the app loaded and the gold vs the tensors, and the RTL output vs the host model (the PE's
# fp_mul rounding) and vs voyager. Every run's hw_output.txt and gold are saved; a passing run is staged for chip runs
# and pushed to khaki, a failing run's bin/ and collateral are saved and the next run still goes. Failures are not
# debugged.
# Usage: [TILES="0 63"] run_mul_tiles.sh <kernel>   (run in tmux with SSH_AUTH_SOCK passed through)
KERNEL=$1
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case $KERNEL in
    x_end)       MU=mamba2_prefill-mul_11; APP=mamba_x_end_fp;       KEY=mamba_x_end_fp_mamba2_prefill_k;              DEF_TILES="0 63";  RAWS="input dt decay" ;;
    state_decay) MU=mamba2_prefill-mul_13; APP=mamba_state_decay_fp; KEY=mamba_state_decay_fp_mamba2_prefill_mul13_k; DEF_TILES="0 7";   RAWS="input decay" ;;
    mul10)       MU=mamba2_prefill-mul_10; APP=rms_norm_pass2_fp;    KEY=rms_norm_pass2_fp_mamba2_prefill_mul10_k;     DEF_TILES="0 127"; RAWS="input weight" ;;
    decay_chain) MU=mamba2_prefill-mul_9;  APP=mamba_decay_chain_fp; KEY=mamba_decay_chain_fp_mamba2_prefill_k;        DEF_TILES="0 127"; RAWS="a mask_strict mask_lower" ;;
    state_decay_lane) MU=mamba2_prefill-mul_13; APP=mamba_state_decay_fp; KEY=mamba_state_decay_fp_mamba2_prefill_mul13_lane_k; DEF_TILES="0 7"; RAWS="input decay" ;;
    exp2)        MU=mamba2_prefill-exp_2;  APP=mamba_exp_fp;         KEY=mamba_exp_fp_mamba2_prefill_exp;              DEF_TILES="2";     RAWS="input" ;;
    transpose)   MU=mamba2_prefill-transpose; APP=transpose_fp;     KEY=transpose_fp_mamba2_prefill_transpose_k;      DEF_TILES="0 31";  RAWS="input" ;;
    permute)     MU=mamba2_prefill-permute;   APP=transpose_fp;     KEY=transpose_fp_mamba2_prefill_permute_k;        DEF_TILES="0 15";  RAWS="input" ;;
    permute_2)   MU=mamba2_prefill-permute_2; APP=transpose_fp;     KEY=transpose_fp_mamba2_prefill_permute_2_k;      DEF_TILES="0 31";  RAWS="input" ;;
    *) echo "usage: run_mul_tiles.sh x_end|state_decay|mul10|decay_chain|exp2|state_decay_lane|transpose|permute|permute_2"; exit 1 ;;
esac
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/mul_$KERNEL}
STATUS=$LOGDIR/$KERNEL.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

PYHEAD="
import numpy as np, os, sys
sys.path.insert(0, '$TOOLS')
from mul_kernels_lib import tensors, model
from check_x_dt_host_lib import bits, bf_rne, pe_mul
raw = lambda f: np.fromfile(f, dtype='>u2').astype(np.int64) if os.path.exists(f) else None
hexf = lambda f: np.array([int(h, 16) for h in open(f).read().split()], dtype=np.int64)
f32 = lambda b: (np.asarray(b, dtype=np.int64).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
"

check_inputs() {  # name run
    $FLOW_PY -I -c "$PYHEAD
raws, gold = tensors('$KERNEL', $2)
a = '$APPS/$APP'
ok = lambda f, t: (lambda r: r is not None and np.array_equal(r, bits(t).reshape(-1)))(raw(f))
gf = '/aha/voyager/compiled_collateral/$MU/compare/gold_activation.txt'
parts = [f'{n} {ok(a + \"/hw_\" + n + \"_stencil.raw\", t)}, {ok(a + \"/bin/\" + n + \"_host_stencil.raw\", t)}' for n, t in raws.items()]
print(f'  $1 inputs (app dir, bin/): ' + '; '.join(parts) +
      f'; gold_activation.txt == voyager slice: {np.array_equal(hexf(gf), bits(gold).reshape(-1)) if os.path.exists(gf) else \"no gold file\"}')
" >> $STATUS 2>&1
}

check_output() {  # name run
    $FLOW_PY -I -c "$PYHEAD
raws, gold = tensors('$KERNEL', $2)
m = model('$KERNEL', raws, pe_mul, $2).reshape(-1)
m_rne = model('$KERNEL', raws, lambda a, b: bf_rne(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)), $2).reshape(-1)
out = f32(hexf('$LOGDIR/$1/hw_output.txt')[:m.size])
gv = gold.astype(np.float64).reshape(-1)
nz = gv != 0
rel = np.abs(out[nz] - gv[nz]) / np.abs(gv[nz])
print(f'  $1 RTL == host model (PE fp_mul rounding): {100 * np.mean(out == m):.3f}% of {m.size} (exact-RNE model {100 * np.mean(out == m_rne):.3f}%); vs voyager: '
      f'{100 * np.mean(out == gv):.3f}% bit-exact, mean rel err {100 * rel.mean():.4f}%, max {100 * rel.max():.4f}%')
" >> $STATUS 2>&1
}

run_tile() {  # run
    local k=$1 name=k$1 app=${KEY}$1_RV_E64_MB start=$(date +%s)
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

TILES=${TILES:-$DEF_TILES}
log "$KERNEL START (runs $TILES)"
for k in $TILES; do
    run_tile $k
done
log "$KERNEL DONE"
