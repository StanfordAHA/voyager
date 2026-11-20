#!/bin/bash
# Mamba2 z_proj -> SiLU bring-up, tile (0,0), reduced scope (user, 2026-11-20): kernel 0 (nop), kernel 1 (first
# psum), kernel 7 (SiLU pass 1, add_gelu_pass1_mu_input_fp) and SiLU pass 2 (add_gelu_pass2_fp, voyager_cgra test).
# Kernels 2-6 are not run: the kernel-6 partial sum read by kernel 7 and pass 2 is the host-computed running sum.
#   1. host golds (make_psum_golds.py, incl. the pass-1 lower gold) + host checks (check_zproj_silu_golds.py)
#   2. kernel 0 -> its live RTL output into the chain folder; kernel 1; host kernel_6 into the chain folder
#   3. kernel 7 -> its live hw_psum1_lower_output becomes pass 2's input (.raw big-endian, untrimmed, + .txt)
#   4. pass 2
# Every passing run is staged for chip runs and pushed to khaki (stage_chip_collateral.sh). On a failure: save the
# app's bin/, the voyager collateral and the RTL output files of that run, then stop. Failures are not debugged.
# Usage: [RESUME=pass2_input] run_zproj_silu.sh   (run in tmux with SSH_AUTH_SOCK passed through)
#   RESUME=pass2_input skips steps 1-3's runs (kernel 7 already passed) and starts at the pass-2 input conversion.
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=/aha/rerun_logs/2026-11-20/zproj_silu
STATUS=$LOGDIR/zproj_silu.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
PSUM=$APPS/zircon_2d_psum_reduction_fp
CHAIN=$PSUM/mamba2_prefill-linear_mx_default_gold
HOST_GOLD=$PSUM/mamba2_prefill-linear_mx_default_host_gold
PASS2_IN=$APPS/add_gelu_pass2_fp/add_gelu_pass1_psums/mamba2_prefill-silu
TB=/aha/garnet/tests/test_app
MU=mamba2_prefill-linear_mx_default
ROOT=post_mamba2_prefill_z_projection
PY=/aha/voyager/.conda-env/bin/python
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x; hw_output_txt_to_raw needs it)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

save_failure() {  # name mu_test halide_app start_epoch
    local dest=$LOGDIR/failed_$1
    mkdir -p $dest
    cp -a $APPS/$3/bin $dest/aha_bin 2>/dev/null && log "  saved aha bin -> $dest/aha_bin"
    [ -d /aha/voyager/compiled_collateral/$2 ] && cp -a /aha/voyager/compiled_collateral/$2 $dest/voyager_compiled_collateral \
        && log "  saved voyager collateral -> $dest"
    local saved=0
    for f in $TB/hw_output.txt $TB/hw_psum1_lower_output.txt; do
        if [ -f $f ] && [ $(stat -c %Y $f) -ge $4 ]; then cp -a $f $dest/ && saved=1 && log "  failed in RTL sim: saved $(basename $f)"; fi
    done
    [ $saved -eq 0 ] && log "  no RTL output from this run (failed before RTL sim)"
}

run_test() {  # name group mu_or_cgra_test app_test halide_app
    local name=$1 group=$2 mu=$3 app=$4 happ=$5 start=$(date +%s) sep=" -> "
    [ $group = voyager_cgra_tests_fp ] && sep="::"
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"'$group'":["'"$mu$sep$app"'"]}'
    log "$name START $mu$sep$app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc"
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$mu" "$app" "$happ" >> $STATUS 2>&1
    else
        save_failure $name $mu $happ $start
        log "STOPPED at $name"; exit $rc
    fi
}

log "ZPROJ_SILU START${RESUME:+ (RESUME=$RESUME)}"
if [ "$RESUME" != pass2_input ]; then
# 1. host golds + host checks
$PY -I $TOOLS/make_psum_golds.py linear_mx_default zircon_2d_nop_${ROOT}_kernel0 $HOST_GOLD \
    --pass1-lower add_gelu_pass1_mu_input_fp_${ROOT}_kernel7 $HOST_GOLD/pass1_hw_psum1_lower_output.txt \
    > $LOGDIR/host_golds.log 2>&1 && grep -q "PASS" $LOGDIR/host_golds.log \
    || { log "host golds FAILED (host_golds.log)"; exit 1; }
log "host golds DONE: $(grep 'vs voyager gold' $LOGDIR/host_golds.log)"
$PY -I $TOOLS/check_zproj_silu_golds.py $HOST_GOLD > $LOGDIR/host_checks.log 2>&1 \
    || { log "host checks FAILED (host_checks.log)"; exit 1; }
log "host checks PASS: $(grep -c 'PASS' $LOGDIR/host_checks.log) lines"

# 2. kernel 0 (nop) and kernel 1 (first psum), then the host kernel-6 running sum for kernel 7 / pass 2
rm -rf $CHAIN && mkdir -p $CHAIN
run_test kernel_0 external_mu_tests_fp $MU zircon_2d_nop_${ROOT}_kernel0_RV_E64_MB zircon_2d_nop
cp $TB/hw_output.txt $CHAIN/kernel_0_output.txt
run_test kernel_1 external_mu_tests_fp $MU zircon_2d_psum_reduction_fp_${ROOT}_kernel1_RV_E64_MB zircon_2d_psum_reduction_fp
cp $HOST_GOLD/kernel_6_output.txt $CHAIN/kernel_6_output.txt
log "chain kernel_6_output.txt = HOST-computed running sum through slice 6 (kernels 2-6 not run)"

# 3. kernel 7 = SiLU pass 1; its live lower output becomes pass 2's input
run_test kernel_7_pass1 external_mu_tests_fp $MU add_gelu_pass1_mu_input_fp_${ROOT}_kernel7_RV_E64_MB add_gelu_pass1_mu_input_fp
fi
mkdir -p $PASS2_IN
cp $TB/hw_psum1_lower_output.txt $PASS2_IN/hw_psum1_lower_output_k_tiling_kernel0.txt
env -u GOLD_CHANNEL_TRIMMING_WORKAROUND $FLOW_PY -I -c "
import sys; sys.path.insert(0, '/aha/voyager/scripts/aha_flow')
import parse_dnnLayer_tensors as p
p.hw_output_txt_to_raw('$PASS2_IN/hw_psum1_lower_output_k_tiling_kernel0.txt', '$PASS2_IN/hw_psum1_lower_output_k_tiling_kernel0.raw')
" >> $STATUS 2>&1 || { log "pass-1 lower output conversion FAILED"; exit 1; }
log "pass-2 input = live pass-1 hw_psum1_lower_output ($(stat -c %s $PASS2_IN/hw_psum1_lower_output_k_tiling_kernel0.raw) bytes)"

# 4. SiLU pass 2 (standalone CGRA)
run_test pass2 voyager_cgra_tests_fp mamba2_prefill-silu add_gelu_pass2_fp_mamba2_prefill_RV_E64_MB add_gelu_pass2_fp
log "ZPROJ_SILU DONE"
