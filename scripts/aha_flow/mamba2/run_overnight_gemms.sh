#!/bin/bash
# Overnight bring-up 2026-11-19: Mamba2 C@B^T (1 run), then xBC_proj (8 psum kernels), then lm_head
# (2 non-uniform K slices x 8 psum kernels), all through RTL sim with aha regress, tile (0,0) only.
#
# Per kernel:
#   pass -> stage its bin/ + voyager collateral for chip runs and push to khaki (stage_chip_collateral.sh),
#           and copy its RTL output into the chain folder the next psum kernel reads;
#   fail -> save the app's bin/, the voyager compiled collateral and (only if RTL sim produced it in this
#           run) hw_output.txt to <log dir>/<gemm>/failed_kernel_<i>/, then stop that GEMM's chain (later
#           kernels need this kernel's output) and move on to the next GEMM. Failures are not debugged.
# Host psum golds (make_psum_golds.py) are generated right before each chain.
# Usage: run_overnight_gemms.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGROOT=/aha/rerun_logs/2026-11-19/overnight
STATUS=$LOGROOT/overnight.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
PSUM=$APPS/zircon_2d_psum_reduction_fp
PY=/aha/voyager/.conda-env/bin/python
mkdir -p $LOGROOT
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

save_failure() {  # gemm idx mu_test halide_app start_epoch
    local dest=$LOGROOT/$1/failed_kernel_$2 hw=/aha/garnet/tests/test_app/hw_output.txt
    mkdir -p $dest
    cp -a $APPS/$4/bin $dest/aha_bin 2>/dev/null && log "  saved aha bin -> $dest/aha_bin"
    if [ -d /aha/voyager/compiled_collateral/$3 ]; then
        cp -a /aha/voyager/compiled_collateral/$3 $dest/voyager_compiled_collateral && log "  saved voyager collateral -> $dest"
    fi
    if [ -f $hw ] && [ $(stat -c %Y $hw) -ge $5 ]; then
        cp -a $hw $dest/hw_output.txt && log "  failed in RTL sim: saved hw_output.txt -> $dest"
    else
        log "  no hw_output.txt from this run (failed before RTL sim)"
    fi
}

run_kernel() {  # gemm idx mu_test app_test halide_app chain_dir_or_-
    local gemm=$1 idx=$2 mu=$3 app=$4 happ=$5 chain=$6 start=$(date +%s)
    mkdir -p $LOGROOT/$gemm
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"external_mu_tests_fp":["'"$mu -> $app"'"]}'
    log "$gemm kernel $idx START $app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGROOT/$gemm/kernel_$idx.log 2>&1
    local rc=$?
    log "$gemm kernel $idx EXIT $rc"
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$mu" "$app" "$happ" >> $STATUS 2>&1
        if [ "$chain" != "-" ]; then
            mkdir -p $chain && cp /aha/garnet/tests/test_app/hw_output.txt $chain/kernel_${idx}_output.txt
        fi
    else
        save_failure $gemm $idx $mu $happ $start
    fi
    return $rc
}

run_chain() {  # gemm mu_test app_root gold_layer chain_dir num_psums
    local gemm=$1 mu=$2 root=$3 layer=$4 chain=$5 n=$6
    local host_gold=${chain%_gold}_host_gold k0=zircon_2d_nop_post_mamba2_prefill_${root}_kernel0
    mkdir -p $LOGROOT/$gemm
    log "$gemm host golds START"
    if ! $PY -I $TOOLS/make_psum_golds.py $layer $k0 $host_gold > $LOGROOT/$gemm/host_golds.log 2>&1 \
            || ! grep -q "PASS" $LOGROOT/$gemm/host_golds.log; then
        log "$gemm host golds FAILED (see host_golds.log); chain not run"
        return 1
    fi
    log "$gemm host golds DONE: $(grep 'vs voyager gold' $LOGROOT/$gemm/host_golds.log)"
    rm -rf $chain
    for i in $(seq 0 $((n - 1))); do
        local app=zircon_2d_psum_reduction_fp happ=zircon_2d_psum_reduction_fp copy=$chain
        [ $i -eq 0 ] && app=zircon_2d_nop && happ=zircon_2d_nop
        [ $i -eq $((n - 1)) ] && copy=-
        run_kernel $gemm $i $mu ${app}_post_mamba2_prefill_${root}_kernel${i}_RV_E64_MB $happ $copy || {
            log "$gemm chain STOPPED at kernel $i"; return 1; }
    done
    log "$gemm chain DONE 0..$((n - 1))"
}

log "OVERNIGHT START"
run_kernel cbt 0 mamba2_prefill-matmul_mx zircon_2d_nop_post_mamba2_prefill_CBt_RV_E64_MB zircon_2d_nop -
run_chain xbc_proj mamba2_prefill-linear_mx_default_1 xBC_projection linear_mx_default_1 \
    $PSUM/mamba2_prefill-linear_mx_default_1_gold 8
run_chain lm_head_slice0 mamba2_prefill-linear_mx_default_4 lm_head_slice0 linear_mx_default_4 \
    $PSUM/mamba2_prefill-linear_mx_default_4_slice_0_gold 8
run_chain lm_head_slice1 mamba2_prefill-linear_mx_default_4 lm_head_slice1 linear_mx_default_4 \
    $PSUM/mamba2_prefill-linear_mx_default_4_slice_1_gold 8
log "OVERNIGHT DONE"
