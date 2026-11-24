#!/bin/bash
# Mamba2 MX quantization ops on Llama's two-pass CGRA apps through RTL sim, X tile 0 of each (voyager_cgra tests):
#   quantize_mx_default*  (axis -1)  get_e8m0_scale_tree_gb_input -> apply_e8m0_scale_single_IO; single_IO's scale is
#                                    the tree run's RTL output (CALCULATED_SCALE_PATH), or its voyager gold scale if the
#                                    tree run left no RTL output ("HOST SCALE")
#   calculate_mx_qparam_default*  (axis -2)  get_e8m0_scale_accum_gb_input
#   quantize_default*             (axis -2)  apply_e8m0_scale_multi_IOs (scale = voyager's calculate_mx_qparam tensor)
# Runs, tiles and checks: quant_two_pass_tools.py. All 20 runs run whatever fails. Every run saves its RTL outputs
# (hw_*output*.txt) and the voyager golds (compare/); a passing run is staged for chip runs and pushed to khaki, a
# failing run's bin/ and collateral are saved. Failures are not debugged. Waits for the gated-norm driver first
# (one regress at a time). At the end, writes the report.
# Usage: [ONLY="tree single1 ..."] run_quantize_two_pass.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/2026-11-21/quantize_two_pass}
GN_DIR=${GN_DIR:-/aha/rerun_logs/2026-11-20/gated_norm_2pass}
REPORT=${REPORT:-/aha/rerun_logs/2026-11-21/REPORT_quantize_two_pass.md}
STATUS=$LOGDIR/quantize_two_pass.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
PY="/aha/bin/python $TOOLS/quant_two_pass_tools.py"
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

if [ -f $GN_DIR/gated_norm_2pass.status ] && ! grep -q "GATED_NORM_2PASS DONE" $GN_DIR/gated_norm_2pass.status; then
    log "waiting for the gated-norm driver to finish"
    until grep -q "GATED_NORM_2PASS DONE" $GN_DIR/gated_norm_2pass.status; do sleep 30; done
fi

run_test() {  # name layer key app_dir kind
    local name=$1 mu=mamba2_prefill-$2 app=$3_RV_E64_MB app_dir=$4 kind=$5 start=$(date +%s)
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$mu::$app"'"]}'
    log "$name START $mu::$app"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    $PY check-inputs $name >> $STATUS 2>&1
    mkdir -p $LOGDIR/$name
    local saved=""
    for f in $TB/hw_*output*.txt; do
        [ -f "$f" ] && [ $(stat -c %Y "$f") -ge $start ] && cp -a "$f" $LOGDIR/$name/ && saved="$saved $(basename $f)"
    done
    log "  saved RTL outputs:${saved:- none (failed before RTL sim)}"
    [ -d /aha/voyager/compiled_collateral/$mu/compare ] && cp -a /aha/voyager/compiled_collateral/$mu/compare $LOGDIR/$name/
    if [ $kind = tree ]; then   # hand the tree's scale to the single_IO run that follows
        $PY scale-raw $2 $start >> $STATUS 2>&1
    fi
    if [ $rc -eq 0 ]; then
        bash $TOOLS/stage_chip_collateral.sh "$mu" "$app" "$app_dir" </dev/null >> $STATUS 2>&1
    else
        cp -a $APPS/$app_dir/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$mu ] && cp -a /aha/voyager/compiled_collateral/$mu $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name FAILED (not debugged); continuing"
    fi
}

log "QUANTIZE_TWO_PASS START (${ONLY:-all 20})"
mapfile -t RUN_LINES < <($PY list)   # read up front: nothing in the loop may consume the run list
for line in "${RUN_LINES[@]}"; do
    read name layer key app_dir kind <<< "$line"
    if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $name "* ]]; then continue; fi
    run_test $name $layer $key $app_dir $kind
done
log "QUANTIZE_TWO_PASS RUNS DONE"
$PY report $LOGDIR $GN_DIR $REPORT > $LOGDIR/report.out 2>&1 && log "REPORT $REPORT" || log "REPORT FAILED (see report.out)"
log "QUANTIZE_TWO_PASS DONE"
