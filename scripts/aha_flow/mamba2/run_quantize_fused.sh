#!/bin/bash
# Mamba2 MX quantization on yuchen's one-pass fused apps through RTL sim (voyager_cgra tests), tile 0 of each layer at
# 131,072 values (4,096 passes per lane): fused_e8m0_quant_single_IO for quantize_mx_default_1 .. _5 (blocks along
# dim -1; _default itself passed on 2026-11-21) and fused_e8m0_quant_multi_IOs for the 4 calculate_mx_qparam + quantize
# pairs (blocks along dim -2), run on the quantize layer with its scale input as the scale gold (FUSED_SCALE_GOLD).
# After each run: test.py's comparison and the cycles, every hw_*output*.txt and the golds are saved, and
# quant_fused_tools.py check compares the inputs, the golds and the RTL output with the HW model and with the two-pass RTL
# outputs on the overlap. A run whose output is explained by the 4 HW quantizer mechanisms (the acceptance rule; test.py
# is expected to FAIL on most) is staged for chip runs, pushed to khaki and verified; otherwise its bin/ and
# collateral are saved and the next run still starts. Failures are not debugged.
# QUANT_MODEL=llama_prefill runs Llama's 6 layers (quant_fused_tools.py); Llama runs are not staged (no khaki destination
# for Llama chip collateral yet).
# Usage: [QUANT_MODEL=llama_prefill] [RUNS="single_5 multi_1 ..."] run_quantize_fused.sh   (run in tmux with SSH_AUTH_SOCK passed through)
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGDIR=${LOGDIR:-/aha/rerun_logs/$(date +%F)/quantize_fused}
STATUS=$LOGDIR/quantize_fused.status
APPS=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps
TB=/aha/garnet/tests/test_app
FLOW_PY=/aha/bin/python   # the Python aha's map.py runs parse_dnnLayer_tensors.py with (NumPy 1.x)
mkdir -p $LOGDIR
source /aha/bin/activate
source /cad/modules/tcl/init/bash && module load base vcs
cd /aha

log() { echo "$* $(date '+%F %T')" >> $STATUS; }

run_one() {  # name layer key app
    local name=$1 layer=$2 key=$3 app=$4 start=$(date +%s)
    local mu=${QUANT_MODEL:-mamba2_prefill}-$layer test=${key}_RV_E64_MB
    local config='{"width":28,"height":16,"cols_removed":12,"mu_oc_0":32,"voyager_cgra_tests_fp":["'"$mu::$test"'"]}'
    log "$name START $mu::$test"
    TOOL=VCS aha regress "$config" --include-no-zircon-tests </dev/null > $LOGDIR/$name.log 2>&1
    local rc=$?
    log "$name EXIT $rc $(grep -a -o 'It takes [0-9.]* ns total time' $LOGDIR/$name.log | tail -1)"
    sed 's/\x1b\[[0-9;]*m//g' $LOGDIR/$name.log | grep -a -E 'Integer.*(passed|failed|tolerance)|Interrupt wait timeout|Error' \
        | grep -a -v -E 'rv_comparison|Mux in' | tail -4 | while read -r line; do log "  $line"; done
    mkdir -p $LOGDIR/$name
    for f in $TB/hw_*output*.txt; do
        [ -f "$f" ] && [ $(stat -c %Y "$f") -ge $start ] && cp -a "$f" $LOGDIR/$name/ && log "  saved $(basename $f)"
    done
    [ -d /aha/voyager/compiled_collateral/$mu/compare ] && cp -a /aha/voyager/compiled_collateral/$mu/compare $LOGDIR/$name/
    cp -a $APPS/$app/bin/design_meta.json $LOGDIR/$name/ 2>/dev/null
    $FLOW_PY -I $TOOLS/quant_fused_tools.py check $name $LOGDIR/$name >> $STATUS 2>&1
    local explained=$?
    # The user's acceptance rule for the quantize layers: not a test.py PASS (voyager's reference differs by design), but
    # an RTL output that is explained by the 4 HW mechanisms (check exit 0: inputs right, RTL == HW model, no difference
    # from voyager left unexplained)
    if [ $explained -eq 0 ]; then
        log "  $name ACCEPTED (explained by the HW mechanisms; test.py exit $rc)"
        if [ "${QUANT_MODEL:-mamba2_prefill}" = mamba2_prefill ]; then
            bash $TOOLS/stage_chip_collateral.sh "$mu" "$test" "$app" </dev/null >> $STATUS 2>&1
            bash $TOOLS/push_staged_to_khaki.sh "$mu,,,$test" </dev/null >> $STATUS 2>&1
        else
            cp -a $APPS/$app/bin $LOGDIR/$name/aha_bin 2>/dev/null
            log "  $name not staged (no khaki destination for $QUANT_MODEL); bin saved -> $LOGDIR/$name/aha_bin"
        fi
    else
        cp -a $APPS/$app/bin $LOGDIR/$name/aha_bin 2>/dev/null && log "  saved aha bin -> $LOGDIR/$name/aha_bin"
        [ -d /aha/voyager/compiled_collateral/$mu ] && cp -a /aha/voyager/compiled_collateral/$mu $LOGDIR/$name/voyager_compiled_collateral \
            && log "  saved voyager collateral -> $LOGDIR/$name"
        log "  $name NOT EXPLAINED by the HW mechanisms (test.py exit $rc); not staged, not debugged; continuing"
    fi
}

ALL=$($FLOW_PY -I $TOOLS/quant_fused_tools.py list)
RUNS=${RUNS:-$(echo "$ALL" | cut -d' ' -f1 | tr '\n' ' ')}
log "QUANTIZE_FUSED START (runs $RUNS)"
for n in $RUNS; do
    run_one $(echo "$ALL" | awk -v n=$n '$1 == n')
done
log "QUANTIZE_FUSED DONE"
