#!/bin/bash
# Save one passing MU+CGRA regress test's collateral for real-chip measurement, then push it to khaki.
# Must run right after the test passes and BEFORE the next aha regress run: regress deletes
# /aha/voyager/compiled_collateral/<mu_test> and overwrites the app's bin/ for every test.
#
# Usage: stage_chip_collateral.sh <mu_test> <app key>_RV_E64_MB <halide app dir name>
#   e.g. stage_chip_collateral.sh mamba2_prefill-submodule_9 \
#            zircon_2d_psum_reduction_fp_post_mamba2_prefill_out_projection_kernel0_RV_E64_MB zircon_2d_psum_reduction_fp
#
# Entry name = the regress test name with " -> " replaced by ",,," (the layout of the user's aha_srcN / voyager_srcN):
#   aha_src_mamba/<name>/     = contents of apps/<app>/bin/
#   voyager_src_mamba/<name>/ = contents of /aha/voyager/compiled_collateral/<mu_test>/
# Local copies stay in /aha/{aha,voyager}_src_mamba even if the push fails (needs the forwarded SSH agent).
MU_TEST=$1; APP_TEST=$2; APP=$3
NAME="$MU_TEST,,,$APP_TEST"
BIN=/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/$APP/bin
COLLATERAL=/aha/voyager/compiled_collateral/$MU_TEST
STAGE_AHA=/aha/aha_src_mamba; STAGE_VOY=/aha/voyager_src_mamba
REMOTE=mcoduoza@khaki.stanford.edu; REMOTE_DIR=/nobackup/mcoduoza/zircon_aha_flow_bins

for d in "$BIN" "$COLLATERAL"; do
    [ -d "$d" ] || { echo "STAGE FAILED: missing $d"; exit 1; }
done
rm -rf "$STAGE_AHA/$NAME" "$STAGE_VOY/$NAME"
mkdir -p "$STAGE_AHA/$NAME" "$STAGE_VOY/$NAME"
cp -a "$BIN/." "$STAGE_AHA/$NAME/" && cp -a "$COLLATERAL/." "$STAGE_VOY/$NAME/" || { echo "STAGE FAILED: copy $NAME"; exit 1; }
echo "STAGED $NAME: aha $(du -sh "$STAGE_AHA/$NAME" | cut -f1) ($(ls "$STAGE_AHA/$NAME" | wc -l) entries), voyager $(du -sh "$STAGE_VOY/$NAME" | cut -f1)"

status=0
for pair in "aha_src_mamba:$STAGE_AHA" "voyager_src_mamba:$STAGE_VOY"; do
    dst=${pair%%:*}; src=${pair#*:}
    if tar -C "$src" -cf - "$NAME" | ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "tar -C $REMOTE_DIR/$dst -xf -"; then
        echo "PUSHED $dst/$NAME"
    else
        echo "PUSH FAILED $dst/$NAME (local copy kept in $src)"; status=2
    fi
done
exit $status
