#!/bin/bash
# Push already-staged chip collateral (local /aha/aha_src_mamba/<name> and /aha/voyager_src_mamba/<name>, written by
# stage_chip_collateral.sh) to khaki, e.g. after a push failed because the forwarded SSH agent was gone, then compare
# names, file counts and file bytes local vs khaki.
# Usage: [STAGE_MODEL=llama] push_staged_to_khaki.sh <entry name> [...]   (run in tmux with a working SSH_AUTH_SOCK;
# STAGE_MODEL picks {aha,voyager}_src_<model>, default mamba)
REMOTE=mcoduoza@khaki.stanford.edu; REMOTE_DIR=/nobackup/mcoduoza/zircon_aha_flow_bins
status=0
for name in "$@"; do
    for dst in aha_src_${STAGE_MODEL:-mamba} voyager_src_${STAGE_MODEL:-mamba}; do
        src=/aha/$dst
        [ -d "$src/$name" ] || { echo "MISSING local $src/$name"; status=1; continue; }
        if tar -C "$src" -cf - "$name" | ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" "tar -C $REMOTE_DIR/$dst -xf -"; then
            loc="$(find "$src/$name" -type f | wc -l) $(find "$src/$name" -type f -printf '%s\n' | awk '{s+=$1} END {print s}')"
            rem="$(ssh -o BatchMode=yes "$REMOTE" "cd $REMOTE_DIR/$dst && echo \$(find '$name' -type f | wc -l) \$(find '$name' -type f -printf '%s\n' | awk '{s+=\$1} END {print s}')")"
            if [ "$loc" = "$rem" ]; then echo "PUSHED+VERIFIED $dst/$name ($loc)"; else echo "PUSHED but MISMATCH $dst/$name local [$loc] khaki [$rem]"; status=2; fi
        else
            echo "PUSH FAILED $dst/$name"; status=2
        fi
    done
done
exit $status
