"""Write the 2026-11-20 RMSNorm + quantize_mx_default bring-up report from the drivers' status files and logs.
Usage: python report_rmsnorm_quantize.py <out.md>   (run after both drivers finish)"""
import os, re, sys, time

D = "/aha/rerun_logs/2026-11-20"
RUNS = [  # name, log, status file, values per lane (ideal kernel cycles)
    ("block norm (layer_norm_default, 64 x 2048)", f"{D}/rmsnorm_v2/block_norm.log", f"{D}/rmsnorm_v2/rmsnorm.status", "block_norm", 8192),
    ("norm_f (layer_norm_default_1, 64 x 2048)", f"{D}/rmsnorm_v2/norm_f.log", f"{D}/rmsnorm_v2/rmsnorm.status", "norm_f", 8192),
    ("gated norm (submodule_8, 32 x 4096)", f"{D}/rmsnorm_v2/gated_norm.log", f"{D}/rmsnorm_v2/rmsnorm.status", "gated_norm", 8192),
    ("quantize_mx_default (64 x 2048)", f"{D}/quantize_mx/quantize_mx_default.log", f"{D}/quantize_mx/quantize_mx.status", "quantize_mx_default", 4096),
]
strip = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)


def read(p):
    return strip(open(p, errors="replace").read()) if os.path.exists(p) else ""


rows, details = [], []
for title, log, status, key, ideal in RUNS:
    lg, st = read(log), read(status)
    m = re.search(rf"^{key} EXIT (\d+)", st, re.M)
    rc = m.group(1) if m else "did not run / no EXIT"
    comp = [l.strip() for l in lg.splitlines() if re.search(r"comparison (mostly )?passed|comparison failed|exceptions in", l)
            and "rv_comparison" not in l and "Mux in" not in l]
    cyc = re.findall(r"It takes ([0-9.]+) ns total time", lg)
    cycles = int(float(cyc[-1])) if cyc else None
    verdict = "PASS" if rc == "0" else ("FAIL" if m else "NOT RUN")
    util = f"{100 * ideal / cycles:.1f}%" if cycles else "-"
    rows.append(f"| {title} | {verdict} (exit {rc}) | {comp[-1] if comp else '-'} | {cycles or '-'} | {ideal} | {util} |")
    block = [l for l in st.splitlines() if key in l or "input" in l.lower()]
    staged = [l for l in st.splitlines() if ("STAGED" in l or "PUSH" in l or "VERIFIED" in l or "MISMATCH" in l)
              and (key.replace("_", "") in l.replace("_", "") or "quantize" in key)]
    details.append(f"### {title}\n" + "\n".join(f"- `{l.strip()}`" for l in block[:6] + staged[:6]) + "\n")
    tb = [l for l in lg.splitlines() if "Traceback" in l or "Error" in l and "Error count" not in l and "compiler failed" not in l]
    if verdict != "PASS" and tb:
        details.append("  First error lines:\n" + "\n".join(f"  - `{l.strip()[:200]}`" for l in tb[:5]) + "\n")

chains = "\n".join(f"- `{l.strip()}`" for p in (f"{D}/rmsnorm_v2/chain.log", f"{D}/quantize_mx/chain.log")
                   for l in read(p).splitlines() if l.strip())
out = f"""# Report: Mamba2 RMSNorm (3 layers) + quantize_mx_default, X tile 0 each

Generated {time.strftime('%F %T')} from the driver logs in `{D}/rmsnorm_v2/` and `{D}/quantize_mx/`.

## Results
| Run | Result | Gold comparison (last line) | RTL kernel cycles | Ideal (values or passes per lane) | Utilization |
|---|---|---|---|---|---|
{chr(10).join(rows)}

- Cycles are from "It takes N ns total time to run kernel" (1 ns clock).
- The ideal is one value per lane per cycle: 131,072 / 16 lanes = 8,192 for RMSNorm; 4,096 passes for the quantize.

## Details (input checks, staging, khaki)
{chr(10).join(details)}
## Chain logs
{chains}

## Deviations from the plans
1. **RMSNorm gold (block norm, norm_f).** Voyager's gold for `layer_norm_default` / `layer_norm_default_1` is a true LayerNorm: the old compiler's `replace_rmsnorm_with_layer_norm` swapped Mamba2's RMSNorm.
   - The first block-norm run passed only marginally: 5.47% out of tolerance vs that LayerNorm, but 0.00% vs a host RMSNorm.
   - Added a host RMSNorm gold hook (`RMS_NORM_GOLD_INPUT` / `_WEIGHT` in `parse_dnnLayer_tensors.parse_output`) to both keys.
   - The gated norm keeps voyager's `submodule_8` gold, the graph's exact RMS formula.
2. **RMSNorm runs restarted.** At the user's request, the in-flight run (norm_f against the LayerNorm gold) was cancelled, and all three norms were re-run from scratch with the new gold (`rmsnorm_v2`).
3. **Gated norm γ.** Handed over with the new `CGRA_WEIGHT_NODE` env (planned). `submodule_8`'s first op carries no weight.
4. **quantize_mx_default run size.** 64 rows x 2048 (16 X tiles, 4,096 passes per lane) instead of 128 rows, at the user's request, to stay well below the template's 8,192-pass FIFO-Pond limit.
5. **quantize code changes (planned).** A `USE_REAL_INPUT` branch in `fused_e8m0_quant_single_IO/process.cpp`; the regress_util keyword `fused_e8m0_quant_single_IO`.
6. **khaki pushes** need the forwarded SSH agent. If a push above says PUSH FAILED, the local copies in `/aha/{{aha,voyager}}_src_mamba` are kept; re-push with `voyager/scripts/aha_flow/mamba2/push_staged_to_khaki.sh <entry>`.

All changes are uncommitted.
"""
open(sys.argv[1], "w").write(out)
print(out)
