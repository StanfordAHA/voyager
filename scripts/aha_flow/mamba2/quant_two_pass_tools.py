"""Helpers for run_quantize_two_pass.sh (Mamba2 MX quantization on Llama's two-pass apps, X tile 0 each).
  list                          the 20 runs, in run order: name layer key app_dir kind
  check-inputs <name>           the inputs the app loaded vs the voyager tensors (tile 0)
  scale-raw <op> <start>        tree RTL scale (hw_scale_output.txt, newer than <start>) -> the single_IO
                                CALCULATED_SCALE_PATH raw; falls back to the tree's voyager gold scale ("HOST SCALE")
  report <logdir> <gated_norm_logdir> <out.md>
Run with /aha/bin/python (NumPy 1.x)."""
import json, os, re, sys, time
import numpy as np

B = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
APPS = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps"
TB = "/aha/garnet/tests/test_app"
COLLATERAL = "/aha/voyager/compiled_collateral"
KEYS = json.load(open("/aha/aha/util/application_parameters.json"))
SHAPES = {"layer_norm_default": [1, 1024, 2048], "permute_3": [1, 4, 1, 128, 256], "reshape_8": [1, 4, 1, 256, 128],
          "mul_10": [1, 4, 1, 64, 256, 256], "submodule_8": [1, 1024, 4096], "layer_norm_default_1": [1, 1024, 2048],
          "permute_1": [1, 4, 1, 64, 256, 64], "permute_2": [1, 4, 1, 128, 256], "reshape_12": [1, 4, 1, 256, 4096],
          "reshape_14": [1, 4, 1, 128, 4096]}
QMX = {"quantize_mx_default": "layer_norm_default", "quantize_mx_default_1": "permute_3", "quantize_mx_default_2": "reshape_8",
       "quantize_mx_default_3": "mul_10", "quantize_mx_default_4": "submodule_8", "quantize_mx_default_5": "layer_norm_default_1"}
PAIRS = {"": "permute_1", "_1": "permute_2", "_2": "reshape_12", "_3": "reshape_14"}
ORDER = ["quantize_mx_default", "quantize_mx_default_5", "quantize_mx_default_4", "pair_1", "pair", "pair_3",
         "quantize_mx_default_1", "quantize_mx_default_2", "pair_2", "quantize_mx_default_3"]
APP_OF = {"tree": "get_e8m0_scale_tree_gb_input", "single": "apply_e8m0_scale_single_IO",
          "accum": "get_e8m0_scale_accum_gb_input", "multi": "apply_e8m0_scale_multi_IOs"}


def runs():
    out = []
    for item in ORDER:
        if item.startswith("pair"):
            sfx = item[4:]
            out += [(f"accum{sfx}", f"calculate_mx_qparam_default{sfx}", "accum", PAIRS[sfx]),
                    (f"multi{sfx}", f"quantize_default{sfx}", "multi", PAIRS[sfx])]
        else:
            sfx = item[len("quantize_mx_default"):]
            out += [(f"tree{sfx}", item, "tree", QMX[item]), (f"single{sfx}", item, "single", QMX[item])]
    return [dict(name=n, layer=l, kind=k, node=x, app=APP_OF[k], key=f"{APP_OF[k]}_mamba2_prefill_{l}") for n, l, k, x in out]


RUNS = {r["name"]: r for r in runs()}
bits16 = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
code = lambda s: ((np.ascontiguousarray(s, dtype=np.float32).view(np.uint32) >> 23) & 0xFF).astype(np.int64)
raw_be = lambda f: np.fromfile(f, dtype=">u2").astype(np.int64) if os.path.exists(f) else None


def tile2d(a, last, key):
    """Tile 0 of a tensor viewed as [rows, last], per the key's IO tiling."""
    e, a = KEYS[key]["default"], a.reshape(-1, last)
    if e.get("IO_DIM1_TILING") == "1":
        a = a[: a.shape[0] // int(e["NUM_IO_DIM1_TILING_KERNELS"])]
    if e.get("IO_INNERMOST_DIM_TILING") == "1":
        a = a[:, : a.shape[1] // int(e["NUM_IO_INNERMOST_DIM_TILING_KERNELS"])]
    return a


def load_hex_block0(path):  # test.py's _load_hex_txt(block_index=0)
    blocks, cur = [], []
    for line in open(path):
        if line.strip():
            cur.extend(int(v, 16) for v in line.split())
        elif cur:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    return np.array(blocks[0] if blocks else [], dtype=np.int64)


def unpack(words):  # test.py's unpack_output: low byte, then high byte
    out = np.empty(words.size * 2, dtype=np.int64)
    out[0::2], out[1::2] = words & 0xFF, (words >> 8) & 0xFF
    return out


def check_inputs(name):
    r = RUNS[name]
    shape, C = SHAPES[r["node"]], SHAPES[r["node"]][-1]
    want = bits16(tile2d(np.fromfile(B + r["node"] + ".bin", dtype=np.float32), C, r["key"])).astype(np.int64).reshape(-1)
    d = f"{APPS}/{r['app']}"
    got = raw_be(f"{d}/hw_input_stencil.raw")
    msg = [f"hw_input_stencil.raw == tile 0 of {r['node']} ({want.size} values): {got is not None and np.array_equal(got, want)}"]
    if r["kind"] == "single":
        act, scale = raw_be(f"{d}/bin/input_bf_act_host_stencil.raw"), raw_be(f"{d}/bin/input_scale_host_stencil.raw")
        calc = raw_be(KEYS[r["key"]]["default"]["CALCULATED_SCALE_PATH"])
        msg.append(f"bin/input_bf_act_host_stencil.raw == it: {act is not None and np.array_equal(act, want)}")
        msg.append(f"bin/input_scale_host_stencil.raw == CALCULATED_SCALE_PATH raw ({0 if calc is None else calc.size} scales): "
                   f"{scale is not None and calc is not None and np.array_equal(scale, calc)}")
    if r["kind"] == "multi":
        calc_node = "calculate_mx_qparam_default" + r["layer"][len("quantize_default"):]
        s = code(tile2d(np.fromfile(B + calc_node + ".bin", dtype=np.float32), C, r["key"])).reshape(-1)
        packed = (s[1::2] << 8) | s[0::2]
        got_s = raw_be(f"{d}/hw_input_scale_stencil.raw")
        msg.append(f"hw_input_scale_stencil.raw == {calc_node} tile, 2 per word ({packed.size} words): "
                   f"{got_s is not None and np.array_equal(got_s, packed)}")
    print(f"  {name} inputs: " + "; ".join(msg))


def scale_raw(op, start):
    """Tree RTL scale -> the single_IO CALCULATED_SCALE_PATH raw (BE u16 per scale, Llama's format)."""
    dest = KEYS[f"apply_e8m0_scale_single_IO_mamba2_prefill_{op}"]["default"]["CALCULATED_SCALE_PATH"]
    gold_path = f"{COLLATERAL}/mamba2_prefill-{op}/compare/gold_scale.txt"
    gold = load_hex_block0(gold_path) if os.path.exists(gold_path) else None
    hw = f"{TB}/hw_scale_output.txt"
    if os.path.exists(hw) and os.path.getmtime(hw) >= start:
        vals, src = load_hex_block0(hw), "RTL"
    elif gold is not None:
        vals, src = gold, "HOST SCALE (the tree run left no RTL output: voyager's gold scale)"
    else:
        print(f"  scale-raw {op}: no RTL output and no gold scale; single_IO will use a stale or missing {dest}")
        return 1
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    vals.astype(">u2").tofile(dest)
    match = "" if gold is None or gold.size != vals.size else f"; == gold_scale.txt on {100 * (vals == gold).mean():.2f}%"
    print(f"  scale-raw {op}: {src} -> {dest} ({vals.size} scales{match})")
    return 0


# ---------------------------------------------------------------- report
GOLD_OF = {"hw_scale_output": "gold_scale.txt", "hw_scale_packed_output": "gold_scale.txt",
           "hw_output_mxint8_act": "gold_activation.txt", "hw_output": "gold_activation.txt"}
strip = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
readf = lambda p: strip(open(p, errors="replace").read()) if os.path.exists(p) else ""


def compare_saved(rundir, kind):
    """Each saved RTL output vs its saved gold: test.py's unsigned-byte view, plus a signed view (int8 data as signed;
    a -1 vs 0 tie shows as 255 vs 0 in the unsigned view)."""
    rows = []
    for out, gname in GOLD_OF.items():
        sim_p, gold_p = f"{rundir}/{out}.txt", f"{rundir}/compare/{gname}"
        if not os.path.exists(sim_p) or not os.path.exists(gold_p):
            continue
        sim, gold = load_hex_block0(sim_p), load_hex_block0(gold_p)
        if kind != "tree":   # PACKED_OUTPUTS on single_IO, accum and multi_IOs
            sim = unpack(sim)
        n = min(sim.size, gold.size)
        sim, gold = sim[:n], gold[:n]
        du = np.abs(sim - gold)
        is_data = gname == "gold_activation.txt" and kind != "accum"
        s8 = lambda a: np.where(a > 127, a - 256, a) if is_data else a
        ds = np.abs(s8(sim) - s8(gold))
        rows.append(f"`{out}` vs `{gname}` ({n} values): unsigned (test.py) {int((du > 1).sum())} off by >1, "
                    f"{100 * (du == 1).mean():.2f}% off by 1; signed {int((ds > 1).sum())} off by >1 (max {int(ds.max()) if n else 0}), "
                    f"{100 * (ds == 1).mean():.2f}% off by 1")
    return rows


def report(logdir, gn_dir, out_md):
    st = readf(f"{logdir}/quantize_two_pass.status")
    lines = ["# Report: Mamba2 two-pass MX quantization (20 runs) + the two-pass gated RMSNorm",
             "", f"Generated {time.strftime('%F %T')}. Logs: `{logdir}/` and `{gn_dir}/`.", ""]
    # gated norm
    gst = readf(f"{gn_dir}/gated_norm_2pass.status")
    lines += ["## Gated RMSNorm (`submodule_8`), two passes, X tile 0 (rows 0-63)", "",
              "| Run | Result | Gold comparison | RTL cycles | Ideal per lane | Utilization |", "|---|---|---|---|---|---|"]
    for name, ideal in (("pass1", 8192), ("pass2_k0", 8192), ("pass2_k1", 8192)):
        m = re.search(rf"^{name} EXIT (\d+)(?: It takes ([0-9.]+) ns)?", gst, re.M)
        lg = readf(f"{gn_dir}/{name}.log")
        comp = [l.strip() for l in lg.splitlines() if re.search(r"comparison (mostly )?passed|comparison failed", l)]
        cyc = int(float(m.group(2))) if m and m.group(2) else None
        res = "NOT RUN" if not m else ("PASS" if m.group(1) == "0" else f"FAIL (exit {m.group(1)})")
        lines.append(f"| {name} | {res} | {comp[-1] if comp else '-'} | {cyc or '-'} | {ideal} | {f'{100 * ideal / cyc:.1f}%' if cyc else '-'} |")
    lines += ["", "Checks logged by the driver:"] + [f"- `{l.strip()}`" for l in gst.splitlines()
                                                    if "inputs" in l or "IDs match" in l or "live pass-2" in l or "path_balancing" in l] + [""]
    # quantize
    lines += ["## MX quantization, X tile 0 each", "",
              "| Run | App | Layer | Tile (rows x cols) | Result | test.py comparison | RTL cycles | Ideal per lane | Utilization | Staged |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    details = []
    for r in runs():
        e = KEYS[r["key"]]["default"]
        C = SHAPES[r["node"]][-1]
        rows = int(np.prod(SHAPES[r["node"]]) // C)
        if e.get("IO_DIM1_TILING") == "1":
            rows //= int(e["NUM_IO_DIM1_TILING_KERNELS"])
        if e.get("IO_INNERMOST_DIM_TILING") == "1":
            C //= int(e["NUM_IO_INNERMOST_DIM_TILING_KERNELS"])
        ideal = rows * C // 32
        m = re.search(rf"^{r['name']} EXIT (\d+)(?: It takes ([0-9.]+) ns)?", st, re.M)
        lg = readf(f"{logdir}/{r['name']}.log")
        comp = [l.strip() for l in lg.splitlines() if re.search(r"(?i)comparison|integer.*(pass|fail)|mismatch", l)
                and "rv_comparison" not in l and "Mux in" not in l and "Warning" not in l]
        cyc = int(float(m.group(2))) if m and m.group(2) else None
        res = "NOT RUN" if not m else ("PASS" if m.group(1) == "0" else f"FAIL (exit {m.group(1)})")
        staged = "yes" if re.search(rf"STAGED mamba2_prefill-{r['layer']},,,{r['key']}_RV_E64_MB", st) else "no"
        lines.append(f"| {r['name']} | `{r['app']}` | `{r['layer']}` | {rows} x {C} | {res} | {'<br>'.join(comp[-2:]) or '-'} | "
                     f"{cyc or '-'} | {ideal} | {f'{100 * ideal / cyc:.1f}%' if cyc else '-'} | {staged} |")
        chk = [l.strip() for l in st.splitlines() if l.strip().startswith(f"{r['name']} inputs") or
               (r["kind"] == "tree" and f"scale-raw {r['layer']}:" in l)]
        cmp_rows = compare_saved(f"{logdir}/{r['name']}", r["kind"])
        details.append(f"### {r['name']} (`{r['key']}`)\n" + "\n".join(f"- {x}" for x in chk + cmp_rows) + "\n")
    lines += ["", "## Per-run checks and output analysis", ""] + details
    open(out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for r in runs():
            print(r["name"], r["layer"], r["key"], r["app"], r["kind"])
    elif cmd == "check-inputs":
        check_inputs(sys.argv[2])
    elif cmd == "scale-raw":
        sys.exit(scale_raw(sys.argv[2], float(sys.argv[3])))
    elif cmd == "report":
        report(sys.argv[2], sys.argv[3], sys.argv[4])
