"""Host checks of the Mamba2 multiply bring-ups #6, #7 and #5, before RTL.

  x_end        #6 mul_11: X_end = (x * dt) * decay_to_end   mamba_x_end_fp        (elementwise_mul_head_gather_bf16)
  state_decay  #7 mul_13: state * chunk decay               mamba_state_decay_fp  (elementwise_mul_head_gather_bf16)
  mul10        #5 mul_10: W = scores * decay                rms_norm_pass2_fp     (elementwise_mul_bf16 broadcast)

1. Template: emit the design as the app's hook does; IO DMA patterns, MEM configs and instance counts.
2. Dataflow: simulate the generated design_top.json (its wiring; every get_interleave_mem with the address patterns
   lake's own config generates; each PE multiply rounded to bf16 as the graph does) on real data for the first and last
   runs, and rebuild the output GLB tensor from the output DMA walk. Must == the gold.
3. Parser: the aha keys' env through parse_extra_inputs_bf16_cgra / parse_output == independent numpy slices.
4. Accuracy: the host bf16 model == voyager's whole-layer tensor; the PE fp_mul rounding's agreement.
Usage: check_mul_gather_host.py <kernel> <workdir>   (run with /aha/bin/python)
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_x_dt_host_lib import (bits, bf_rne, pe_mul, walk, lake_walk, read_port_rate, interleave_read,  # noqa: E402
                                 design_drivers)

KERNEL, WORK = sys.argv[1], sys.argv[2]
os.makedirs(WORK, exist_ok=True)
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
PARAMS = json.load(open("/aha/aha/util/application_parameters.json"))
U, LAYOUT, H, HEAD_DIM = 16, 32, 8, 64
B = HEAD_DIM // LAYOUT
fails = []
load = lambda node, *shape: np.fromfile(BASE + node + ".bin", dtype=np.float32).reshape(*shape)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


# Per kernel: the run's tensors from independent numpy slices. x: [rows, words per row * 16] (lane = col mod 16, word =
# row * WPR + col div 16); scalars: [words, 8] (lane j, word = row * slots + s or s); gold: as x.
def x_end_tensors(k):
    c, hb, q = k // 16, (k // 2) % 8, k % 2
    rows = slice(256 * c, 256 * c + 256)
    half = lambda a: a[rows, 512 * hb:512 * hb + 512].reshape(256, 16, 2, 16)[:, :, q, :].reshape(256, 256)
    return (half(load("reshape", 1024, 4096)),
            {"dt": load("softplus", 1024, 64)[rows, 8 * hb:8 * hb + 8],
             "decay": load("transpose", 1024, 64)[rows, 8 * hb:8 * hb + 8]},
            half(load("mul_11", 1024, 4096)))


def state_decay_tensors(k):
    u, q = k // 2, k % 2
    run = lambda a: a.reshape(128, 2, 4, 8, 2, 2, 16)[:, :, u, :, :, q, :].reshape(128, 512)
    return run(load("add_6", 128, 4096)), {"decay": load("unsqueeze_7", 2, 4, 8)[:, u, :]}, run(load("mul_13", 128, 4096))


GATHER = {
    "x_end": dict(top="mamba_x_end_fp", scalars=(("dt", 1), ("decay", 1)), slots=1, rows=256, runs=(0, 63),
                  key="mamba_x_end_fp_mamba2_prefill_k{k}", layer="mul_11", out_shape=[1, 4, 1, 256, 64, 64],
                  tensors=x_end_tensors, leaf=([2, 256], [0, 1])),
    "state_decay": dict(top="mamba_state_decay_fp", scalars=(("decay", 0),), slots=2, rows=128, runs=(0, 7),
                        key="mamba_state_decay_fp_mamba2_prefill_mul13_k{k}", layer="mul_13",
                        out_shape=[1, 1, 128, 64, 64], tensors=state_decay_tensors, leaf=([2, 2, 128], [0, 1, -1])),
}

import parse_dnnLayer_tensors as P  # noqa: E402


def with_env(key_name, fn):
    saved = dict(os.environ)
    os.environ.update(PARAMS[key_name]["default"])
    try:
        return fn()
    finally:
        os.environ.clear()
        os.environ.update(saved)


def parse_run(key_name, layer, out_shape, d):
    os.makedirs(d, exist_ok=True)

    def parse():
        zw = P.parse_zircon_workarounds()
        P.parse_extra_inputs_bf16_cgra(BASE, d, zw)
        return P.parse_output("mamba2_prefill", layer, BASE, {"node": layer, "shape": out_shape, "dtype": "bfloat16"}, zw)

    gold = np.array(with_env(key_name, parse), dtype=np.int64)
    raw = lambda name: np.fromfile(os.path.join(d, f"hw_{name}_stencil.raw"), dtype=">u2").astype(np.int64)
    return raw, gold


def load_design(top):
    design = json.load(open(os.path.join(WORK, "design_top.json")))
    mod = design["namespaces"]["global"]["modules"][top]
    inst = mod["instances"]
    meta = lambda n, key: (lambda v: json.loads(v) if isinstance(v, str) else v)(inst[n].get("metadata", {}).get(key))
    return inst, mod["connections"], meta


lane = lambda n: int(n.split("_clkwrk_")[1].split("_")[0])
pat = lambda s, key: (s["extent"], s[f"{key}_data_stride"])


def check_gather(spec):
    from strait.coreir_backend.templates.elementwise_mul_head_gather_bf16 import emit_elementwise_mul_head_gather_bf16_design

    names = [n for n, _ in spec["scalars"]]
    rows, slots = spec["rows"], spec["slots"]
    wpr = H * slots * B
    # 1. template, as the hook emits it
    emit_elementwise_mul_head_gather_bf16_design(U, H, rows, WORK, layout_lanes=LAYOUT, head_dim=HEAD_DIM,
                                                 scalars=spec["scalars"], slots=slots, heads_first_output=False,
                                                 top_module=spec["top"])
    inst, conns, meta = load_design(spec["top"])
    ios = {n for n, i in inst.items() if i.get("modref") == "global.IO"}
    in_io = sorted((n for n in ios if n.startswith("io16in_input_")), key=lane)
    sc_io = {s: sorted((n for n in ios if n.startswith(f"io16in_{s}_")), key=lane) for s in names}
    out_io = sorted((n for n in ios if n.startswith("io16_hw_output")), key=lane)
    check(f"IO counts 16 input / {' / '.join(f'8 {s}' for s in names)} / 16 output",
          (len(in_io), [len(v) for v in sc_io.values()], len(out_io)) == (16, [8] * len(names), 16))
    lin = ([wpr, rows], [1, 1])
    check(f"input DMA linear {lin[0]}/{lin[1]}", all(pat(meta(n, "glb2out_0"), "read") == lin for n in in_io))
    for s in names:
        check(f"{s} leaf DMA {spec['leaf'][0]}/{spec['leaf'][1]}",
              all(pat(meta(n, "glb2out_0"), "read") == spec["leaf"] for n in sc_io[s]))
    check(f"output DMA linear {lin[0]}/{lin[1]}", all(pat(meta(n, "in2glb_0"), "write") == lin for n in out_io))
    mems = {n: meta(n, "lake_rv_config") for n, i in inst.items() if i.get("genref") == "cgralib.Mem"}
    cnt = Counter((c["type"], c.get("single_input_stream_size", c.get("input_stream_size")), c.get("row_size") or c.get("turn"))
                  for c in mems.values())
    print("  MEM configs:", dict(cnt))
    n_s = len(names)
    check(f"MEM configs: per scalar interleave (turn 2) 4x512 + 2x1024 + 1x2048 and 8 dual_read 4096; 16 fifo 4096 "
          f"({len(mems)} MEMs)",
          cnt == Counter({("get_interleave_mem", 512, 2): 4 * n_s, ("get_interleave_mem", 1024, 2): 2 * n_s,
                          ("get_interleave_mem", 2048, 2): n_s, ("dual_read", 4096, 4): 8 * n_s, ("fifo", 4096, 8): 16}))
    pes = [n for n, i in inst.items() if i.get("modref") == "global.PE"]
    check(f"PEs: {16 * n_s} mul + {7 * n_s} pass-through", len(pes) == 23 * n_s, f"{len(pes)} PEs")
    from lake.spec.hack_rv_mem_pond_bitstream import get_interleave_mem
    root_rate = read_port_rate(lake_walk(get_interleave_mem(2048, turn=2)[3]["config"]).tolist())
    check("root MEM read port (turn 2) >= 0.95 values/cycle", root_rate >= 0.95, f"{root_rate:.3f}")

    # 2. dataflow simulation of the generated design
    drivers = design_drivers(inst, conns)
    for k in spec["runs"]:
        x, scalars, gold = spec["tensors"](k)
        cache = {}

        def stream(node, port):
            if (node, port) in cache:
                return cache[(node, port)]
            if node in ios:
                words = walk(meta(node, "glb2out_0"), "read")
                scalar = next((s for s in names if node.startswith(f"io16in_{s}_")), None)
                if scalar:
                    vals = [scalars[scalar][w, lane(node)] for w in words]
                else:
                    vals = [x[w // wpr, (w % wpr) * 16 + lane(node)] for w in words]
            elif node in mems:
                cfg = mems[node]
                if cfg["type"] == "get_interleave_mem":
                    vals = interleave_read(cfg["single_input_stream_size"], cfg.get("turn", 1),
                                           stream(*drivers[(node, "data_in_0")]), stream(*drivers[(node, "data_in_1")]))
                else:
                    vals = stream(*drivers[(node, "data_in_0")])
            elif node.startswith("mul_"):
                a = np.array(stream(*drivers[(node, "data0")]), dtype=np.float64)
                b = np.array(stream(*drivers[(node, "data1")]), dtype=np.float64)
                vals = list(bf_rne(a * b))
            else:
                vals = stream(*drivers[(node, "data0")])
            cache[(node, port)] = vals
            return vals

        out = np.full(gold.shape, np.nan)
        for n in out_io:
            vals = stream(*drivers[(n, "in")])
            words = walk(meta(n, "in2glb_0"), "write")
            assert len(vals) == len(words) == H * slots * B * rows
            for w, v in zip(words, vals):
                out[w // wpr, (w % wpr) * 16 + lane(n)] = v
        check(f"run {k}: simulated design == gold {spec['layer']} slice", np.array_equal(out, gold.astype(np.float64)),
              f"{100 * np.mean(out == gold):.3f}% equal, {np.isnan(out).sum()} unwritten")

        # 3. parser
        raw, g = parse_run(spec["key"].format(k=k), spec["layer"], spec["out_shape"], os.path.join(WORK, f"parser_k{k}"))
        check(f"run {k}: hw_input_stencil.raw == x slice {list(x.shape)}", np.array_equal(raw("input"), bits(x).reshape(-1)))
        for s in names:
            check(f"run {k}: hw_{s}_stencil.raw == {s} slice {list(scalars[s].shape)}",
                  np.array_equal(raw(s), bits(scalars[s]).reshape(-1)))
        check(f"run {k}: gold == {spec['layer']} slice", np.array_equal(g, bits(gold).reshape(-1)))


def check_mul10():
    from strait.coreir_backend.templates.elementwise_mul_bf16 import emit_elementwise_mul_bf16_design

    width, heads = 32768, 4
    emit_elementwise_mul_bf16_design(U, width * heads, WORK, mode="input_x_weight_broadcast", num_vecs=heads,
                                     top_module="rms_norm_pass2_fp")
    inst, conns, meta = load_design("rms_norm_pass2_fp")
    ios = {n for n, i in inst.items() if i.get("modref") == "global.IO"}
    in_io = sorted((n for n in ios if n.startswith("io16in_input_")), key=lane)
    w_io = sorted((n for n in ios if n.startswith("io16in_weight_")), key=lane)
    out_io = sorted((n for n in ios if n.startswith("io16_hw_output")), key=lane)
    check("IO counts 16 input / 16 weight / 16 output", (len(in_io), len(w_io), len(out_io)) == (16, 16, 16))
    check("weight DMA [2048,4]/[1,-2047] (one chunk-half scores block per head)",
          all(pat(meta(n, "glb2out_0"), "read") == ([2048, 4], [1, -2047]) for n in w_io))
    check("input / output DMA linear [8192]/[1]",
          all(pat(meta(n, "glb2out_0"), "read") == ([8192], [1]) for n in in_io)
          and all(pat(meta(n, "in2glb_0"), "write") == ([8192], [1]) for n in out_io))
    mul9 = load("mul_9", 4, 64, 256, 256)
    scores = load("unsqueeze_2", 4, 256, 256)
    mul10 = load("mul_10", 4, 64, 256, 256)
    for k in (0, 127):
        c, hg, q = k // 32, (k // 2) % 16, k % 2
        half = lambda a: a.reshape(*a.shape[:-1], 8, 2, 16)[..., q, :].reshape(*a.shape[:-2], width)
        dec, w, gold = half(mul9[c, 4 * hg:4 * hg + 4]), half(scores[c]), half(mul10[c, 4 * hg:4 * hg + 4])
        # simulation: the template's DMA walks on the E64 layout (lane = col mod 16, word = row * 2048 + col div 16)
        out = np.full(gold.shape, np.nan)
        for li, wi, oi in zip(in_io, w_io, out_io):
            j = lane(li)
            a = np.array([dec[wd // 2048, (wd % 2048) * 16 + j] for wd in walk(meta(li, "glb2out_0"), "read")], dtype=np.float64)
            b = np.array([w[wd * 16 + j] for wd in walk(meta(wi, "glb2out_0"), "read")], dtype=np.float64)
            for wd, v in zip(walk(meta(oi, "in2glb_0"), "write"), bf_rne(a * b)):
                out[wd // 2048, (wd % 2048) * 16 + j] = v
        check(f"run {k}: simulated design == gold mul_10 slice", np.array_equal(out, gold.astype(np.float64)),
              f"{100 * np.mean(out == gold):.3f}% equal")
        raw, g = parse_run(f"rms_norm_pass2_fp_mamba2_prefill_mul10_k{k}", "mul_10", [1, 4, 1, 64, 256, 256],
                           os.path.join(WORK, f"parser_k{k}"))
        check(f"run {k}: hw_input_stencil.raw == decay slice [4 x 32768]", np.array_equal(raw("input"), bits(dec).reshape(-1)))
        check(f"run {k}: hw_weight_stencil.raw == scores slice [32768]", np.array_equal(raw("weight"), bits(w).reshape(-1)))
        check(f"run {k}: gold == mul_10 slice", np.array_equal(g, bits(gold).reshape(-1)))
    model = bf_rne(scores[:, None].astype(np.float64) * mul9)
    check("bf16 RNE scores * decay == mul_10 (whole layer)", np.array_equal(model, mul10), f"{100 * np.mean(model == mul10):.4f}%")
    print(f"  PE fp_mul model == mul_10: {100 * np.mean(pe_mul(scores[:, None], mul9) == mul10):.3f}%")


if KERNEL in GATHER:
    check_gather(GATHER[KERNEL])
    if KERNEL == "x_end":
        x = load("reshape", 1024, 64, 64).astype(np.float64)
        dt = load("softplus", 1024, 64, 1).astype(np.float64)
        dte = load("transpose", 1024, 64, 1).astype(np.float64)
        m11 = load("mul_11", 1024, 64, 64)
        model = bf_rne(bf_rne(x * dt) * dte)
        check("bf16(bf16(x * dt) * decay_to_end) == mul_11 (whole layer)", np.array_equal(model, m11),
              f"{100 * np.mean(model == m11):.4f}%")
        print(f"  PE fp_mul model == mul_11: {100 * np.mean(pe_mul(pe_mul(x, dt), dte) == m11):.3f}%")
    else:
        sd = load("submodule_5", 4, 64)
        for layer, state, chunk in (("mul_12", "_tensor_constant4", 0), ("mul_13", "add_6", 1), ("mul_14", "add_7", 2)):
            s = load(state, 128, 64, 64).astype(np.float64)
            model = bf_rne(s * sd[chunk][None, :, None])
            ref = load(layer, 128, 64, 64)
            check(f"bf16 {state} * submodule_5[{chunk}] == {layer}", np.array_equal(model, ref), f"{100 * np.mean(model == ref):.4f}%")
        print(f"  mul_12's state is all zeros: {not load('_tensor_constant4', 128, 4096).any()}")
        print(f"  PE fp_mul model == mul_13: {100 * np.mean(pe_mul(load('add_6', 128, 64, 64), sd[1][None, :, None]) == load('mul_13', 128, 64, 64)):.3f}%")
elif KERNEL == "mul10":
    check_mul10()
else:
    raise SystemExit(f"unknown kernel {KERNEL}")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
