"""Host checks of the batched transposes on transpose_fp (strait transpose_bf16), before RTL: transpose
(decay_to_end), permute (a, from #1's 16 lanes) and permute_2 (B^T; permute_3 reads the same output).

1. Graph: slice_17 == mul_9[..., 255, :]; each output == its input with the last two dims swapped; permute_3 ==
   permute_2 (all bit-exact).
2. Template: the 2D designs byte-identical to the template before the change (a copy of it is the argument); the
   batched streams with batch 1 reproduce the 2D streams; the design of each key's pass as the hook emits it: counts,
   IO names, the one connected input IO, the MEM filter configs, the streams.
3. Dataflow: the GLB input image (natural lane layout) -> the pass's read walk -> the 32 filter MEMs -> the write walks:
   all passes == the output tensor; each key's pass alone == its rows, 0 elsewhere.
4. Parser: each key's env -> hw_input_stencil.raw == its GLB tile of the input image, gold == the simulated pass ==
   process.cpp's gold formula == run_mul_tiles.sh's tensors / model (mul_kernels_lib); IO positions.
Usage: check_transpose_host.py <workdir> <transpose_bf16.py before the change>   (run with /aha/bin/python)
"""
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_x_dt_host_lib import bits, walk  # noqa: E402
from mul_kernels_lib import TRANSPOSES, model, tensors  # noqa: E402

WORK, ORIG = sys.argv[1], sys.argv[2]
os.makedirs(WORK, exist_ok=True)
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
PARAMS = json.load(open("/aha/aha/util/application_parameters.json"))
U = 32
# layer: passes to check (the keys), expected read (extent, stride), write (start per pass, extent, stride), words / MEM
EXPECT = {
    "transpose": ((0, 31), ([64, 8, 4], [8, -503, 1]), (2, [2, 32], [1, 63]), 64),
    "permute": ((0, 15), ([256, 4, 4], [4, -1019, 1]), (8, [8, 16], [1, 121]), 128),
    "permute_2": ((0, 31), ([256, 4, 4], [4, -1019, 1]), (8, [8, 16], [1, 249]), 128),
}
fails = []
load = lambda node, *shape: np.fromfile(BASE + node + ".bin", dtype=np.float32).reshape(*shape)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


def image(m, lanes):
    """GLB image [words, lanes] of a [rows, cols] tensor in its natural lane layout (lane = col mod lanes)."""
    r, c = m.shape
    return m.reshape(r, c // lanes, lanes).reshape(r * (c // lanes), lanes)


print("== 1. graph", flush=True)
mul9 = np.fromfile(BASE + "mul_9.bin", dtype=np.float32).reshape(4, 64, 256, 256)
check("slice_17 == mul_9[..., 255, :] bit-exact", np.array_equal(bits(load("slice_17", 4, 64, 256)), bits(mul9[:, :, 255, :])))
del mul9
check("reshape_3 == mul_6 (a, #1's output) bit-exact", np.array_equal(bits(load("reshape_3", 1024, 64)), bits(load("mul_6", 1024, 64))))
for layer, (src, dst, b, X, Y, lanes) in TRANSPOSES.items():
    check(f"{dst} == {src} [{b}, {X}, {Y}] transposed bit-exact",
          np.array_equal(bits(load(dst, b, Y, X)), bits(load(src, b, X, Y).transpose(0, 2, 1))))
check("permute_3 == permute_2 bit-exact (one transpose serves both)",
      np.array_equal(bits(load("permute_3", 4, 128, 256)), bits(load("permute_2", 4, 128, 256))))

print("== 2. template", flush=True)
from strait.coreir_backend.coreir_backend import _io_logical_names_from_design_top  # noqa: E402
from strait.coreir_backend.templates import transpose_bf16 as N  # noqa: E402

spec = importlib.util.spec_from_file_location("transpose_bf16_orig", ORIG)
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)


def saved_md5(top, path):
    top.save_to_file(path)
    return hashlib.md5(open(path, "rb").read()).hexdigest()


same = []
for shape, k in (((512, 64), 0), ((512, 64), 1), ((512, 64), 31), ((128, 768), 5), ((128, 768), 0)):
    a = saved_md5(O.build_transpose_bf16_context(32, shape, k, k % 8, "input", "transpose")[1], os.path.join(WORK, "a.json"))
    b = saved_md5(N.build_transpose_bf16_context(32, shape, k, k % 8, "input", "transpose")[1], os.path.join(WORK, "b.json"))
    same.append(a == b)
check("2D designs ([512, 64] passes 0 / 1 / 31, [128, 768] passes 5 / 0) byte-identical to before", all(same))

ok = True
for k in (0, 1, 31):
    r, w, my = N.transpose_streams(U, (512, 64), k, batch=1, in_lanes=U)
    ok &= (r["read_data_starting_addr"], r["read_data_stride"], r["extent"]) == ([0], [2, -(2 * 511) + 1], [512, 2])
    ok &= (w["write_data_starting_addr"], w["write_data_stride"], w["extent"]) == ([16 * k], [1, 512 + 1 - 16], [16, 2])
    ok &= my == 64
check("batched streams with batch 1 == the 2D streams ([512, 64], passes 0 / 1 / 31)", ok)

key_of = lambda layer, k: f"transpose_fp_mamba2_prefill_{layer}_k{k}"
for layer, (passes, (rext, rstr), (wstep, wext, wstr), words) in EXPECT.items():
    _, _, b, X, Y, lanes = TRANSPOSES[layer]
    for k in passes:
        env = PARAMS[key_of(layer, k)]["default"]
        tag = f"{layer} k{k}"
        check(f"{tag}: key TRANSPOSE_* == ({b}, {X}, {Y}, {lanes}), kernel id == CGRA_RUN_IDX == {k}",
              tuple(int(env[f"TRANSPOSE_{n}"]) for n in ("BATCH", "X", "Y", "IN_LANES")) == (b, X, Y, lanes) and
              int(env["TRANSPOSE_KERNEL_ID"]) == int(env["CGRA_RUN_IDX"]) == k)
        d = os.path.join(WORK, f"{layer}_k{k}")
        os.makedirs(d, exist_ok=True)
        N.emit_transpose_bf16_design(int(dict(kv.split("=") for kv in env["HALIDE_GEN_ARGS"].split())["glb_i"]), (X, Y),
                                     k, d, batch=b, in_lanes=lanes)
        mod = json.load(open(os.path.join(d, "design_top.json")))["namespaces"]["global"]["modules"]["transpose_fp"]
        inst, conns = mod["instances"], mod["connections"]
        ref = lambda v: v.get("modref") or v.get("genref")
        cnt = Counter(ref(v) for v in inst.values())
        names = _io_logical_names_from_design_top(os.path.join(d, "design_top.json"))
        ins = sorted((n for n in inst if n.startswith("io16in_")), key=lambda n: int(n.split("_clkwrk_")[1].split("_")[0]))
        fan = [sum(1 for a_, b_ in conns if a_.startswith(n + ".") or b_.startswith(n + ".")) for n in ins]
        meta = lambda n, f: (lambda v: json.loads(v) if isinstance(v, str) else v)(inst[n]["metadata"][f])
        cfgs = [meta(f"mem_lane{i}", "lake_rv_config") for i in range(32)]
        r = meta(ins[0], "glb2out_0")
        w = meta("io16_hw_output_stencil_clkwrk_0_op_hcompute_stencil_0_write_0", "in2glb_0")
        check(f"{tag}: 32 MEMs, 0 PEs, 40 IOs; names input_host_stencil / hw_output_stencil; only IO {k % 8} drives the "
              f"MEMs", (cnt["cgralib.Mem"], cnt["global.PE"], cnt["global.IO"]) == (32, 0, 40) and
              names == (["input_host_stencil"], ["hw_output_stencil"]) and fan == [33 if i == k % 8 else 1 for i in range(8)],
              str(dict(cnt)))
        check(f"{tag}: MEM i keeps stream values i mod 32, {words} words",
              all(4 * c["output_glb_bank_idx"] + c["lane_idx_within_bank"] == i and c["unroll"] == U and
                  c["X"] * c["Y"] // U // U == words == b * X * Y // lanes // U for i, c in enumerate(cfgs)))
        check(f"{tag}: read {rext} / {rstr} start 0; write start {wstep * k} {wext} / {wstr}",
              (r["extent"], r["read_data_stride"], r["read_data_starting_addr"], w["extent"], w["write_data_stride"],
               w["write_data_starting_addr"]) == (rext, rstr, [0], wext, wstr, [wstep * k]))


def simulate(g_in, X, Y, batch, lanes, passes):
    """The pass's read walk on its lane -> MEM i keeps stream values i mod 32 -> the write walk of each output lane."""
    out = np.zeros((batch * Y * X // U, U), dtype=g_in.dtype)
    for k in passes:
        r, w, my = N.transpose_streams(U, (X, Y), k, batch=batch, in_lanes=lanes)
        stream = g_in[walk(r, "read"), k]
        n_mem = X * my // U // U
        assert len(stream) == n_mem * U
        wa = walk(w, "write")
        for i in range(U):
            out[wa, i] = stream[i::U][:n_mem]
    return out


print("== 3. dataflow", flush=True)
sims = {}
for layer, (src, dst, b, X, Y, lanes) in TRANSPOSES.items():
    g_in = image(load(src, b * X, Y), lanes)
    want = load(dst, b * Y, X)
    check(f"{layer}: all {lanes} passes == {dst} (natural layout) bit-exact",
          np.array_equal(bits(simulate(g_in, X, Y, b, lanes, range(lanes))), bits(image(want, U))))
    for k in EXPECT[layer][0]:
        sims[layer, k] = simulate(g_in, X, Y, b, lanes, [k])
        part = np.zeros_like(want)
        rows = (np.arange(b)[:, None] * Y + np.arange(k, Y, lanes)[None, :]).reshape(-1)
        part[rows] = want[rows]
        check(f"{layer} pass {k} alone == {dst} rows y = {k} mod {lanes}, 0 elsewhere",
              np.array_equal(bits(sims[layer, k]), bits(image(part, U))))

print("== 4. parser", flush=True)
import parse_dnnLayer_tensors as P  # noqa: E402

for layer, (src, dst, b, X, Y, lanes) in TRANSPOSES.items():
    g_in = image(load(src, b * X, Y), lanes)
    for k in EXPECT[layer][0]:
        env = PARAMS[key_of(layer, k)]["default"]
        tag = f"{layer} k{k}"
        pd = os.path.join(WORK, f"parser_{layer}_k{k}")
        os.makedirs(pd, exist_ok=True)
        saved = dict(os.environ)
        os.environ.update(env)
        try:
            zw = P.parse_zircon_workarounds()
            P.parse_extra_inputs_bf16_cgra(BASE, pd, zw)
            shape = json.load(open(f"/aha/voyager/hardcoded_tensor_metadata/mamba2_prefill-{dst}_tensor_metadata.json"))
            gold = np.array(P.parse_output("mamba2_prefill", dst, BASE, shape["outputs"][0], zw), dtype=np.int64) & 0xFFFF
        finally:
            os.environ.clear()
            os.environ.update(saved)
        raw = np.fromfile(os.path.join(pd, "hw_input_stencil.raw"), dtype=">u2").astype(np.int64)
        T = k // 8
        check(f"{tag}: hw_input_stencil.raw == GLB tile {T} of the input image ([{raw.size // 8} words x 8 lanes])",
              np.array_equal(raw, bits(g_in[:, 8 * T:8 * T + 8]).reshape(-1)))
        check(f"{tag}: parser gold == the simulated pass", np.array_equal(gold, bits(sims[layer, k]).reshape(-1)))
        raws, lib_gold = tensors(layer, k)
        check(f"{tag}: process.cpp's gold formula (mul_kernels_lib model) == run_mul_tiles' tensors == the parser",
              np.array_equal(bits(raws["input"]).reshape(-1), raw) and np.array_equal(bits(lib_gold).reshape(-1), gold) and
              np.array_equal(bits(model(layer, raws, None, k).astype(np.float32)).reshape(-1), gold))
        check(f"{tag}: input IO at x {12 + 2 * T}, {13 + 2 * T}; output x 20-27; HALIDE_GEN_ARGS output [{X} x {b * Y}]",
              env["INPUT_HOST_STENCIL_POS"].split(",") == [str(12 + 2 * T)] * 4 + [str(13 + 2 * T)] * 4 and
              env["HW_OUTPUT_POS"].split(",") == [str(x) for x in range(20, 28) for _ in range(4)] and
              env["HALIDE_GEN_ARGS"] == f"vec_width={X} vec_height={b * Y} glb_i=32")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + '; '.join(fails)}", flush=True)
sys.exit(1 if fails else 0)
