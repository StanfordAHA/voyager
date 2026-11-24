"""Host checks of the Mamba2 #7 rework (mul_13 on mamba_state_decay_fp, the chunk decays read through one lane of exp_2's
output GLB tile 3 instead of a gather tree), before RTL.

1. Graph: exp_2's output GLB tile 3, lane 7, word 512 c + 8 h + 7 == exp_2[c, h, 255] == submodule_5[c, h]; chunk 1's
   == unsqueeze_7 (the tree runs' decay), all bit-exact.
2. Template: the tree designs (#2 x_dt, #2 chip dt, #6 x_end, #7 tree) byte-identical to before the change (md5
   arguments); the lane design as the hook emits it for runs 0 and 7: counts, only lane 7 connected, the decay walk
   (word 512 + 8 h + 7 per head, held 2 beats, the same for every row), linear state / output walks.
3. Dataflow: simulate the design's wiring (DMA walks into the run's GLB tiles, register and buffers, the PE multiply)
   and rebuild the output from the output walk; == the PE model of the run.
4. Parser: the keys' env -> raws == numpy slices, gold == the mul_13 slice.
5. Accuracy: the exact-RNE bf16 model == voyager's mul_13 slice; the PE fp_mul model's agreement.
Usage: check_state_decay_lane_host.py <workdir> <md5 x_dt> <md5 x_dt_chip> <md5 x_end> <md5 state_decay>
"""
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_x_dt_host_lib import bits, bf_rne, pe_mul, walk  # noqa: E402
from mul_kernels_lib import model, tensors  # noqa: E402

WORK = sys.argv[1]
MD5 = dict(zip(("x_dt", "x_dt_chip", "x_end", "state_decay"), sys.argv[2:6]))
os.makedirs(WORK, exist_ok=True)
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
PARAMS = json.load(open("/aha/aha/util/application_parameters.json"))
KEY = "mamba_state_decay_fp_mamba2_prefill_mul13_lane_k{k}"
RUNS = (0, 7)
fails = []
load = lambda node, *shape: np.fromfile(BASE + node + ".bin", dtype=np.float32).reshape(*shape)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


print("== 1. graph", flush=True)
exp2 = load("exp_2", 4, 64, 256)
tile3 = load("exp_2", 256, 8, 4, 8)[:, :, 3, :].reshape(2048, 8)
w = 512 * np.arange(4)[:, None] + 8 * np.arange(64)[None] + 7
check("exp_2 tile 3 lane 7 word 512 c + 8 h + 7 == exp_2[c, h, 255]", np.array_equal(bits(tile3[w, 7]), bits(exp2[..., 255])))
check("== submodule_5", np.array_equal(bits(tile3[w, 7]), bits(load("submodule_5", 4, 64))))
check("chunk 1 == unsqueeze_7 (the decay of mul_13)", np.array_equal(bits(tile3[w[1], 7]), bits(load("unsqueeze_7", 64))))

print("== 2. template", flush=True)
from strait.coreir_backend.templates.elementwise_mul_head_gather_bf16 import \
    emit_elementwise_mul_head_gather_bf16_design as emit  # noqa: E402

TREES = {
    "x_dt": dict(args=(16, 8, 256), kw=dict(scalars=(("dt", 1, 0),))),
    "x_dt_chip": dict(args=(16, 8, 256), kw=dict(scalars=(("dt", 2, 1537),))),
    "x_end": dict(args=(16, 8, 256), kw=dict(scalars=(("dt", 1), ("decay", 1)), slots=1, heads_first_output=False,
                                             top_module="mamba_x_end_fp")),
    "state_decay": dict(args=(16, 8, 128), kw=dict(scalars=(("decay", 0),), slots=2, heads_first_output=False,
                                                   top_module="mamba_state_decay_fp")),
}
for name, cfg in TREES.items():
    d = os.path.join(WORK, f"tree_{name}")
    os.makedirs(d, exist_ok=True)
    emit(*cfg["args"], d, **cfg["kw"])
    md5 = hashlib.md5(open(os.path.join(d, "design_top.json"), "rb").read()).hexdigest()
    check(f"tree design {name} byte-identical to before the change", md5 == MD5[name], md5)

meta = lambda inst, n, key: (lambda v: json.loads(v) if isinstance(v, str) else v)(inst[n]["metadata"].get(key))
lane_of = lambda n: int(n.split("_clkwrk_")[1].split("_")[0])
designs = {}
for k in RUNS:
    v, q = k // 2, k % 2
    env = PARAMS[KEY.format(k=k)]["default"]
    gen = dict(kv.split("=") for kv in env["HALIDE_GEN_ARGS"].split())
    d = os.path.join(WORK, f"lane_k{k}")
    os.makedirs(d, exist_ok=True)
    lane = {"lane": int(env["MUL_GATHER_SCALAR_LANE"]), "lanes": int(env["MUL_GATHER_SCALAR_LANES"]),
            "head_stride": int(env["MUL_GATHER_SCALAR_HEAD_STRIDE"])}
    emit(int(gen["glb_i"]), int(env["MUL_GATHER_NUM_HEADS"]), int(gen["vec_height"]), d,
         scalars=(("decay", 0, int(env["MUL_GATHER_SCALAR_START"]), lane),), slots=int(env["MUL_GATHER_SLOTS"]),
         heads_first_output=False, top_module="mamba_state_decay_fp")
    mod = json.load(open(os.path.join(d, "design_top.json")))["namespaces"]["global"]["modules"]["mamba_state_decay_fp"]
    inst, conns = mod["instances"], mod["connections"]
    designs[k] = (inst, conns)
    ref = lambda x: x.get("modref") or x.get("genref")
    cnt = Counter(ref(x) for x in inst.values())
    check(f"k{k} counts: 16 PEs, 24 MEMs, 1 register, 40 IOs, no interleave MEM",
          (cnt["global.PE"], cnt["cgralib.Mem"], cnt["coreir.reg"], cnt["global.IO"]) == (16, 24, 1, 40)
          and not any("gather" in n for n in inst), str(dict(cnt)))
    dio = sorted((n for n in inst if n.startswith("io16in_decay")), key=lane_of)
    used = sorted({lane_of(a.split(".")[0]) for a, b in conns for a, b in ((a, b), (b, a))
                   if a.startswith("io16in_decay") and a.endswith(".out")})
    check(f"k{k} decay: 8 lane IOs, only lane 7 connected (to decay_in_reg)", len(dio) == 8 and used == [7] and
          any({a, b} == {f"{dio[7]}.out", "decay_in_reg.in"} for a, b in conns), f"connected {used}")
    sd = meta(inst, dio[0], "glb2out_0")
    exp_w = [512 + 8 * (16 * v + hh) + 7 for i in range(128) for hh in range(16) for g in range(2)]
    check(f"k{k} decay walk == word 512 + 8 (16 v + hh) + 7, held 2 beats, every row",
          walk(sd, "read") == exp_w and all(meta(inst, n, "glb2out_0") == sd for n in dio)
          and sd.get("tb_write_starting_addr") == [0], f"{sd['extent']} / {sd['read_data_stride']} from {sd['read_data_starting_addr']}")
    lin = list(range(4096))
    check(f"k{k} state / output walks linear [32, 128]", all(walk(meta(inst, n, "glb2out_0"), "read") == lin
          for n in inst if n.startswith("io16in_input")) and all(walk(meta(inst, n, "in2glb_0"), "write") == lin
          for n in inst if n.startswith("io16_hw_output")))

print("== 3. dataflow", flush=True)
for k in RUNS:
    inst, conns = designs[k]
    drv = {}
    for a, b in conns:
        for src, dst in ((a, b), (b, a)):
            sn, _, sp = src.partition(".")
            dn, _, dp = dst.partition(".")
            if sp in ("out", "O0", "data_out_0", "data_out_1") and dp in ("in", "data0", "data1", "data_in_0") \
                    and not sn.startswith("const"):
                drv[(dn, dp)] = (sn, sp)
    raws, gold = tensors("state_decay_lane", k)
    tiles = {"input": raws["input"].astype(np.float64).reshape(4096, 16), "decay": raws["decay"].astype(np.float64)}

    def stream(n):
        x = inst[n]
        r = x.get("modref") or x.get("genref")
        if r == "global.IO":
            name = n[len("io16in_"):].split("_host_stencil")[0]
            return tiles[name][walk(meta(inst, n, "glb2out_0"), "read"), lane_of(n)]
        if r in ("coreir.reg", "cgralib.Mem"):
            return stream(drv[(n, "in" if r == "coreir.reg" else "data_in_0")][0])
        if r == "global.PE":
            return pe_mul(stream(drv[(n, "data0")][0]), stream(drv[(n, "data1")][0]))
        raise ValueError(r)

    gl = np.zeros((4096, 16))
    for n in inst:
        if n.startswith("io16_hw_output"):
            gl[walk(meta(inst, n, "in2glb_0"), "write"), lane_of(n)] = stream(drv[(n, "in")][0])
    m = model("state_decay_lane", raws, pe_mul, k).reshape(-1)
    check(f"k{k} simulated design == PE model (100%)", np.array_equal(gl.reshape(-1), m), f"{100 * np.mean(gl.reshape(-1) == m):.4f}%")

print("== 4. parser", flush=True)
import parse_dnnLayer_tensors as P  # noqa: E402

for k in RUNS:
    d = os.path.join(WORK, f"parser_k{k}")
    os.makedirs(d, exist_ok=True)
    saved = dict(os.environ)
    os.environ.update(PARAMS[KEY.format(k=k)]["default"])
    try:
        zw = P.parse_zircon_workarounds()
        P.parse_extra_inputs_bf16_cgra(BASE, d, zw)
        g = np.array(P.parse_output("mamba2_prefill", "mul_13", BASE,
                                    {"node": "mul_13", "shape": [1, 1, 128, 64, 64], "dtype": "bfloat16"}, zw),
                     dtype=np.int64) & 0xFFFF
    finally:
        os.environ.clear()
        os.environ.update(saved)
    raws, gold = tensors("state_decay_lane", k)
    for name, t in raws.items():
        r = np.fromfile(os.path.join(d, f"hw_{name}_stencil.raw"), dtype=">u2").astype(np.int64)
        check(f"k{k} parser hw_{name}_stencil.raw == numpy slice {tuple(t.shape)}", np.array_equal(r, bits(t).reshape(-1)))
    v, q = k // 2, k % 2
    indep = load("mul_13", 128, 64, 64)[:, 16 * v:16 * v + 16].reshape(128, 16, 2, 2, 16)[:, :, :, q, :].reshape(-1)
    check(f"k{k} parser gold == mul_13[:, heads 16 v .., half q]", np.array_equal(g, bits(indep)) and
          np.array_equal(bits(gold).reshape(-1), bits(indep)))

print("== 5. accuracy", flush=True)
for k in RUNS:
    raws, gold = tensors("state_decay_lane", k)
    rne = model("state_decay_lane", raws, lambda a, b: bf_rne(np.asarray(a, np.float64) * np.asarray(b, np.float64)), k)
    pe = model("state_decay_lane", raws, pe_mul, k)
    check(f"k{k} exact-RNE bf16 model == voyager mul_13 slice", np.array_equal(bits(rne), bits(gold)),
          f"{100 * np.mean(bits(rne) == bits(gold)):.3f}%; PE model == voyager {100 * np.mean(bits(pe) == bits(gold)):.3f}%")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + '; '.join(fails)}", flush=True)
sys.exit(1 if fails else 0)
