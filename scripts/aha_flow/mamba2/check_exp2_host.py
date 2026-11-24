"""Host checks of Mamba2 exp_2 = exp(a_cum) on mamba_exp_fp (strait elementwise_sub_exp_bf16, subtract=False, clamp -127,
rne), before RTL.

1. Graph: exp_2 == bf16(exp(cumsum)); submodule_5 == exp_2[..., 255] (so submodule_5 needs no kernel).
2. Template: stable_softmax_pass2_fp's design byte-identical to the one emitted before the change (md5 argument); the
   exp_2 design as the hook emits it: counts, no vec_max IO, ROM table, linear walks, the PE chain wiring.
3. Dataflow: the input walk -> the exp table -> the output walk == the table on a_cum.
4. Parser: the key's env -> hw_input_stencil.raw == cumsum bits, gold == exp_2 bits.
5. Accuracy: the exp table on a_cum vs exp_2 by range; every wrap-range value -> 0; test.py predicted.
Usage: check_exp2_host.py <workdir> <softmax reference md5>   (run with /aha/bin/python)
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
from decay_chain_lib import INV_LN2_BITS, exp_chain_lut, exp_rom, f32  # noqa: E402

WORK, SOFTMAX_MD5 = sys.argv[1], sys.argv[2]
os.makedirs(WORK, exist_ok=True)
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
PARAMS = json.load(open("/aha/aha/util/application_parameters.json"))
KEY = "mamba_exp_fp_mamba2_prefill_exp2"
ATOL, RTOL, FRAC = 1.5e-4, 0.2, 6e-2
fails = []
load = lambda node, *shape: np.fromfile(BASE + node + ".bin", dtype=np.float32).reshape(*shape)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


print("== 1. graph", flush=True)
acum, exp2 = load("cumsum", 4, 64, 256), load("exp_2", 4, 64, 256)
check("exp_2 == bf16(exp(cumsum))", np.array_equal(bits(exp2), bits(bf_rne(np.exp(acum.astype(np.float64))))))
sub5 = load("submodule_5", 4, 64)
check("submodule_5 == exp_2[..., 255] bit-exact", np.array_equal(bits(sub5), bits(exp2[..., 255])))
print(f"      a_cum min {acum.min():.2f}; values with fp_mul(a_cum, 1/ln2) < -126: "
      f"{int(np.sum(pe_mul(acum, f32(INV_LN2_BITS)) < -126))}, at the chunk end (i = 255): "
      f"{int(np.sum(pe_mul(acum[..., 255], f32(INV_LN2_BITS)) < -126))}", flush=True)

print("== 2. template", flush=True)
from strait.coreir_backend.templates.elementwise_sub_exp_bf16 import emit_elementwise_sub_exp_bf16_design  # noqa: E402

d = os.path.join(WORK, "softmax")
os.makedirs(d, exist_ok=True)
emit_elementwise_sub_exp_bf16_design(32, 512, 512, d)
md5 = hashlib.md5(open(os.path.join(d, "design_top.json"), "rb").read()).hexdigest()
check("stable_softmax_pass2_fp design (32 lanes, 512 x 512) byte-identical to before the change", md5 == SOFTMAX_MD5, md5)
env = PARAMS[KEY]["default"]
gen = dict(kv.split("=") for kv in env["HALIDE_GEN_ARGS"].split())
d = os.path.join(WORK, "exp2")
os.makedirs(d, exist_ok=True)
emit_elementwise_sub_exp_bf16_design(int(gen["glb_i"]), int(gen["vec_width"]), int(gen["vec_height"]), d, subtract=False,
                                     clamp=float(env["MAMBA_EXP_CLAMP"]), rne=True, top_module="mamba_exp_fp")
mod = json.load(open(os.path.join(d, "design_top.json")))["namespaces"]["global"]["modules"]["mamba_exp_fp"]
inst, conns = mod["instances"], mod["connections"]
ref = lambda v: v.get("modref") or v.get("genref")
cnt = Counter("ROM" if ref(v) == "cgralib.Mem" and v["metadata"].get("is_rom") else ref(v) for v in inst.values())
check("counts: 160 PEs, 32 ROMs, no other MEM, 64 IOs", (cnt["global.PE"], cnt["ROM"], cnt["cgralib.Mem"], cnt["global.IO"])
      == (160, 32, 0, 64), str(dict(cnt)))
check("no vec_max IO", not any("vec_max" in n for n in inst))
check("ROMs hold the RNE exp table (== the decay chain's)",
      all(inst[f"exp_rom_lane{i}"]["metadata"]["init"] == list(exp_rom()) for i in range(32)))
edges = {(a.split(".")[0], b.split(".")[0]) for a, b in conns} | {(b.split(".")[0], a.split(".")[0]) for a, b in conns}
chain = [("io16in_input_host_stencil_clkwrk_0_op_hcompute_input_glb_stencil_read_0", "mul_b_pe_0"),
         ("mul_b_pe_0", "clamp_pe_0"), ("clamp_pe_0", "getffrac_pe_0"), ("clamp_pe_0", "getfint_pe_0"),
         ("getffrac_pe_0", "exp_rom_lane0"), ("exp_rom_lane0", "addiexp_pe_0"), ("getfint_pe_0", "addiexp_pe_0")]
check("lane 0 chain input -> mul_b -> clamp -> getffrac / getfint -> ROM -> addiexp", all(e in edges for e in chain))
meta = lambda n, key: (lambda v: json.loads(v) if isinstance(v, str) else v)(inst[n]["metadata"].get(key))
lane_of = lambda n: int(n.split("_clkwrk_")[1].split("_")[0])
ins = [n for n in inst if n.startswith("io16in_")]
outs = [n for n in inst if n.startswith("io16_")]
check("32 input and 32 output lane IOs, linear [2048] / [1]", len(ins) == len(outs) == 32 and
      all(walk(meta(n, "glb2out_0"), "read") == list(range(2048)) for n in ins) and
      all(walk(meta(n, "in2glb_0"), "write") == list(range(2048)) for n in outs))

print("== 3. dataflow", flush=True)
lut = exp_chain_lut()
x = acum.reshape(-1).astype(np.float64)
gl_in = x.reshape(2048, 32)                    # word w, lane l = value 32 w + l (row-major, 32 lanes)
gl_out = np.zeros((2048, 32))
for n in outs:
    lane = lane_of(n)
    src = gl_in[walk(meta([m for m in ins if lane_of(m) == lane][0], "glb2out_0"), "read"), lane]
    gl_out[walk(meta(n, "in2glb_0"), "write"), lane] = f32(lut[bits(src)])
model = f32(lut[bits(x)])
check("simulated design == the table on a_cum", np.array_equal(gl_out.reshape(-1), model))
check("chunk-end values (i = 255) in lane 31 at word 512 c + 8 h + 7",
      all(np.argwhere(np.arange(65536).reshape(4, 64, 256)[..., 255].reshape(-1, 1) == np.arange(65536).reshape(2048, 32)
                      [None].reshape(1, -1))[:, 1] % 32 == 31) and
      np.array_equal(np.arange(65536).reshape(4, 64, 256)[..., 255].reshape(-1) // 32,
                     (512 * np.arange(4)[:, None] + 8 * np.arange(64)[None] + 7).reshape(-1)))

print("== 4. parser", flush=True)
import parse_dnnLayer_tensors as P  # noqa: E402

pd = os.path.join(WORK, "parser")
os.makedirs(pd, exist_ok=True)
saved = dict(os.environ)
os.environ.update(env)
try:
    zw = P.parse_zircon_workarounds()
    P.parse_extra_inputs_bf16_cgra(BASE, pd, zw)
    gold = np.array(P.parse_output("mamba2_prefill", "exp_2", BASE,
                                   {"node": "exp_2", "shape": [1, 4, 1, 64, 256], "dtype": "bfloat16"}, zw),
                    dtype=np.int64) & 0xFFFF
finally:
    os.environ.clear()
    os.environ.update(saved)
raw = np.fromfile(os.path.join(pd, "hw_input_stencil.raw"), dtype=">u2").astype(np.int64)
check("parser hw_input_stencil.raw == cumsum bits", np.array_equal(raw, bits(acum).reshape(-1)))
check("parser gold == exp_2 bits", np.array_equal(gold, bits(exp2).reshape(-1)))

print("== 5. accuracy vs voyager", flush=True)
g = exp2.reshape(-1).astype(np.float64)
nz = g != 0
rel = np.zeros_like(g)
rel[nz] = np.abs(model[nz] - g[nz]) / np.abs(g[nz])
print(f"      bit-exact {100 * np.mean(model == g):.3f}%, mean rel {100 * rel.mean():.3f}%, p99 {100 * np.percentile(rel, 99):.3f}%, "
      f"max {100 * rel.max():.3f}%", flush=True)
for lo, hi in ((0, -1), (-1, -5), (-5, -20), (-20, -50), (-50, -87.3), (-87.3, -1e9)):
    sel = (x <= lo) & (x > hi)
    if sel.any():
        print(f"      a_cum in ({hi:g}, {lo:g}]: {int(sel.sum()):>6} values, bit-exact {100 * np.mean(model[sel] == g[sel]):7.3f}%, "
              f"mean rel {100 * rel[sel].mean():7.3f}%, max {100 * rel[sel].max():8.3f}%", flush=True)
wrapz = pe_mul(x, f32(INV_LN2_BITS)) <= -127
check("every clamped value (z <= -127) -> exactly 0", np.all(model[wrapz] == 0), f"{int(wrapz.sum())} values")
exc = np.abs(model - g) > ATOL + RTOL * np.abs(g)
f = exc.mean()
print(f"      predicted test.py: {'PASS' if f == 0 else ('mostly passed' if f <= FRAC else 'FAIL')} "
      f"({int(exc.sum())} beyond tolerance, {100 * f:.3f}%)", flush=True)
check("predicted test.py exit 0", f <= FRAC)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + '; '.join(fails)}", flush=True)
sys.exit(1 if fails else 0)
