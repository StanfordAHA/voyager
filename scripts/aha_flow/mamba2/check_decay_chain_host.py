"""Host checks of the Mamba2 fused decay chain #3 (mul_8 -> cumsum_1 -> exp -> mul_9 on mamba_decay_chain_fp, strait
masked_col_cumsum_exp_bf16), before RTL.

1. Graph identities (layer 0): a = permute == mul_6 rearranged; the masks are exactly 0/1 strict-lower / lower;
   mul_8 == a * strict; exp == bf16(exp(cumsum_1)); mul_9 == exp * lower.
2. Exp chain: numpy getffrac / getfint / addiexp == lassen's functional model on every pattern; the table's edges (S = 0
   -> 1.0, every clamped z -> 0).
3. Template: emit the design for runs 0 and 127 as the hook does (the aha keys' env); instance counts, the one connected
   a lane, PE FIFO bypass config, ROM table, lake configs and the IO DMA walks.
4. Dataflow: simulate the generated design's wiring (DMA walks into the run's GLB tiles, register and buffers, the
   Pond recurrence by its config, the ROM, the PE ops by role) and rebuild the output GLB tensor from the output walk.
   Must == the host model of the run (decay_chain_lib.run_model).
5. Parser: the aha keys' env through parse_extra_inputs_bf16_cgra / parse_output == independent numpy slices.
6. Accuracy (whole layer 0): the host model (PE fp_mul rounding) vs voyager's mul_9, by segment-sum range, and test.py's
   comparison (atol 1.5e-4, rtol 0.2, <= 6% exceptions) predicted for runs 0 and 127 and for all 128 runs.
Usage: check_decay_chain_host.py <workdir>   (run with /aha/bin/python)
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_x_dt_host_lib import bits, bf_rne, pe_mul, walk  # noqa: E402
from decay_chain_lib import (CLAMP, INV_LN2_BITS, cross_check_numpy_ops, exp_chain_lut, exp_rom, f32,  # noqa: E402
                             run_digits, run_model, scan)
from mul_kernels_lib import tensors  # noqa: E402

WORK = sys.argv[1]
os.makedirs(WORK, exist_ok=True)
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
PARAMS = json.load(open("/aha/aha/util/application_parameters.json"))
KEY = "mamba_decay_chain_fp_mamba2_prefill_k{k}"
RUNS = (0, 127)
ATOL, RTOL, FRAC = 1.5e-4, 0.2, 6e-2
fails = []
load = lambda node, *shape: np.fromfile(BASE + node + ".bin", dtype=np.float32).reshape(*shape)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


def with_env(key_name, fn):
    saved = dict(os.environ)
    os.environ.update(PARAMS[key_name]["default"])
    try:
        return fn()
    finally:
        os.environ.clear()
        os.environ.update(saved)


# 1. graph identities -------------------------------------------------------------------------------------------------
print("== 1. graph identities (layer 0)", flush=True)
mul6 = load("mul_6", 1024, 64)
a = load("permute", 4, 64, 256)
check("permute[c, h, i] == mul_6[256 c + i, h]", np.array_equal(a, mul6.reshape(4, 256, 64).transpose(0, 2, 1)))
strict, lower = load("_tensor_constant2", 256, 256), load("_tensor_constant3", 256, 256)
ii, jj = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
check("_tensor_constant2 == (i > j), _tensor_constant3 == (i >= j)",
      np.array_equal(strict, (ii > jj).astype(np.float32)) and np.array_equal(lower, (ii >= jj).astype(np.float32)))
print(f"      a range [{a.min():.4f}, {a.max():.6f}]", flush=True)
mul8 = load("mul_8", 4, 64, 256, 256)
check("mul_8 == a * strict", np.array_equal(mul8, a[..., None] * strict[None, None]))
cs1 = load("cumsum_1", 4, 64, 256, 256)
ex = load("exp", 4, 64, 256, 256)
check("exp == bf16(exp(cumsum_1))", np.array_equal(bits(ex), bits(bf_rne(np.exp(cs1.astype(np.float64))))),
      f"{100 * np.mean(bits(ex) == bits(bf_rne(np.exp(cs1.astype(np.float64))))):.4f}%")
mul9 = load("mul_9", 4, 64, 256, 256)
check("mul_9 == exp * lower", np.array_equal(mul9, ex * lower[None, None]))
print(f"      cumsum_1 min {cs1.min():.2f}; segment sums with S / ln2 < -126 (clamped): "
      f"{int(np.sum(cs1 * 1.4453125 < -126))}", flush=True)
del mul8, ex

# 2. exp chain --------------------------------------------------------------------------------------------------------
print("== 2. exp chain table", flush=True)
mism = cross_check_numpy_ops()
check("numpy getffrac / getfint / addiexp == lassen FPCustom on every pattern", mism == (0, 0, 0), f"mismatches {mism}")
lut = exp_chain_lut()
s_all = np.arange(65536)
neg = (s_all >> 15 == 1) & (((s_all >> 7) & 0xFF) != 0xFF)
sv = f32(s_all)
zv = pe_mul(sv, f32(INV_LN2_BITS))
check("table(+0) == table(-0) == 1.0", lut[0] == 0x3F80 and lut[0x8000] == 0x3F80)
check("every S with fp_mul(S, 1/ln2) <= -127 -> exactly 0", np.all(lut[neg & (zv <= -127)] == 0),
      f"{int(np.sum(neg & (zv <= -127)))} patterns")
unclamped = exp_chain_lut(clamp=-1e30)
wrap = neg & (zv < -127) & (f32(unclamped) > 1.0)
print(f"      without the clamp {int(np.sum(wrap))} negative-S patterns would give > 1 (exponent wrap)", flush=True)
np.save(os.path.join(WORK, "exp_chain_lut.npy"), lut)

# 3. template ---------------------------------------------------------------------------------------------------------
print("== 3. template", flush=True)
from strait.coreir_backend.templates.masked_col_cumsum_exp_bf16 import emit_masked_col_cumsum_exp_bf16_design  # noqa
from strait.coreir_backend.templates.elementwise_broadcast_add_softplus_bf16 import bf16_bits_rne, exp_rom_table  # noqa
check("lib constants == strait's (1/ln2, ROM)", bf16_bits_rne(1 / np.log(2)) == INV_LN2_BITS
      and list(exp_rom()) == exp_rom_table() and INV_LN2_BITS == 0x3FB9)
TOP = "mamba_decay_chain_fp"


def emit(k):
    env = PARAMS[KEY.format(k=k)]["default"]
    gen = dict(kv.split("=") for kv in env["HALIDE_GEN_ARGS"].split())
    d = os.path.join(WORK, f"design_k{k}")
    os.makedirs(d, exist_ok=True)
    unroll, vec_width, vec_height = int(gen["glb_i"]), int(gen["vec_width"]), int(gen["vec_height"])
    emit_masked_col_cumsum_exp_bf16_design(unroll, vec_width // unroll, 256, vec_height // 256, d,
                                           a_lane=int(env["DECAY_CHAIN_A_LANE"]), a_start=int(env["DECAY_CHAIN_A_START"]))
    design = json.load(open(os.path.join(d, "design_top.json")))
    mod = design["namespaces"]["global"]["modules"][TOP]
    return d, mod["instances"], mod["connections"], env


meta = lambda inst, n, key: (lambda v: json.loads(v) if isinstance(v, str) else v)(inst[n].get("metadata", {}).get(key))
lane_of = lambda n: int(n.split("_clkwrk_")[1].split("_")[0])


def drivers(conns):
    out = {}
    for x, y in conns:
        for src, dst in ((x, y), (y, x)):
            sn, _, sp = src.partition(".")
            dn, _, dp = dst.partition(".")
            if sp in ("out", "O0", "data_out_0", "data_out_1", "data_out_pond_0", "data_out_pond_1") and \
                    dp in ("in", "data0", "data1", "data_in_0", "data_in_1", "data_in_pond_0", "addr_in_0", "inst") \
                    and not sn.startswith("const_inst"):
                out[(dn, dp)] = (sn, sp)
    return out


designs = {}
for k in RUNS:
    c, t, jl, q = run_digits(k)
    d, inst, conns, env = emit(k)
    designs[k] = (inst, conns, env)
    ref = lambda v: v.get("modref") or v.get("genref")
    cnt = Counter("ROM" if ref(v) == "cgralib.Mem" and v["metadata"].get("is_rom") else ref(v) for v in inst.values())
    check(f"k{k} counts: 128 PEs, 16 Ponds, 16 ROMs, 40 MEMs (8 a buffers, 16 z buffers, 16 FIFOs), 56 IOs",
          (cnt["global.PE"], cnt["cgralib.Pond"], cnt["ROM"], cnt["cgralib.Mem"], cnt["global.IO"]) == (128, 16, 16, 40, 56),
          str(dict(cnt)))
    zdrv = drivers(conns)
    check(f"k{k} z fork buffers: clamp -> lane<i>_z_buffer; port 0 -> getffrac, port 1 -> getfint (16 lanes)",
          all(zdrv[(f"lane{i}_z_buffer", "data_in_0")] == (f"lane{i}_clamp_pe", "O0")
              and zdrv[(f"lane{i}_getffrac_pe", "data0")] == (f"lane{i}_z_buffer", "data_out_0")
              and zdrv[(f"lane{i}_getfint_pe", "data0")] == (f"lane{i}_z_buffer", "data_out_1") for i in range(16))
          and meta(inst, "lane0_z_buffer", "lake_rv_config") == {"type": "dual_read", "input_stream_size": 8192,
                                                                   "row_size": 8})
    drv = drivers(conns)
    a_ios = sorted((n for n in inst if n.startswith("io16in_a_host")), key=lane_of)
    used = [lane_of(n) for n in a_ios if any(src[0] == n for src in drv.values())]
    check(f"k{k} a: 8 lane IOs, only lane {jl} drives the register", len(a_ios) == 8 and used == [jl]
          and drv[("a_in_reg", "in")][0] == a_ios[jl], f"connected {used}")
    mulb = sorted(n for n in inst if "mulb" in n and (inst[n].get("modref") == "global.PE"))
    check(f"k{k} the 16 x 1/ln2 PEs are named *_mulb_bias_pe (kept out of Pond packing)",
          mulb == sorted(f"lane{i}_mulb_bias_pe" for i in range(16)))
    byp = json.load(open(os.path.join(d, "PE_fifos_bypass_config.json")))
    check(f"k{k} PE FIFO bypass: the 16 sum PEs, output FIFO bypassed", sorted(byp) == sorted(f"lane{i}_sum_pe" for i in range(16))
          and all(v == {"input_fifo_bypass": [0, 0, 0], "output_fifo_bypass": 1} for v in byp.values()))
    rom_ok = all(inst[f"lane{i}_exp_rom"]["metadata"]["init"] == exp_rom_table() for i in range(16))
    check(f"k{k} ROMs hold 2^(k/128) (RNE), no 0 word", rom_ok and 0 not in exp_rom_table())
    cfgs = {meta(inst, n, "lake_rv_config").get("type") for n in inst if meta(inst, n, "lake_rv_config")}
    check(f"k{k} lake configs", cfgs == {"get_col_cumsum_pond", "dual_read", "fifo"}
          and meta(inst, "lane0_col_sum_pond", "lake_rv_config") ==
          {"type": "get_col_cumsum_pond", "num_groups": 8, "num_rows": 256, "num_mats": 4}, str(cfgs))
    sa = meta(inst, a_ios[0], "glb2out_0")
    wa = walk(sa, "read")
    exp_a = [1024 * c + 4 * i + s for s in range(4) for i in range(256) for m in range(8)]
    check(f"k{k} a walk == word 1024 c + 4 i + s, held for the 8 groups", wa == exp_a and
          all(meta(inst, n, "glb2out_0") == sa for n in a_ios) and
          (sa.get("tb_write_starting_addr") == [0] if c else "tb_write_starting_addr" not in sa),
          f"start {sa['read_data_starting_addr']}, tb_write {sa.get('tb_write_starting_addr')}")
    mio = [n for n in inst if n.startswith("io16in_mask_")]
    wm = [walk(meta(inst, n, "glb2out_0"), "read") for n in mio]
    check(f"k{k} mask walks == word 8 i + m per matrix (32 lane IOs)", len(mio) == 32 and
          all(w == [8 * i + m for s in range(4) for i in range(256) for m in range(8)] for w in wm))
    oio = [n for n in inst if n.startswith("io16_hw_output")]
    check(f"k{k} output walks linear 0 .. 8191 (16 lane IOs)", len(oio) == 16 and
          all(walk(meta(inst, n, "in2glb_0"), "write") == list(range(8192)) for n in oio))

# 4. dataflow simulation ----------------------------------------------------------------------------------------------
print("== 4. dataflow simulation of the generated designs", flush=True)


def simulate(k, raws):
    inst, conns, env = designs[k]
    drv = drivers(conns)
    tiles = {"a": np.asarray(raws["a"], dtype=np.float64).reshape(-1, 8),
             "mask_strict": np.asarray(raws["mask_strict"], dtype=np.float64).reshape(-1, 16),
             "mask_lower": np.asarray(raws["mask_lower"], dtype=np.float64).reshape(-1, 16)}
    memo = {}

    def stream(node, port):
        if (node, port) in memo:
            return memo[(node, port)]
        v = inst[node]
        ref = v.get("modref") or v.get("genref")
        if ref == "global.IO":
            name = node[len("io16in_"):].split("_host_stencil")[0]
            out = tiles[name][walk(meta(inst, node, "glb2out_0"), "read"), lane_of(node)]
        elif ref == "coreir.reg":
            out = stream(*drv[(node, "in")])
        elif ref == "cgralib.Mem" and v["metadata"].get("is_rom"):
            out = f32(np.array(v["metadata"]["init"])[stream(*drv[(node, "addr_in_0")]).astype(np.int64)])
        elif ref == "cgralib.Mem":    # dual_read buffer or FIFO: the written stream, in order, on every read port
            out = stream(*drv[(node, "data_in_0")])
        elif ref == "cgralib.Pond":   # get_col_cumsum_pond: port 1 = the new sums, the recurrence closed through the PE
            cfg = meta(inst, node, "lake_rv_config")
            sum_pe = drv[(node, "data_in_pond_0")][0]
            assert drv[(sum_pe, "data0")] == (node, "data_out_pond_0")
            x = stream(*drv[(sum_pe, "data1")])
            out = scan(x.reshape(cfg["num_mats"], cfg["num_rows"], cfg["num_groups"])).reshape(-1)
        elif ref == "global.PE":
            role = node.split("_", 1)[1].rsplit("_pe", 1)[0].replace("_bias", "")   # lane<i>_mulb_bias_pe -> mulb
            d0 = lambda: stream(*drv[(node, "data0")])
            d1 = lambda: stream(*drv[(node, "data1")])
            if role in ("mul8", "mul9"):
                out = pe_mul(d0(), d1())
            elif role == "mulb":
                out = pe_mul(f32(INV_LN2_BITS), d1())
            elif role == "clamp":
                out = np.maximum(d0(), CLAMP)
            elif role == "getffrac":
                from decay_chain_lib import lassen_ops
                out = lassen_ops()[0][bits(d0())].astype(np.float64)
            elif role == "getfint":
                from decay_chain_lib import lassen_ops
                out = lassen_ops()[1][bits(d0())].astype(np.float64)
            elif role == "addiexp":
                from decay_chain_lib import lassen_ops
                addi = lassen_ops()[2]
                rom = {int(b): idx for idx, b in enumerate(exp_rom())}
                romv = bits(d0())
                out = f32(addi[[rom[int(b)] for b in romv], d1().astype(np.int64) & 0xFF])
            elif role == "sum":
                raise AssertionError("the sum PE is evaluated through its Pond")
            else:
                raise ValueError(role)
        else:
            raise ValueError(ref)
        memo[(node, port)] = np.asarray(out)
        return memo[(node, port)]

    gl = np.zeros((8192, 16))
    for n in inst:
        if n.startswith("io16_hw_output"):
            gl[walk(meta(inst, n, "in2glb_0"), "write"), lane_of(n)] = stream(*drv[(n, "in")])
    return gl.reshape(-1)


for k in RUNS:
    raws, gold = tensors("decay_chain", k)
    sim = simulate(k, raws)
    model = run_model(raws, k, lut=lut).reshape(-1)
    check(f"k{k} simulated design == host model (100%)", np.array_equal(sim, model),
          f"{100 * np.mean(sim == model):.4f}%")

# 5. parser -----------------------------------------------------------------------------------------------------------
print("== 5. parser", flush=True)
import parse_dnnLayer_tensors as P  # noqa: E402

for k in RUNS:
    d = os.path.join(WORK, f"parser_k{k}")
    os.makedirs(d, exist_ok=True)

    def parse():
        zw = P.parse_zircon_workarounds()
        P.parse_extra_inputs_bf16_cgra(BASE, d, zw)
        return P.parse_output("mamba2_prefill", "mul_9", BASE,
                              {"node": "mul_9", "shape": [1, 4, 1, 64, 256, 256], "dtype": "bfloat16"}, zw)

    out = with_env(KEY.format(k=k), parse)
    g = np.array(out, dtype=np.int64).reshape(-1) & 0xFFFF      # parse_output returns the gold's bf16 codes
    raws, gold = tensors("decay_chain", k)
    raw = lambda name: np.fromfile(os.path.join(d, f"hw_{name}_stencil.raw"), dtype=">u2").astype(np.int64)
    for name, t in raws.items():
        check(f"k{k} parser hw_{name}_stencil.raw == numpy slice {tuple(t.shape)}", np.array_equal(raw(name), bits(t).reshape(-1)))
    c, t, jl, q = run_digits(k)
    heads = [16 * s + 8 * t + jl for s in range(4)]
    indep = mul9[c][heads].reshape(4, 256, 8, 2, 16)[:, :, :, q, :].reshape(-1)
    check(f"k{k} parser gold == mul_9[c {c}, heads {heads}, half {q}]", np.array_equal(g, bits(indep))
          and np.array_equal(bits(gold).reshape(-1), bits(indep)))

# 6. accuracy ---------------------------------------------------------------------------------------------------------
print("== 6. accuracy vs voyager (whole layer 0)", flush=True)
x = pe_mul(a[..., None].astype(np.float64), strict[None, None].astype(np.float64)).reshape(256, 256, 256)
s_hw = scan(x).reshape(4, 64, 256, 256)
del x
out = pe_mul(f32(lut[bits(s_hw)]), lower[None, None].astype(np.float64))
gv = mul9.astype(np.float64)
check("host model rows above the diagonal == 0 (as voyager)", np.all(out[..., ii < jj] == 0) and np.all(gv[..., ii < jj] == 0))
low = np.broadcast_to(ii >= jj, out.shape)
o, g, s0 = out[low], gv[low], s_hw[low]
nz = g != 0
rel = np.zeros_like(g)
rel[nz] = np.abs(o[nz] - g[nz]) / np.abs(g[nz])
print(f"      lower-triangle values {o.size}: bit-exact {100 * np.mean(o == g):.3f}%, mean rel {100 * rel.mean():.3f}%, "
      f"p99 {100 * np.percentile(rel, 99):.3f}%, max {100 * rel.max():.3f}%", flush=True)
for lo, hi in ((0, -1), (-1, -5), (-5, -20), (-20, -50), (-50, -87.3), (-87.3, -1e9)):
    sel = (s0 <= lo) & (s0 > hi)
    if sel.any():
        exc = np.abs(o[sel] - g[sel]) > ATOL + RTOL * np.abs(g[sel])
        print(f"      S in ({hi:g}, {lo:g}]: {int(sel.sum()):>9} values, bit-exact {100 * np.mean(o[sel] == g[sel]):7.3f}%, "
              f"mean rel {100 * rel[sel].mean():7.3f}%, max {100 * rel[sel].max():8.3f}%, beyond test.py tol {int(exc.sum())}",
              flush=True)
cmp_s = s_hw - cs1.astype(np.float64)
print(f"      scan vs voyager cumsum_1: bit-exact {100 * np.mean(cmp_s[low.reshape(cmp_s.shape)] == 0):.3f}%", flush=True)
# The error split: the hardware exp chain on voyager's own cumsum_1, and voyager's exp (bf16(exp)) on the hardware scan
g_lo = gv[low]
for label, val in (("HW exp chain on voyager's cumsum_1", f32(lut[bits(cs1.astype(np.float64))])[low]),
                   ("exact exp (bf16) on the HW scan", bf_rne(np.exp(s_hw))[low])):
    r = np.zeros_like(g_lo)
    r[g_lo != 0] = np.abs(val[g_lo != 0] - g_lo[g_lo != 0]) / np.abs(g_lo[g_lo != 0])
    exc = np.abs(val - g_lo) > ATOL + RTOL * np.abs(g_lo)
    print(f"      {label}: bit-exact {100 * np.mean(val == g_lo):.3f}%, mean rel {100 * r.mean():.3f}%, "
          f"p99 {100 * np.percentile(r, 99):.3f}%, beyond test.py tol {int(exc.sum())}", flush=True)
    for lo, hi in ((0, -1), (-1, -5), (-5, -20), (-20, -50)):
        sel = (cs1[low] <= lo) & (cs1[low] > hi)
        print(f"        S in ({hi:g}, {lo:g}]: mean rel {100 * r[sel].mean():7.3f}%, max {100 * r[sel].max():8.3f}%",
              flush=True)
exc_all = np.abs(out - gv) > ATOL + RTOL * np.abs(gv)          # [c, h, i, j]
per_run = exc_all.reshape(4, 4, 2, 8, 256, 8, 2, 16).transpose(0, 2, 3, 6, 1, 4, 5, 7).reshape(128, -1).mean(axis=1)
verdict = lambda f: "PASS" if f == 0 else ("mostly passed" if f <= FRAC else "FAIL")
for k in RUNS:
    print(f"      predicted test.py run {k}: {verdict(per_run[k])} ({100 * per_run[k]:.3f}% beyond tolerance)", flush=True)
print(f"      all 128 runs: {Counter(verdict(f) for f in per_run)}; worst {100 * per_run.max():.3f}% (run {per_run.argmax()})",
      flush=True)
check("predicted test.py exit 0 on runs 0 and 127 (<= 6% exceptions)", all(per_run[k] <= FRAC for k in RUNS))

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + '; '.join(fails)}", flush=True)
sys.exit(1 if fails else 0)
