"""Host checks of the Mamba2 X = x * dt bring-up (mamba_x_dt_fp, strait elementwise_mul_head_gather_bf16), before RTL.

1. Template: emit the design; IO DMA patterns, MEM configs and instance counts.
2. Dataflow: simulate the generated design_top.json (its actual wiring; every get_interleave_mem with the address
   patterns lake's own config generates) on real data for runs 0 and 63, and rebuild the output GLB tensor from the
   output DMA walk. Must == the gold.
3. Parser: the aha keys' env through parse_extra_inputs_bf16_cgra / parse_output == independent numpy slices of
   reshape, softplus and permute_1; the skip-gate keys' extra inputs unchanged.
4. Accuracy: bf16 RNE x * dt == voyager's mul_7 / permute_1; and the PE fp_mul rounding's agreement.
Usage: check_x_dt_host.py <workdir>   (run with /aha/bin/python)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
WORK = sys.argv[1]
os.makedirs(WORK, exist_ok=True)
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
PARAMS = json.load(open("/aha/aha/util/application_parameters.json"))
U, LAYOUT, H, HEAD_DIM, R = 16, 32, 8, 64, 256
B = HEAD_DIM // LAYOUT
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


def bits(a):
    return (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64)


def f32(b):
    return (np.asarray(b, dtype=np.int64).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def bf_rne(a):
    m, e = np.frexp(np.asarray(a, dtype=np.float64))
    return np.ldexp(np.rint(m * 256.0) / 256.0, e)


def pe_mul(a, b):  # Zircon PE fp_mul: RNE from the product truncated to 3 bits below the bf16 LSB (no sticky)
    m, e = np.frexp(np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64))
    q = np.trunc(np.abs(m) * 256.0 * 8.0) / 8.0
    r = np.floor(q)
    f = q - r
    return np.ldexp(np.sign(m) * (r + ((f > 0.5) | ((f == 0.5) & (r % 2 == 1)))) / 256.0, e)


def walk(stream, key):
    """Word addresses of a relative-stride GLB stream dict (SKIP_GLB_DMA_STRIDE_ADJUSTMENT semantics)."""
    ext, rel = stream["extent"], stream[f"{key}_data_stride"]
    a, idx, out = stream[f"{key}_data_starting_addr"][0], [0] * len(ext), []
    total = int(np.prod(ext))
    for n in range(total):
        out.append(a)
        if n == total - 1:
            break
        k = 0
        while idx[k] == ext[k] - 1:
            idx[k] = 0
            k += 1
        idx[k] += 1
        a += rel[k]
    return out


def lake_walk(cfg):
    """Absolute addresses of a lake port config (addr = offset + sum idx_d stride_d, innermost first)."""
    ext, st, off = cfg["extents"], cfg["address"]["strides"], cfg["address"]["offset"]
    grids = np.meshgrid(*[np.arange(e) for e in ext], indexing="ij")
    addr = off + sum(g * s for g, s in zip(grids, st))
    return addr.transpose(*reversed(range(len(ext)))).reshape(-1)   # innermost dim varies fastest


def run_digits(k):
    return k // 16, (k // 2) % 8, k % 2   # c, hb, q (CGRA_RUN_DIGITS c:4,hb:8,q:2)


def expected(k):
    c, hb, q = run_digits(k)
    x = np.fromfile(BASE + "reshape.bin", dtype=np.float32).reshape(1024, 4096)
    xh = x[256 * c:256 * c + 256, 512 * hb:512 * hb + 512].reshape(256, 16, 2, 16)[:, :, q, :].reshape(256, 256)
    dt = np.fromfile(BASE + "softplus.bin", dtype=np.float32).reshape(1024, 64)[256 * c:256 * c + 256, 8 * hb:8 * hb + 8]
    p1 = np.fromfile(BASE + "permute_1.bin", dtype=np.float32).reshape(4, 64, 256, 64)
    gold = p1[c, 8 * hb:8 * hb + 8].reshape(8, 256, 2, 2, 16)[:, :, :, q, :].reshape(2048, 32)
    return xh, dt, gold


# ---------------------------------------------------------------- 1. template
from strait.coreir_backend.templates.elementwise_mul_head_gather_bf16 import emit_elementwise_mul_head_gather_bf16_design
from lake.spec.hack_rv_mem_pond_bitstream import get_interleave_mem

emit_elementwise_mul_head_gather_bf16_design(U, H, R, WORK, layout_lanes=LAYOUT, head_dim=HEAD_DIM)
design = json.load(open(os.path.join(WORK, "design_top.json")))
mod = design["namespaces"]["global"]["modules"]["mamba_x_dt_fp"]
inst, conns = mod["instances"], mod["connections"]


def meta(name, key):
    v = inst[name].get("metadata", {}).get(key)
    return json.loads(v) if isinstance(v, str) else v


ios = {n: i for n, i in inst.items() if i.get("modref") == "global.IO"}
lane = lambda n: int(n.split("_clkwrk_")[1].split("_")[0])
in_io = sorted((n for n in ios if n.startswith("io16in_input_")), key=lane)
dt_io = sorted((n for n in ios if n.startswith("io16in_dt_")), key=lane)
out_io = sorted((n for n in ios if n.startswith("io16_hw_output")), key=lane)
check("IO counts 16 input / 8 dt / 16 output", (len(in_io), len(dt_io), len(out_io)) == (16, 8, 16))
pat = lambda s, key: (s["extent"], s[f"{key}_data_stride"])
check("input DMA linear [16,256]/[1,1]", all(pat(meta(n, "glb2out_0"), "read") == ([16, 256], [1, 1]) for n in in_io))
check("dt DMA [2,256]/[0,1]", all(pat(meta(n, "glb2out_0"), "read") == ([2, 256], [0, 1]) for n in dt_io))
check("output DMA [2,8,256]/[1,511,-3583]",
      all(pat(meta(n, "in2glb_0"), "write") == ([2, 8, 256], [1, 511, -3583]) for n in out_io))
mems = {n: meta(n, "lake_rv_config") for n, i in inst.items() if i.get("genref") == "cgralib.Mem"}
kinds = sorted((c["type"], c.get("single_input_stream_size", c.get("input_stream_size")), c.get("row_size") or c.get("turn")) for c in mems.values())
from collections import Counter
cnt = Counter(kinds)
print("  MEM configs:", dict(cnt))
check("MEM configs: interleave (turn 2) 4x512 + 2x1024 + 1x2048, 8 dual_read 4096, 16 fifo 4096",
      cnt == Counter({("get_interleave_mem", 512, 2): 4, ("get_interleave_mem", 1024, 2): 2,
                      ("get_interleave_mem", 2048, 2): 1, ("dual_read", 4096, 4): 8, ("fifo", 4096, 8): 16}))
def read_port_rate(addrs, fw=4):
    """Values per cycle of a lake opt_rv read port (lake/spec/port.py output path, opt_timing=True as for
    build_four_port_wide_fetch_rv's out ports) on an address sequence: 2-entry word cache that cannot refill in the cycle
    it pops, 8-deep address queue; SRAM always granted, downstream always ready."""
    q, wcb, bus, already, last, i, cyc = [], [], None, False, None, 0, 0
    while i < len(addrs) or q:
        cyc += 1
        dbw = bus is not None and len(wcb) < 2
        head_ok = bool(q) and len(wcb) > 0 and q[0][0] in wcb
        pop_wcb = head_ok and q[0][1] and len(wcb) == 2
        read = push = False
        if i < len(addrs) and len(q) < 8:
            w = addrs[i] // fw
            if not already:
                read = push = True
            elif w != last:
                read = push = (bus is None or dbw)
            else:
                push = True
        if head_ok:
            q.pop(0)
            if pop_wcb:
                wcb.pop(0)
        if dbw:
            wcb.append(bus)
            bus = None
        if read:
            bus = last = addrs[i] // fw
            already = True
        if push:
            q.append((addrs[i] // fw, read))
            i += 1
    return len(addrs) / cyc


for size in (512, 1024, 2048):
    for turn in (1, 2):
        rate = read_port_rate(lake_walk(get_interleave_mem(size, turn=turn)[3]["config"]).tolist())
        print(f"  read-port model: get_interleave_mem {size} turn {turn}: {rate:.3f} values/cycle")
root_rate = read_port_rate(lake_walk(get_interleave_mem(2048, turn=2)[3]["config"]).tolist())
check("root MEM read port (turn 2) >= 0.95 values/cycle (the root gives one value per beat)", root_rate >= 0.95, f"{root_rate:.3f}")

pes = [n for n, i in inst.items() if i.get("modref") == "global.PE"]
check("PEs: 16 mul + 7 pass-through", len(pes) == 23 and sum(n.startswith("mul_pe_") for n in pes) == 16, f"{len(pes)} PEs")

# ---------------------------------------------------------------- 2. dataflow simulation of the generated design
SINK_PORTS = ("data_in_0", "data_in_1", "data0", "data1", "in")
drivers = {}
for a, b in conns:
    for src, dst in ((a, b), (b, a)):
        dn, _, dp = dst.partition(".")
        sn, _, sp = src.partition(".")
        if dn in inst and dp in SINK_PORTS and sn != "self" and not sn.startswith("const_inst") and sp in ("out", "O0", "data_out_0", "data_out_1"):
            drivers[(dn, dp)] = (sn, sp)


def interleave_read(size, turn, w0, w1):
    cfg = get_interleave_mem(size, turn=turn)
    sram = {}
    for port, vals in ((0, w0), (1, w1)):
        addrs = lake_walk(cfg[port]["config"])
        assert len(addrs) == len(vals) == size
        sram.update(zip(addrs.tolist(), vals))
    return [sram[a] for a in lake_walk(cfg[3]["config"]).tolist()]


def simulate(k):
    xh, dt, gold = expected(k)
    cache = {}

    def stream(node, port):   # the value stream on an instance's output port
        if (node, port) in cache:
            return cache[(node, port)]
        if node in ios:       # an input IO: its DMA walk over its lane's words
            words = walk(meta(node, "glb2out_0"), "read")
            if node.startswith("io16in_dt_"):
                vals = [dt[w, lane(node)] for w in words]
            else:
                vals = [xh[w // 16, (w % 16) * 16 + lane(node)] for w in words]
        elif node in mems:
            cfg = mems[node]
            if cfg["type"] == "get_interleave_mem":
                vals = interleave_read(cfg["single_input_stream_size"], cfg.get("turn", 1), stream(*drivers[(node, "data_in_0")]),
                                       stream(*drivers[(node, "data_in_1")]))
            else:             # dual_read / fifo: every read port carries the written stream
                vals = stream(*drivers[(node, "data_in_0")])
        elif node.startswith("mul_pe_"):
            a, b = np.array(stream(*drivers[(node, "data0")])), np.array(stream(*drivers[(node, "data1")]))
            vals = list(bf_rne(a.astype(np.float64) * b))
        else:                 # pass-through PE (or_ 0)
            vals = stream(*drivers[(node, "data0")])
        cache[(node, port)] = vals
        return vals

    out = np.full((2048, 32), np.nan)
    for n in out_io:
        vals = stream(*drivers[(n, "in")])
        words = walk(meta(n, "in2glb_0"), "write")
        assert len(vals) == len(words) == 4096
        for w, v in zip(words, vals):
            out[w // 2, (w % 2) * 16 + lane(n)] = v
    check(f"run {k}: simulated design (lake interleave patterns, DMA walks) == gold permute_1 block",
          np.array_equal(out, gold.astype(np.float64)),
          f"{100 * np.mean(out == gold):.3f}% equal, {np.isnan(out).sum()} unwritten")


for k in (0, 63):
    simulate(k)

# ---------------------------------------------------------------- 3. parser
import parse_dnnLayer_tensors as P


def with_env(key_name, fn):
    saved = dict(os.environ)
    os.environ.update(PARAMS[key_name]["default"])
    try:
        return fn()
    finally:
        os.environ.clear()
        os.environ.update(saved)


for k in (0, 63):
    xh, dt, gold = expected(k)
    d = os.path.join(WORK, f"parser_k{k}")
    os.makedirs(d, exist_ok=True)

    def parse():
        zw = P.parse_zircon_workarounds()
        P.parse_extra_inputs_bf16_cgra(BASE, d, zw)
        return P.parse_output("mamba2_prefill", "permute_1", BASE,
                              {"node": "permute_1", "shape": [1, 4, 1, 64, 256, 64], "dtype": "bfloat16"}, zw)

    g = with_env(f"mamba_x_dt_fp_mamba2_prefill_k{k}", parse)
    raw = lambda f: np.fromfile(os.path.join(d, f), dtype=">u2").astype(np.int64)
    check(f"run {k}: hw_input_stencil.raw == x half slab [256 x 256]", np.array_equal(raw("hw_input_stencil.raw"), bits(xh).reshape(-1)))
    check(f"run {k}: hw_dt_stencil.raw == dt [256 x 8]", np.array_equal(raw("hw_dt_stencil.raw"), bits(dt).reshape(-1)))
    check(f"run {k}: gold == permute_1 half block [2048 x 32]", np.array_equal(np.array(g, dtype=np.int64), bits(gold).reshape(-1)))

# skip-gate keys unchanged: k0's extra inputs == the raws its last RTL run (k0, 2026-11-22) left in the app dir
sg = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/mamba_skip_gate_fp"
d = os.path.join(WORK, "parser_skip_gate_k0")
os.makedirs(d, exist_ok=True)
with_env("mamba_skip_gate_fp_mamba2_prefill_k0", lambda: P.parse_extra_inputs_bf16_cgra(BASE, d, P.parse_zircon_workarounds()))
for f in ("hw_gate_stencil.raw", "hw_yheads_stencil.raw"):
    check(f"skip-gate k0 {f} unchanged", open(os.path.join(d, f), "rb").read() == open(os.path.join(sg, f), "rb").read())

# ---------------------------------------------------------------- 4. accuracy
x = np.fromfile(BASE + "reshape.bin", dtype=np.float32).reshape(1024, 64, 64).astype(np.float64)
dt = np.fromfile(BASE + "softplus.bin", dtype=np.float32).reshape(1024, 64, 1).astype(np.float64)
m7 = np.fromfile(BASE + "mul_7.bin", dtype=np.float32).reshape(1024, 64, 64).astype(np.float64)
p1 = np.fromfile(BASE + "permute_1.bin", dtype=np.float32).reshape(4, 64, 256, 64).astype(np.float64)
check("bf16 RNE x * dt == mul_7 (whole layer)", np.array_equal(bf_rne(x * dt), m7), f"{100 * np.mean(bf_rne(x * dt) == m7):.4f}%")
check("permute_1 == permute(mul_7) (c, l, h, p) -> (c, h, l, p)", np.array_equal(m7.reshape(4, 256, 64, 64).transpose(0, 2, 1, 3), p1))
print(f"  PE fp_mul model == mul_7: {100 * np.mean(pe_mul(x, dt) == m7):.3f}% (expected ~95%: 3 guard bits, no sticky)")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
