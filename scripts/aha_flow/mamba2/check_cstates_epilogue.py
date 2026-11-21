"""Host checks for the Mamba2 C@states x exp(a_cum) epilogue (zircon_2d_broadcast_mul_fp), before RTL. The chunk is the
app keys' EXP_EPILOGUE_CHUNK (GEMM matmul_mx_(264 + chunk)); chunk 0 is all zeros (zero entering state), so use chunk >= 1.

(a) Pattern: decode the MU-tiled output address pattern of the GEMM-only run's bin/design_meta.json into the MU
    output order (row, 32-column block) and check that the broadcast IO's pattern (strait template
    elementwise_mul_broadcast_bf16) reads scale address head(block) * rows + row at every handshake.
(b) Gold: parse_output's EXP_EPILOGUE_NODE gold, for all K tiles, must equal bf16(C@states * exp(a_cum)), and
    bf16(Y_diag (view_1) + that, head-major) must reproduce the graph's submodule_6 bit for bit, which shows the gold is
    the fused-away mul_15. The GEMM output must be nonzero (a zero chunk makes every check vacuous).
(c) Inputs: write_exp_epilogue_input for K tiles 0 and 1 must write exp_2 heads 16k .. 16k+15 of the chunk.
Prints PASS/FAIL per check and exits 1 on any failure. Run with /aha/bin/python (NumPy 1.x, like the flow).

Usage: python check_cstates_epilogue.py <GEMM-only run's design_meta.json>
"""
import json, os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
import parse_dnnLayer_tensors as pdt
from strait.coreir_backend.templates.elementwise_mul_broadcast_bf16 import broadcast_address_pattern

APP_PARAMS = "/aha/aha/util/application_parameters.json"
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
KEY = "zircon_2d_broadcast_mul_fp_post_mamba2_prefill_Cstates_exp_kernel{}"
PARAMS = json.load(open(APP_PARAMS))
CHUNK = int(PARAMS[KEY.format(0)]["default"]["EXP_EPILOGUE_CHUNK"])
GEMM = f"matmul_mx_{264 + CHUNK}"
GEMM_SPEC = {"node": GEMM, "shape": [256, 4096], "dtype": "bfloat16"}
ROWS, COLS, HEADS, P = 256, 4096, 64, 64
CLEAN_ENV = dict(os.environ)


def with_env(k):
    os.environ.clear()
    os.environ.update(CLEAN_ENV)
    os.environ.update(PARAMS[KEY.format(k)]["default"])
    return pdt.parse_zircon_workarounds()


def addresses(start, extent, stride):
    """Address sequence of a GLB address generator: relative strides, innermost loop first (odometer)."""
    idx, addr, out = [0] * len(extent), start, []
    for _ in range(int(np.prod(extent))):
        out.append(addr)
        for lvl in range(len(extent)):
            idx[lvl] += 1
            if idx[lvl] < extent[lvl]:
                addr += stride[lvl]
                break
            idx[lvl] = 0
    return out


def bits(t):
    return (t.to(torch.float32).numpy().view(np.uint32) >> 16).astype(np.uint16)


ok_all = True
gen = dict(kv.split("=", 1) for kv in PARAMS[KEY.format(0)]["default"]["HALIDE_GEN_ARGS"].split())
x1, x2, k1, k2, group_cols = (int(gen[k]) for k in ("x1", "x2", "k1", "k2", "group_cols"))
run_cols = COLS // int(PARAMS[KEY.format(0)]["default"]["NUM_K_HOST_TILING_KERNELS"])

# (a) MU output order from the GEMM-only run's output IO tile, then the broadcast pattern against it
meta = json.load(open(sys.argv[1]))
tile = meta["IOs"]["outputs"][0]["io_tiles"][0]
addr = tile["addr"]
out_seq = addresses(addr["write_data_starting_addr"][0], addr["extent"], addr["write_data_stride"])
words_per_row = run_cols // 32
order = [(a // words_per_row, a % words_per_row) for a in out_seq]     # (row, 32-column block) per handshake
covers = len(order) == x1 * x2 * k1 * k2 and sorted(order) == [(r, b) for r in range(x1 * x2) for b in range(words_per_row)]
b_extent, b_stride, num_scales = broadcast_address_pattern(x1, x2, k1, k2, group_cols)
b_seq = addresses(0, b_extent, b_stride)
expected = [(blk * 32 // group_cols) * (x1 * x2) + row for row, blk in order]
ok = covers and b_seq == expected and max(b_seq) == num_scales - 1 and min(b_seq) == 0
print(f"[CHECK a] output tile pattern extent {addr['extent']} stride {addr['write_data_stride']} -> MU order covers "
      f"all {len(order)} (row, block) words once: {covers}; broadcast extent {b_extent} stride {b_stride} reads "
      f"head(block)*{x1 * x2}+row at every handshake: {b_seq == expected} -> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

# (b) the gold hook for all K tiles vs bf16(C@states * exp), and against the graph's submodule_6
y_off = torch.from_numpy(np.fromfile(BASE + GEMM + ".bin", dtype=np.float32).reshape(ROWS, HEADS, P))
nonzero = int((y_off != 0).sum())
exp = torch.from_numpy(np.fromfile(BASE + "exp_2.bin", dtype=np.float32).reshape(4, HEADS, ROWS))[CHUNK]
host = (y_off.double() * exp.transpose(0, 1).double().unsqueeze(-1)).to(torch.bfloat16)      # [l, h, p]
tiles = []
for k in range(COLS // run_cols):
    zw = with_env(0)
    os.environ["K_DIM_HOST_TILING_IDX"] = str(k)
    zw = pdt.parse_zircon_workarounds()
    tiles.append(np.asarray(pdt.parse_output("mamba2_prefill", GEMM, BASE, GEMM_SPEC, zw), dtype=np.uint16).reshape(ROWS, run_cols))
hook = np.concatenate(tiles, axis=1)
hook_ok = np.array_equal(hook, bits(host).reshape(ROWS, COLS))
r15 = np.fromfile(BASE + "reshape_15.bin", dtype=np.float32).reshape(4, ROWS, HEADS, P)[CHUNK]
layout_ok = np.array_equal(r15, y_off.numpy())
view_1 = torch.from_numpy(np.fromfile(BASE + "view_1.bin", dtype=np.float32).reshape(4, HEADS, ROWS, P)[CHUNK])
sub6 = np.fromfile(BASE + "submodule_6.bin", dtype=np.float32).reshape(4, HEADS, ROWS, P)[CHUNK]
recombined = (view_1.to(torch.bfloat16).float() + host.float().permute(1, 0, 2)).to(torch.bfloat16).float().numpy()
graph_ok = np.array_equal(recombined, sub6)
nonzero_ok = nonzero > 0.99 * y_off.numel()
ok = hook_ok and layout_ok and graph_ok and nonzero_ok
print(f"[CHECK b] chunk {CHUNK}: {GEMM} nonzero {nonzero}/{y_off.numel()}; gold hook (4 K tiles) == bf16(C@states * exp): {hook_ok}; reshape_15 == {GEMM}: {layout_ok}; "
      f"bf16(view_1 + gold) == submodule_6 bit for bit: {graph_ok} (mismatches {np.sum(recombined != sub6)}) -> "
      f"{'PASS' if ok else 'FAIL'}")
ok_all &= ok

# (c) the broadcast input for K tiles 0 and 1
ok = True
for k in (0, 1):
    zw = with_env(k)
    with tempfile.TemporaryDirectory() as tmp:
        pdt.write_exp_epilogue_input(BASE, tmp, zw, GEMM_SPEC["shape"])
        raw = np.fromfile(f"{tmp}/hw_broadcast_input_stencil.raw", dtype=">u2").astype(np.uint16)
    per = HEADS // (COLS // run_cols)
    want = bits(exp[k * per:(k + 1) * per].contiguous()).reshape(-1)
    this_ok = raw.size == num_scales and np.array_equal(raw, want)
    print(f"[CHECK c] K tile {k}: broadcast input == exp_2 chunk {CHUNK} heads {k * per}..{(k + 1) * per - 1} "
          f"({raw.size} values): {this_ok}")
    ok &= this_ok
print(f"[CHECK c] -> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

print("[RESULT]", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
