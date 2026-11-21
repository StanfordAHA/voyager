"""Host checks for the Mamba2 B^T@X_end + decayed entering state epilogue (zircon_2d_psum_reduction_fp), chunk 1.

The GEMM is matmul_mx_261 (B^T [128 x 256] @ X_end [256 x 4096]); the epilogue adds mul_13 (entering state x chunk decay),
which model.txt keeps as the separate add_7 behind nop stack/view/reshape/select ops (same element order).
(a) Graph: chunk 0's state mul_12 is all zeros (add_6 == matmul_mx_260), and bf16(matmul_mx_261 + mul_13) == add_7 bit
    for bit. Also guards against a vacuous test: the GEMM output and the residual of every K tile must be nonzero.
(b) Gold: parse_output's ADD_EPILOGUE_NODE gold, for all K tiles, must equal bf16(GEMM + mul_13).
(c) Inputs: write_add_epilogue_input for every K tile must write mul_13's columns of that tile (app dir and bin/).
(d) Visibility: how many outputs the add changes per K tile (test.py's tolerance cannot see a missing add here).
Prints PASS/FAIL per check and exits 1 on any failure. Run with /aha/bin/python (NumPy 1.x, like the flow).

Usage: python check_btxend_epilogue.py
       python check_btxend_epilogue.py --exact <GEMM-only hw_output.txt> <fused hw_output.txt> [K tile, default 1]
         After RTL: the fused output must equal bf16(GEMM-only RTL output + residual) bit for bit. Both runs see the
         same MU output, so this proves the add and its residual slice.
"""
import json, os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
import parse_dnnLayer_tensors as pdt

APP_PARAMS = "/aha/aha/util/application_parameters.json"
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
KEY = "zircon_2d_psum_reduction_fp_post_mamba2_prefill_BtXend_state_kernel{}"
GEMM_SPEC = {"node": "matmul_mx_261", "shape": [128, 4096], "dtype": "bfloat16"}
ROWS, COLS = 128, 4096
PARAMS = json.load(open(APP_PARAMS))
CLEAN_ENV = dict(os.environ)
NUM_K = int(PARAMS[KEY.format(0)]["default"]["NUM_K_HOST_TILING_KERNELS"])
RUN_COLS = COLS // NUM_K


def with_env(k):
    os.environ.clear()
    os.environ.update(CLEAN_ENV)
    os.environ.update(PARAMS[KEY.format(k)]["default"])
    return pdt.parse_zircon_workarounds()


def load(name):
    return torch.from_numpy(np.fromfile(BASE + name + ".bin", dtype=np.float32).reshape(ROWS, COLS))


def bits(t):
    return (t.to(torch.float32).numpy().view(np.uint32) >> 16).astype(np.uint16)


def from_bits(b):
    return torch.from_numpy((b.astype(np.uint32) << 16).view(np.float32))


def read_hw_output(path):
    return np.array([int(h, 16) for h in open(path).read().split()], dtype=np.uint16)


def tile(t, k):
    return t[:, k * RUN_COLS:(k + 1) * RUN_COLS]


gemm, res = load("matmul_mx_261"), load("mul_13")
host = (gemm.double() + res.double()).to(torch.bfloat16)

if len(sys.argv) >= 4 and sys.argv[1] == "--exact":
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    nop, fused = read_hw_output(sys.argv[2]), read_hw_output(sys.argv[3])
    ok = nop.size == fused.size == ROWS * RUN_COLS
    expect = bits((from_bits(nop).double().reshape(ROWS, RUN_COLS) + tile(res, k).double()).to(torch.bfloat16)).reshape(-1) if ok else None
    if ok:
        exact = expect == fused
        changed = expect != nop          # outputs the add changes, given the RTL GEMM output
        ok = bool(exact.all())
        print(f"[EXACT] K tile {k}: fused == bf16(GEMM-only RTL + mul_13 tile) on {exact.sum()}/{exact.size} values "
              f"({100 * exact.mean():.4f}%); on the {changed.sum()} values ({100 * changed.mean():.2f}%) the add changes: "
              f"{exact[changed].sum()} exact, {(fused[changed] == nop[changed]).sum()} unchanged (add missing) -> "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        print(f"[EXACT] size mismatch: GEMM-only {nop.size}, fused {fused.size}, expected {ROWS * RUN_COLS} -> FAIL")
    sys.exit(0 if ok else 1)

ok_all = True

# (a) graph identities, and nonzero data (a zero GEMM output or residual would make every RTL check vacuous)
mul_12 = np.fromfile(BASE + "mul_12.bin", dtype=np.float32)
add_6 = np.fromfile(BASE + "add_6.bin", dtype=np.float32)
mm_260 = np.fromfile(BASE + "matmul_mx_260.bin", dtype=np.float32)
add_7 = load("add_7")
chunk0_ok = not mul_12.any() and np.array_equal(add_6, mm_260)
graph_ok = np.array_equal(bits(host), bits(add_7))
nz = [(int((tile(gemm, k) != 0).sum()), int((tile(res, k) != 0).sum())) for k in range(NUM_K)]
nonzero_ok = all(g > 0.99 * ROWS * RUN_COLS and r > 0 for g, r in nz)
ok = chunk0_ok and graph_ok and nonzero_ok
print(f"[CHECK a] chunk 0: mul_12 all zero and add_6 == matmul_mx_260: {chunk0_ok}; bf16(matmul_mx_261 + mul_13) == add_7 "
      f"bit for bit: {graph_ok} (mismatches {np.sum(bits(host) != bits(add_7))}); nonzero (GEMM, residual) per K tile "
      f"{nz} -> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

# (b) the gold hook for all K tiles
tiles = []
for k in range(NUM_K):
    zw = with_env(k)
    tiles.append(np.asarray(pdt.parse_output("mamba2_prefill", "matmul_mx_261", BASE, GEMM_SPEC, zw), dtype=np.uint16).reshape(ROWS, RUN_COLS))
hook_ok = np.array_equal(np.concatenate(tiles, axis=1), bits(host))
print(f"[CHECK b] gold hook ({NUM_K} K tiles) == bf16(matmul_mx_261 + mul_13): {hook_ok} -> {'PASS' if hook_ok else 'FAIL'}")
ok_all &= hook_ok

# (c) the partial-sum input for every K tile, in the app dir and in bin/
ok = True
for k in range(NUM_K):
    zw = with_env(k)
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(f"{tmp}/bin")
        pdt.write_add_epilogue_input(BASE, tmp, zw, GEMM_SPEC["shape"])
        raws = [np.fromfile(f"{tmp}/{d}hw_partial_sum_input_stencil.raw", dtype=">u2").astype(np.uint16) for d in ("", "bin/")]
    want = bits(tile(res, k).contiguous()).reshape(-1)
    this_ok = all(r.size == ROWS * RUN_COLS and np.array_equal(r, want) for r in raws)
    print(f"[CHECK c] K tile {k}: hw_partial_sum_input_stencil.raw (app dir and bin/) == mul_13 columns "
          f"{k * RUN_COLS}..{(k + 1) * RUN_COLS - 1} ({raws[0].size} values): {this_ok}")
    ok &= this_ok
print(f"[CHECK c] -> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

# (d) visibility of the add (informational)
for k in range(NUM_K):
    changed = bits(tile(host, k)) != bits(tile(gemm, k))
    print(f"[INFO d] K tile {k}: the add changes {100 * changed.mean():.2f}% of the bf16 outputs; max |mul_13| "
          f"{float(tile(res, k).abs().max()):.4g} vs max |GEMM| {float(tile(gemm, k).abs().max()):.4g}")

print("[RESULT]", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
