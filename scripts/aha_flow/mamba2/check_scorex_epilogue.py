"""Host checks for the Mamba2 score@x + Y_off * exp(a_cum) epilogue (zircon_2d_psum_reduction_fp), before RTL.

score@x of chunk c, head h is matmul_mx_(4 + 64c + h) (W [256 x 256] @ X [256 x 64] -> Y_diag [256 x 64]); the epilogue
adds head h's [256 x 64] block of the C@states epilogue output (Y_off * exp(a_cum), chunk c), which removes submodule_6.
The chunk and head are the app key's YOFF_EPILOGUE_CHUNK / YOFF_EPILOGUE_HEAD.
(a) Graph: view_1[c, h] == the score@x GEMM output, and bf16(view_1[c] + host block, all heads) == submodule_6[c] bit
    for bit (so the epilogue is exactly the fused-away add_9). The GEMM output and the block must be nonzero.
(b) Gold: parse_output's gold == bf16(score@x + HOST block).
(c) Inputs: write_add_epilogue_input writes the host block without YOFF_EPILOGUE_LIVE_TXT, and the live block (head h's
    columns of the C@states RTL output) with it, in the app dir and bin/.
(d) Live vs host block: bit-exact share and the share out of test.py-style tolerance (atol 1.5e-4, rtol 0.2).
(c-live) and (d) need the live C@states output; they are skipped if it is missing, unless --require-live is given.
Prints PASS/FAIL per check and exits 1 on any failure. Run with /aha/bin/python (NumPy 1.x, like the flow).

Usage: python check_scorex_epilogue.py [--require-live]
"""
import json, os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
import parse_dnnLayer_tensors as pdt

APP_PARAMS = "/aha/aha/util/application_parameters.json"
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
KEY = "zircon_2d_psum_reduction_fp_post_mamba2_prefill_scorex_yoff_kernel0"
ROWS, HEADS, P, CHUNKS = pdt.MAMBA2_ROWS, pdt.MAMBA2_HEADS, pdt.MAMBA2_P, pdt.MAMBA2_CHUNKS
ENV = json.load(open(APP_PARAMS))[KEY]["default"]
CHUNK, HEAD = int(ENV["YOFF_EPILOGUE_CHUNK"]), int(ENV["YOFF_EPILOGUE_HEAD"])
GEMM = f"matmul_mx_{4 + HEADS * CHUNK + HEAD}"
GEMM_SPEC = {"node": GEMM, "shape": [ROWS, P], "dtype": "bfloat16"}
LIVE = ENV.get("YOFF_EPILOGUE_LIVE_TXT", "")
REQUIRE_LIVE = "--require-live" in sys.argv
CLEAN_ENV = dict(os.environ)


def with_env(live=True):
    os.environ.clear()
    os.environ.update(CLEAN_ENV)
    os.environ.update(ENV)
    if not live:
        os.environ.pop("YOFF_EPILOGUE_LIVE_TXT", None)
    return pdt.parse_zircon_workarounds()


def load(name, shape):
    return torch.from_numpy(np.fromfile(BASE + name + ".bin", dtype=np.float32).reshape(shape))


def bits(t):
    return (t.to(torch.float32).contiguous().numpy().view(np.uint32) >> 16).astype(np.uint16)


def from_bits(b):
    return torch.from_numpy((b.astype(np.uint32) << 16).view(np.float32))


def written_input(zw):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(f"{tmp}/bin")
        pdt.write_add_epilogue_input(BASE, tmp, zw, GEMM_SPEC["shape"])
        return [np.fromfile(f"{tmp}/{d}hw_partial_sum_input_stencil.raw", dtype=">u2").astype(np.uint16) for d in ("", "bin/")]


ok_all = True
print(f"[INFO] chunk {CHUNK}, head {HEAD}: score@x = {GEMM}; live C@states output: {LIVE or '(none)'}")
with_env(live=False)
host = pdt.yoff_epilogue_host_block(BASE)                                     # [256, 64] float32 (bf16 values)
gemm = load(GEMM, (ROWS, P))
gold_host = (gemm.double() + host.double()).to(torch.bfloat16)

# (a) graph identities over the whole chunk, and nonzero data
y_off = load(f"matmul_mx_{264 + CHUNK}", (ROWS, HEADS, P))
exp = load("exp_2", (CHUNKS, HEADS, ROWS))[CHUNK]
blocks = (y_off.double() * exp.transpose(0, 1).double().unsqueeze(-1)).to(torch.bfloat16)   # [l, h, p]
view_1 = load("view_1", (CHUNKS, HEADS, ROWS, P))[CHUNK]
sub6 = load("submodule_6", (CHUNKS, HEADS, ROWS, P))[CHUNK]
recombined = (view_1.double() + blocks.permute(1, 0, 2).double()).to(torch.bfloat16)
map_ok = torch.equal(view_1[HEAD], gemm)
graph_ok = np.array_equal(bits(recombined), bits(sub6))
block_ok = np.array_equal(bits(host), bits(blocks[:, HEAD, :]))
nonzero = (int((gemm != 0).sum()), int((host != 0).sum()))
nonzero_ok = all(n > 0.99 * ROWS * P for n in nonzero)
ok = map_ok and graph_ok and block_ok and nonzero_ok
print(f"[CHECK a] view_1[{CHUNK},{HEAD}] == {GEMM}: {map_ok}; bf16(view_1 + Y_off*exp blocks) == submodule_6[{CHUNK}] bit for "
      f"bit: {graph_ok} (mismatches {np.sum(bits(recombined) != bits(sub6))}); helper block == head {HEAD}'s block: {block_ok}; "
      f"nonzero (GEMM, block) {nonzero} of {ROWS * P} -> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

# (b) the gold hook
zw = with_env()
hook = np.asarray(pdt.parse_output("mamba2_prefill", GEMM, BASE, GEMM_SPEC, zw), dtype=np.uint16).reshape(ROWS, P)
ok = np.array_equal(hook, bits(gold_host))
changed = bits(gold_host) != bits(gemm)
print(f"[CHECK b] gold hook == bf16({GEMM} + host block): {ok} (the add changes {100 * changed.mean():.1f}% of the outputs) "
      f"-> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

# (c) the partial-sum input: host block without the live file
raws = written_input(with_env(live=False))
ok = all(r.size == ROWS * P and np.array_equal(r, bits(host).reshape(-1)) for r in raws)
print(f"[CHECK c] without YOFF_EPILOGUE_LIVE_TXT: hw_partial_sum_input_stencil.raw (app dir and bin/) == host block: {ok} "
      f"-> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

# (c-live), (d): the live block from the C@states RTL output
if LIVE and os.path.exists(LIVE):
    with_env()
    live = pdt.yoff_epilogue_live_block()
    raws = written_input(with_env())
    ok = all(r.size == ROWS * P and np.array_equal(r, live.reshape(-1)) for r in raws)
    print(f"[CHECK c-live] hw_partial_sum_input_stencil.raw (app dir and bin/) == head {HEAD}'s columns of {LIVE}: {ok} "
          f"-> {'PASS' if ok else 'FAIL'}")
    ok_all &= ok
    lv, hv = from_bits(live).double(), host.double()
    exact = live == bits(host)
    out_tol = ((lv - hv).abs() > 1.5e-4 + 0.2 * hv.abs()).numpy()
    ok = live.any() and out_tol.mean() <= 0.06
    print(f"[CHECK d] live vs host block: {100 * exact.mean():.1f}% bit-exact, max |diff| {float((lv - hv).abs().max()):.4g}, "
          f"{100 * out_tol.mean():.2f}% out of tolerance (limit 6%), live nonzero {int((live != 0).sum())} -> "
          f"{'PASS' if ok else 'FAIL'}")
    ok_all &= bool(ok)
elif REQUIRE_LIVE:
    print(f"[CHECK c-live] live C@states output missing: {LIVE or '(YOFF_EPILOGUE_LIVE_TXT unset)'} -> FAIL")
    ok_all = False
else:
    print(f"[SKIP c-live, d] live C@states output not there yet: {LIVE or '(YOFF_EPILOGUE_LIVE_TXT unset)'}")

print("[RESULT]", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
