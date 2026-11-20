"""Host checks for the Mamba2 z_proj -> SiLU bring-up (tile (0,0)), run after make_psum_golds.py.

1. The SiLU gold the flow writes for pass 1 (trim index 0) and pass 2 (trim index 1), via parse_output with
   SILU_GOLD_INPUT_NODE, interleaves back into the host SiLU of the full 512 x 512 GEMM tile, bit for bit.
2. That host SiLU tile is close to the graph's own silu.bin tile (test.py fp criteria).
3. The pass-1 lower gold (slice-7 MU partial, lanes 16..31) plus the host kernel-6 running sum (lanes 16..31)
   matches voyager's GEMM tile on lanes 16..31 (test.py fp criteria): this is what pass 2 adds before its SiLU.
Prints PASS/FAIL per check and exits 1 on any failure.

Usage: python check_zproj_silu_golds.py <host gold dir>
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
import parse_dnnLayer_tensors as pdt

APP_PARAMS = "/aha/aha/util/application_parameters.json"
BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
PASS1, PASS2 = "add_gelu_pass1_mu_input_fp_post_mamba2_prefill_z_projection_kernel7", "add_gelu_pass2_fp_mamba2_prefill"
GEMM_SPEC = {"node": "linear_mx_default", "shape": [1, 1024, 4096], "dtype": "bfloat16"}
SILU_SPEC = {"node": "silu", "shape": [1, 1024, 4096], "dtype": "bfloat16"}
PARAMS = json.load(open(APP_PARAMS))
CLEAN_ENV = dict(os.environ)


def with_env(key, **overrides):
    os.environ.clear()
    os.environ.update(CLEAN_ENV)
    os.environ.update(PARAMS[key]["default"])
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return pdt.parse_zircon_workarounds()


def gold(key, layer, spec, **overrides):
    zw = with_env(key, **overrides)
    return np.asarray(pdt.parse_output("mamba2_prefill", layer, BASE, spec, zw), dtype=np.uint16)


def f32(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def fp_check(g, s, atol=1.5e-4, rtol=0.2, limit=0.06):
    bad = np.abs(s.astype(np.float64) - g) > atol + rtol * np.abs(g.astype(np.float64))
    return bad.mean(), bad.mean() <= limit


def read_txt(path):
    return np.array([int(h, 16) for h in open(path).read().split()], dtype=np.uint16)


host_gold = sys.argv[1]
ok_all = True
upper = gold(PASS1, "linear_mx_default", GEMM_SPEC)                    # trim 0, SiLU hook on
lower_silu = gold(PASS2, "silu", SILU_SPEC)                            # trim 1, SiLU hook on, IO tiling
x = torch.from_numpy(np.fromfile(BASE + "linear_mx_default.bin", dtype=np.float32).reshape(1024, 4096))[:512, :512]
x = x.double()
silu_tile = (x * torch.sigmoid(x)).to(torch.bfloat16).to(torch.float32).numpy()
silu_bits = (silu_tile.view(np.uint32) >> 16).astype(np.uint16).reshape(512, 16, 32)
interleaved = np.concatenate([upper.reshape(512, 16, 16), lower_silu.reshape(512, 16, 16)], axis=2)
ok = upper.size == lower_silu.size == 131072 and np.array_equal(interleaved, silu_bits)
print(f"[CHECK 1] pass-1 (trim 0) + pass-2 (trim 1) SiLU golds interleave to the host SiLU tile: {'PASS' if ok else 'FAIL'} "
      f"({upper.size} + {lower_silu.size} values)")
ok_all &= ok

graph = np.fromfile(BASE + "silu.bin", dtype=np.float32).reshape(1024, 4096)[:512, :512]
frac, ok = fp_check(graph, silu_tile)
print(f"[CHECK 2] host SiLU tile vs graph silu.bin tile: bit-exact {np.mean(graph == silu_tile):.4f}, "
      f"max |diff| {np.abs(graph - silu_tile).max():.4g}, mismatch {frac:.4%} -> {'PASS' if ok else 'FAIL'}")
ok_all &= ok

lower_partial = f32(read_txt(f"{host_gold}/pass1_hw_psum1_lower_output.txt"))
# Lanes 16..31 of each 32 in row-major tile order, the same as trim_channels_if_needed with index 1
k6 = read_txt(f"{host_gold}/kernel_6_output.txt").reshape(-1, 2, 16)[:, 1, :].reshape(-1)
gemm_tile = np.fromfile(BASE + "linear_mx_default.bin", dtype=np.float32).reshape(1024, 4096)[:512, :512]
gemm_lower = gemm_tile.reshape(512, 16, 2, 16)[:, :, 1, :].reshape(-1)
summed = (torch.from_numpy(lower_partial + f32(k6))).to(torch.bfloat16).to(torch.float32).numpy()
frac, ok = fp_check(gemm_lower, summed)
print(f"[CHECK 3] slice-7 partial + kernel-6 running sum (lanes 16..31) vs GEMM tile lanes 16..31: "
      f"max |diff| {np.abs(gemm_lower - summed).max():.4g}, mismatch {frac:.4%} -> {'PASS' if ok else 'FAIL'}")
ok_all &= ok
print("[RESULT]", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
