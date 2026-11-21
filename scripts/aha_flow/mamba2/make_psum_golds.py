"""Host-computed partial-sum golds for a psum-chained Zircon GEMM (Mamba2 out_proj, xBC_proj, lm_head).

Kernel p of the chain outputs [residual +] the sum of the MU partial GEMMs over reduction slices 0..p. On
hardware each MU partial is rounded to BF16 and the CGRA adds it to the previous partial sum in BF16, so:
    running_{-1} = bf16(residual), or 0 when the layer has no residual (kernel 0 is then zircon_2d_nop)
    running_p    = bf16(running_{p-1} + bf16(A[:, slice_p] @ W[slice_p, :]))
with A and W the dequantized MXINT8 operands (int8 values times their e8m0 block scales, blocks of 64 along C).

Each running_p is sliced and ordered exactly like voyager's gold_activation.txt by calling
parse_dnnLayer_tensors.parse_output on it with the kernel's host-tiling env (K/X tile index), and written as
<out_dir>/kernel_<p>_output.txt. The last one is also compared with voyager's own gold (the model.txt output
tensor) using test.py's fp criteria.

Usage: python make_psum_golds.py <layer> <app key of kernel 0 in application_parameters.json> <out_dir>
                                  [--pass1-lower <app key of the SiLU pass-1 kernel> <out file>]
  <layer> is a key of LAYERS (the model.txt layer name). For lm_head, run once per non-uniform K slice
  with that slice's kernel-0 key.
  --pass1-lower (GEMM -> SiLU layers, e.g. z_proj): also writes the gold of add_gelu_pass1_mu_input_fp's
  hw_psum1_lower_output, i.e. the last reduction slice's MU partial bf16(A[:, slice] @ W[slice, :]) on MU lanes
  16..31 of each 32 (channel trim index 1), sliced with that kernel's host-tiling env.
"""
import json, os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
import parse_dnnLayer_tensors as pdt

APP_PARAMS = "/aha/aha/util/application_parameters.json"
NETWORK_DIR = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8"
MODEL, BLOCK = "mamba2_prefill", 64
# layer -> (activation, activation scale, weight, weight scale, residual or None, X, C, K), from model.txt
LAYERS = {
    # out_proj (linear_mx_default_3) + add_12 (residual)
    "submodule_9": ("getitem_12", "getitem_11", "_param_constant9_quantized_0", "_param_constant9_scale_0",
                    "arg0_1", 1024, 4096, 2048),
    # xBC_proj
    "linear_mx_default_1": ("getitem_4", "getitem_3", "_param_constant2_quantized_0", "_param_constant2_scale_0",
                            None, 1024, 2048, 4352),
    # z_proj (its gate SiLU is fused into the last psum kernel + a CGRA pass 2)
    "linear_mx_default": ("getitem_4", "getitem_3", "_param_constant1_quantized_0", "_param_constant1_scale_0",
                          None, 1024, 2048, 4096),
    # dt_proj (linear_mx_default_2) + add_5 (dt_bias [64], broadcast over positions) = submodule_4
    "submodule_4": ("getitem_4", "getitem_3", "_param_constant3_quantized_0", "_param_constant3_scale_0",
                    "_param_constant6", 1024, 2048, 64),
    # lm_head (vocabulary padded to 50304)
    "linear_mx_default_4": ("getitem_14", "getitem_13", "_param_constant11_quantized_0", "_param_constant11_scale_0",
                            None, 1024, 2048, 50304),
}


def load(node, shape):
    return torch.from_numpy(np.fromfile(f"{NETWORK_DIR}/tensor_files/{node}.bin", dtype=np.float32).reshape(shape))


def bf16(t):
    return t.to(torch.bfloat16).to(torch.float32)


def fp_check(gold, sim, atol=1.5e-4, rtol=0.2, max_mismatch=0.06):
    """test.py's fp comparison: |sim - gold| <= atol + rtol * |gold| for all but at most 6% of the elements."""
    gold, sim = np.asarray(gold, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    bad = np.abs(sim - gold) > atol + rtol * np.abs(gold)
    return bad.mean(), bad.mean() <= max_mismatch


def bits_to_float(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def main(layer, app_key, out_dir, pass1_lower=None):
    ACT, ACT_SCALE, WEIGHT, WEIGHT_SCALE, RESIDUAL, X, C, K = LAYERS[layer]
    LAYER = layer
    env = json.load(open(APP_PARAMS))[app_key]["default"]
    os.environ.update(env)
    num_psums = int(env["NUM_PSUMS"])
    zw = pdt.parse_zircon_workarounds()
    os.makedirs(out_dir, exist_ok=True)

    act = load(ACT, (1, X, C)) * load(ACT_SCALE, (1, X, C // BLOCK)).repeat_interleave(BLOCK, dim=-1)
    weight = load(WEIGHT, (C, K)) * load(WEIGHT_SCALE, (C // BLOCK, K)).repeat_interleave(BLOCK, dim=0)
    act, weight = act.double(), weight.double()
    if RESIDUAL:  # a [K] residual (e.g. dt_bias) broadcasts over the rows, as RESIDUAL_BROADCAST_TO_OUTPUT does
        res = torch.from_numpy(np.fromfile(f"{NETWORK_DIR}/tensor_files/{RESIDUAL}.bin", dtype=np.float32))
        running = bf16((res.reshape(1, 1, K).expand(1, X, K) if res.numel() == K else res.reshape(1, X, K)).contiguous())
    else:
        running = torch.zeros(1, X, K)
    out_spec = {"node": LAYER, "shape": [1, X, K], "dtype": "bfloat16"}
    slice_c = C // num_psums

    with tempfile.TemporaryDirectory() as tmp:
        for p in range(num_psums):
            partial = act[..., p * slice_c:(p + 1) * slice_c] @ weight[p * slice_c:(p + 1) * slice_c]
            partial = bf16(partial.float())
            running = bf16(running + partial)
            running.numpy().astype(np.float32).tofile(f"{tmp}/{LAYER}.bin")
            bits = pdt.parse_output(MODEL, LAYER, f"{tmp}/", out_spec, zw)
            pdt.write_list_to_hex(bits, f"{out_dir}/kernel_{p}_output.txt", add_metadata=False, hex_width=4)
            print(f"[INFO] kernel_{p}_output.txt: {len(bits)} values", flush=True)

    # The last partial sum against voyager's gold (the compiled model's output tensor), same slicing
    voyager = bits_to_float(pdt.parse_output(MODEL, LAYER, f"{NETWORK_DIR}/tensor_files/", out_spec, zw))
    host = bits_to_float(bits)
    frac, ok = fp_check(voyager, host)
    print(f"[INFO] host kernel_{num_psums - 1} vs voyager gold: bit-exact {np.mean(host == voyager):.4f}, "
          f"max |diff| {np.abs(host - voyager).max():.4g}, test.py mismatch {frac:.4%} -> {'PASS' if ok else 'FAIL'}")

    if pass1_lower:
        # The SiLU pass-1 kernel passes the last slice's raw MU partial through on lanes 16..31 of each 32
        p1_key, p1_out = pass1_lower
        os.environ.update(json.load(open(APP_PARAMS))[p1_key]["default"])
        os.environ.pop("SILU_GOLD_INPUT_NODE", None)
        os.environ["GOLD_CHANNEL_TRIMMING_WORKAROUND"] = "1"
        os.environ["GOLD_CHANNEL_TRIMMING_WORKAROUND_NUM_KERNELS"] = "2"
        os.environ["GOLD_CHANNEL_TRIMMING_WORKAROUND_KERNEL_IDX"] = "1"
        zw1 = pdt.parse_zircon_workarounds()
        with tempfile.TemporaryDirectory() as tmp:
            partial.numpy().astype(np.float32).tofile(f"{tmp}/{LAYER}.bin")
            lower = pdt.parse_output(MODEL, LAYER, f"{tmp}/", out_spec, zw1)
        os.makedirs(os.path.dirname(p1_out), exist_ok=True)
        pdt.write_list_to_hex(lower, p1_out, add_metadata=False, hex_width=4)
        print(f"[INFO] pass-1 lower gold {p1_out}: {len(lower)} values (slice {num_psums - 1} MU partial, lanes 16..31)", flush=True)


if __name__ == "__main__":
    extra = sys.argv[4:]
    if extra and (extra[0] != "--pass1-lower" or len(extra) != 3):
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], tuple(extra[1:]) if extra else None)
