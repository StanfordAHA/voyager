"""The Mamba2 multiply bring-ups' per-run tensors and host models, for run_mul_tiles.sh.

kernel: x_end (#6 mul_11, mamba_x_end_fp), state_decay (#7 mul_13, mamba_state_decay_fp), mul10 (#5 mul_10,
rms_norm_pass2_fp), decay_chain (#3 mul_8 -> cumsum_1 -> exp -> mul_9, mamba_decay_chain_fp), exp2 (exp_2 = exp(a_cum),
mamba_exp_fp), state_decay_lane (#7 with the decay through one lane of exp_2's output), and the batched transposes on
transpose_fp: transpose (decay_to_end), permute (a), permute_2 (B^T). tensors(kernel, k) gives the run's app-dir raws
(name -> array, as the parser writes them) and the gold slice; model(kernel, raws, mul, k) the kernel's output with `mul`
as the multiply (the PE's rounding or exact RNE; the transposes ignore it).
"""
import numpy as np

BASE = "/aha/voyager/test/compiler/networks/mamba2_prefill/MXINT8/tensor_files/"
load = lambda node, *shape: np.fromfile(BASE + node + ".bin", dtype=np.float32).reshape(*shape)

# transpose_fp layers: input node, output node, batch, X, Y, input lanes ([batch, X, Y] -> [batch, Y, X]; the input in
# its natural lane layout, lane = y mod lanes; pass k reads lane k and writes the output rows y = k mod lanes)
TRANSPOSES = {
    "transpose": ("slice_17", "transpose", 4, 64, 256, 32),
    "permute": ("reshape_3", "permute", 4, 256, 64, 16),
    "permute_2": ("reshape_5", "permute_2", 4, 256, 128, 32),
}


def _transpose_tensors(kernel, k):
    """The pass's input = its GLB tile [words, 8 lanes] of the input image; gold = the output [batch Y, X] with the
    pass's rows kept and the rest 0."""
    src, dst, b, X, Y, lanes = TRANSPOSES[kernel]
    image = load(src, b * X, Y).reshape(b * X, Y // lanes, lanes).reshape(-1, lanes)
    gold = np.zeros((b * Y, X), dtype=np.float32)
    rows = (np.arange(b)[:, None] * Y + np.arange(k, Y, lanes)[None, :]).reshape(-1)
    gold[rows] = load(dst, b * Y, X)[rows]
    return {"input": image[:, 8 * (k // 8):8 * (k // 8) + 8]}, gold


def _transpose_model(kernel, raws, k):
    """out[b Y + yg lanes + k, x] = tile[(b X + x) Y / lanes + yg, k mod 8] (process.cpp's gold), the rest 0."""
    _, _, b, X, Y, lanes = TRANSPOSES[kernel]
    g = Y // lanes
    tile = raws["input"].astype(np.float64)
    out = np.zeros((b * Y, X))
    for bb in range(b):
        for yg in range(g):
            out[bb * Y + yg * lanes + k] = tile[(bb * X + np.arange(X)) * g + yg, k % 8]
    return out


def tensors(kernel, k):
    if kernel in TRANSPOSES:
        return _transpose_tensors(kernel, k)
    if kernel == "x_end":            # run k = 16 c + 2 hb + q: chunk c, heads 8 hb .. 8 hb + 7, lane half q
        c, hb, q = k // 16, (k // 2) % 8, k % 2
        rows = slice(256 * c, 256 * c + 256)
        half = lambda a: a[rows, 512 * hb:512 * hb + 512].reshape(256, 16, 2, 16)[:, :, q, :].reshape(256, 256)
        return ({"input": half(load("reshape", 1024, 4096)), "dt": load("softplus", 1024, 64)[rows, 8 * hb:8 * hb + 8],
                 "decay": load("transpose", 1024, 64)[rows, 8 * hb:8 * hb + 8]}, half(load("mul_11", 1024, 4096)))
    if kernel == "state_decay":      # run k = 2 u + q: chunk-decay GLB tile u (heads 8u + j, 32 + 8u + j), lane half q
        u, q = k // 2, k % 2
        run = lambda a: a.reshape(128, 2, 4, 8, 2, 2, 16)[:, :, u, :, :, q, :].reshape(128, 512)
        return ({"input": run(load("add_6", 128, 4096)), "decay": load("unsqueeze_7", 2, 4, 8)[:, u, :]},
                run(load("mul_13", 128, 4096)))
    if kernel == "mul10":            # run k = 32 c + 2 hg + q: chunk c, heads 4 hg .. 4 hg + 3, lane half q
        c, hg, q = k // 32, (k // 2) % 16, k % 2
        half = lambda a: a.reshape(*a.shape[:-1], 8, 2, 16)[..., q, :].reshape(*a.shape[:-2], 32768)
        return ({"input": half(load("mul_9", 4, 64, 256, 256)[c, 4 * hg:4 * hg + 4]),
                 "weight": half(load("unsqueeze_2", 4, 256, 256)[c])},
                half(load("mul_10", 4, 64, 256, 256)[c, 4 * hg:4 * hg + 4]))
    if kernel == "decay_chain":      # run k = 32 c + 16 t + 2 jl + q: chunk c, mul_6 GLB tile t, lane jl, lane half q
        c, t, jl, q = k // 32, (k // 16) % 2, (k // 2) % 8, k % 2
        mask = lambda node: load(node, 256, 8, 2, 16)[:, :, q, :]
        gold = load("mul_9", 4, 4, 2, 8, 256, 8, 2, 16)[c, :, t, jl, :, :, q, :]
        return ({"a": load("mul_6", 1024, 4, 2, 8)[:, :, t, :], "mask_strict": mask("_tensor_constant2"),
                 "mask_lower": mask("_tensor_constant3")}, gold)
    if kernel == "state_decay_lane":  # run k = 2 v + q: heads 16 v .. 16 v + 15, lane half q; decay = exp_2's GLB tile 3
        v, q = k // 2, k % 2
        run = lambda a: a.reshape(128, 4, 16, 2, 2, 16)[:, v, :, :, q, :].reshape(128, 512)
        return ({"input": run(load("add_6", 128, 4096)), "decay": load("exp_2", 256, 8, 4, 8)[:, :, 3, :].reshape(2048, 8)},
                run(load("mul_13", 128, 4096)))
    if kernel == "exp2":             # one run per layer: the whole a_cum [256 rows 64 c + h, 256 columns i]
        return {"input": load("cumsum", 256, 256)}, load("exp_2", 256, 256)
    raise ValueError(kernel)


def model(kernel, raws, mul, k=0):
    if kernel in TRANSPOSES:
        return _transpose_model(kernel, raws, k)
    if kernel == "decay_chain":
        from decay_chain_lib import run_model
        return run_model(raws, k, mul)
    if kernel == "exp2":             # the decay chain's exp table (clamp -127, RNE constant and ROM)
        from check_x_dt_host_lib import bits
        from decay_chain_lib import exp_chain_lut, f32
        return f32(exp_chain_lut(mul)[bits(raws["input"])])
    x = raws["input"].astype(np.float64)
    if kernel == "x_end":            # column 32 h + 16 g + j: head h = column // 32
        head = np.arange(x.shape[1]) // 32
        return mul(mul(x, raws["dt"].astype(np.float64)[:, head]), raws["decay"].astype(np.float64)[:, head])
    if kernel == "state_decay_lane":  # column 32 hh + 16 g + j: head 16 v + hh, decay word 512 + 8 h + 7 of lane 7
        head = 16 * (k // 2) + np.arange(x.shape[1]) // 32
        return mul(x, raws["decay"].astype(np.float64).reshape(2048, 8)[512 + 8 * head + 7, 7][None, :])
    if kernel == "state_decay":      # column 32 (8 s + h) + 16 g + j: decay[s, h]
        head = np.arange(x.shape[1]) // 32
        return mul(x, raws["decay"].astype(np.float64).reshape(-1)[head][None, :])
    if kernel == "mul10":
        return mul(x, raws["weight"].astype(np.float64)[None, :])
    raise ValueError(kernel)
