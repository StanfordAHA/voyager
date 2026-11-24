"""The Mamba2 fused decay chain's host models (#3, mamba_decay_chain_fp on strait masked_col_cumsum_exp_bf16).

exp_chain_lut(): the hardware exp chain z = max(fp_mul(S, bf16(1/ln2)), -127) -> fp_getffrac -> exp ROM, fp_getfint ->
fp_addiexp, as a 65,536-entry table over S's bf16 bits. The getffrac / getfint / addiexp steps come from lassen's own
functional model (lassen/float/float_custom.py) and are cached; numpy versions of the three ops are cross-checked
against it on every bf16 pattern. run_model(): one run's output from its app-dir raws (the scan as sequential bf16 fp_add
with exact RNE, the multiplies by `mul`).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_x_dt_host_lib import bits, bf_rne, pe_mul  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__", "decay_chain_lassen_ops.npz")
CLAMP = -127.0


def f32(b):
    b = np.asarray(b, dtype=np.int64)
    return (b.reshape(-1).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(b.shape)


def bf16_rne_bits(x):
    """strait's bf16_bits_rne (constants and ROM entries)."""
    u = np.float32(x).view(np.uint32).item()
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


INV_LN2_BITS = bf16_rne_bits(1.0 / np.log(2.0))   # 0x3FB9 = 1.4453125


def exp_rom():
    """strait elementwise_broadcast_add_softplus_bf16.exp_rom_table: 2^(i/128), i in [0, 128) then [-128, 0)."""
    return np.array([bf16_rne_bits(2.0 ** (i / 128.0)) for i in range(128)] +
                    [bf16_rne_bits(2.0 ** (i / 128.0)) for i in range(-128, 0)], dtype=np.int64)


def np_getffrac(a):
    a = np.asarray(a, dtype=np.int64)
    mant = (a & 0x7F) | 0x80
    e = ((a >> 7) & 0xFF) - 127
    sh = np.where(e < 0, mant >> np.minimum(-e, 31), (mant << np.clip(e, 0, 31)) & 0xFFFF)
    u = sh & 0x7F
    return np.where(a & 0x8000, (-u) & 0xFF, u & 0xFF)


def np_getfint(a):
    a = np.asarray(a, dtype=np.int64)
    mant = (a & 0x7F) | 0x80
    e = ((a >> 7) & 0xFF) - 127
    sh = np.where(e < 0, 0, (mant << np.clip(e, 0, 31)) & ((1 << 23) - 1))   # lassen shifts in 23 bits
    u = (sh >> 7) & 0xFFFF
    return np.where(a & 0x8000, (-u) & 0xFFFF, u)


def np_addiexp(a, b):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    exp = (((a >> 7) & 0xFF) + (b & 0xFF)) & 0xFF
    return (a & 0x8000) | (exp << 7) | (a & 0x7F)


def lassen_ops():
    """getffrac and getfint of every 16-bit pattern and addiexp(ROM[k], int) for every ROM entry k and int byte, from
    lassen's FPCustom (cached; ~1 minute to build)."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["frac"], d["fint"], d["addiexp"]
    from hwtypes import BitVector
    from lassen.alu import Signed_t
    from lassen.float.float_custom import FPCustom_fc, FPCustom_t
    fpc = FPCustom_fc.Py()
    call = lambda op, a, b: int(fpc(op, Signed_t.signed, BitVector[16](int(a)), BitVector[16](int(b)))[0]) & 0xFFFF
    frac = np.array([call(FPCustom_t.FGetFFrac, a, 0) for a in range(65536)], dtype=np.int64)
    fint = np.array([call(FPCustom_t.FGetFInt, a, 0) for a in range(65536)], dtype=np.int64)
    rom = exp_rom()
    addi = np.array([[call(FPCustom_t.FAddIExp, rom[k], b) for b in range(256)] for k in range(256)], dtype=np.int64)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, frac=frac, fint=fint, addiexp=addi)
    return frac, fint, addi


def exp_chain_lut(mul=pe_mul, clamp=CLAMP):
    """Output bits of the exp chain for every input bit pattern S (non-finite patterns -> 0)."""
    frac, fint, addi = lassen_ops()
    s = np.arange(65536, dtype=np.int64)
    finite = ((s >> 7) & 0xFF) != 0xFF
    v = np.where(finite, f32(s), 0.0)
    z = np.maximum(mul(v, f32(INV_LN2_BITS)), clamp)    # fp_max
    zb = bits(z)
    out = addi[frac[zb], fint[zb] & 0xFF]                # ROM[frac], then the exponent byte add
    return np.where(finite, out, 0)


def cross_check_numpy_ops():
    """numpy getffrac / getfint / addiexp == lassen on every pattern (returns the mismatch counts)."""
    frac, fint, addi = lassen_ops()
    s = np.arange(65536)
    rom = exp_rom()
    k, b = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    return (int(np.sum(np_getffrac(s) != frac)), int(np.sum(np_getfint(s) != fint)),
            int(np.sum(np_addiexp(rom[k], b) != addi)))


def run_digits(k):
    """Run k = 32 c + 16 t + 2 jl + q (CGRA_RUN_DIGITS c:4,t:2,jl:8,q:2)."""
    return k // 32, (k // 16) % 2, (k // 2) % 8, k % 2


def scan(x):
    """Sequential bf16 column scan down axis 1 of [mats, rows, cols] (fp_add, exact RNE), the Pond starting at 0."""
    s = np.zeros((x.shape[0], x.shape[2]))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[1]):
        s = bf_rne(s + x[:, i])
        out[:, i] = s
    return out


def run_model(raws, k, mul=pe_mul, lut=None, return_scan=False, scan_rows=256, num_mats=4, a_row_words=4):
    """One run's [num_mats * scan_rows, cols] output from the app-dir raws (`a` [a_rows, 8], masks [scan_rows, cols])."""
    c, t, jl, q = run_digits(k)
    a_lane = np.asarray(raws["a"], dtype=np.float64).reshape(-1, 8)[:, jl]
    start = 1024 * c
    a = a_lane[start + a_row_words * np.arange(scan_rows)[None, :] + np.arange(num_mats)[:, None]]   # [mats, rows]
    strict = np.asarray(raws["mask_strict"], dtype=np.float64).reshape(scan_rows, -1)
    lower = np.asarray(raws["mask_lower"], dtype=np.float64).reshape(scan_rows, -1)
    x = mul(a[:, :, None], strict[None])
    s = scan(x)
    lut = exp_chain_lut(mul) if lut is None else lut
    e = f32(lut[bits(s)])
    out = mul(e, lower[None]).reshape(num_mats * scan_rows, -1)
    return (out, s) if return_scan else out
