"""Helpers shared by the Mamba2 multiply host checks (check_mul_gather_host.py): bf16 bit and rounding models, GLB DMA
and lake address walks, the lake read-port rate model, get_interleave_mem's data order, and design wiring."""
import numpy as np


def bits(a):
    return (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64)


def bf_rne(a):
    """bf16 round to nearest even, subnormals included: the quantum is 2^(e - 8) for |a| = m 2^e (m in [0.5, 1)),
    and 2^-133 below 2^-126 (bf16's subnormal grid; voyager / PyTorch round there, e.g. mul_10, mul_11)."""
    a = np.asarray(a, dtype=np.float64)
    _, e = np.frexp(a)
    q = np.ldexp(1.0, np.maximum(e, -125) - 8)
    return np.rint(a / q) * q


def pe_mul(a, b):
    """Zircon PE fp_mul: round half to even from the product truncated to 3 bits below the bf16 LSB (no sticky). The
    LSB is 2^(e - 8) for |p| = m 2^e, and 2^-133 below 2^-126: subnormal results are rounded on bf16's subnormal grid
    the same way (matched the RTL on 100% of mamba_x_end_fp run 0, 942 subnormal products, 2026-11-22)."""
    p = np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
    _, e = np.frexp(p)
    lsb = np.ldexp(1.0, np.maximum(e, -125) - 8)
    t = np.trunc(np.abs(p) / lsb * 8.0) / 8.0
    r = np.floor(t)
    f = t - r
    return np.sign(p) * (r + ((f > 0.5) | ((f == 0.5) & (r % 2 == 1)))) * lsb


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


def interleave_read(size, turn, w0, w1):
    """The values a get_interleave_mem MEM emits, from lake's own write and read address patterns."""
    from lake.spec.hack_rv_mem_pond_bitstream import get_interleave_mem
    cfg = get_interleave_mem(size, turn=turn)
    sram = {}
    for port, vals in ((0, w0), (1, w1)):
        addrs = lake_walk(cfg[port]["config"])
        assert len(addrs) == len(vals) == size
        sram.update(zip(addrs.tolist(), vals))
    return [sram[a] for a in lake_walk(cfg[3]["config"]).tolist()]


def design_drivers(inst, conns):
    """(instance, sink port) -> (driving instance, port) from a design_top.json module's connections."""
    sinks = ("data_in_0", "data_in_1", "data0", "data1", "in")
    drivers = {}
    for a, b in conns:
        for src, dst in ((a, b), (b, a)):
            dn, _, dp = dst.partition(".")
            sn, _, sp = src.partition(".")
            if dn in inst and dp in sinks and sn != "self" and not sn.startswith("const_inst") \
                    and sp in ("out", "O0", "data_out_0", "data_out_1"):
                drivers[(dn, dp)] = (sn, sp)
    return drivers
