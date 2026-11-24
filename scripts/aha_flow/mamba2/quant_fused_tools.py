"""Helpers for run_quantize_fused.sh (Mamba2 MX quantization on yuchen's one-pass fused apps, tile 0 each, 131,072 values):
fused_e8m0_quant_single_IO for the quantize_mx layers (blocks along dim -1) and fused_e8m0_quant_multi_IOs for the
calculate_mx_qparam + quantize pairs (blocks along dim -2, run on the quantize layer with FUSED_SCALE_GOLD).
  list                      the runs, in run order: name layer key app
  host-check                per run, before RTL: the parser's input / golds (with the key's env) vs the tensors, the new
                            golds vs the two-pass golds on the overlap, and the HW model's predicted test.py verdict
  check <name> <rundir>     after a run: the inputs the app loaded, the golds, and the saved RTL outputs vs the HW model
                            and vs the two-pass RTL outputs on the overlap
  report <logdir> <out.md>
HW model (validated == RTL on all 20 two-pass runs): scale code = bf16 exponent field of the block |max| - 6 (mod 256;
127 for an all-zero block); int8 = x / 2^(code - 127) rounded half away from zero, wrapped to 8 bits.
QUANT_MODEL=llama_prefill selects Llama's 6 layers (same apps and rule; no two-pass RTL outputs to compare against, so
the golds are checked against voyager's tensors directly). Run with /aha/bin/python (NumPy 1.x)."""
import json, os, re, sys, tempfile, time
import numpy as np

MODEL = os.environ.get("QUANT_MODEL", "mamba2_prefill")
B = f"/aha/voyager/test/compiler/networks/{MODEL}/MXINT8/tensor_files/"
APPS = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps"
COLLATERAL = "/aha/voyager/compiled_collateral"
TWO_PASS = "/aha/rerun_logs/2026-11-21/quantize_two_pass"
KEYS = json.load(open("/aha/aha/util/application_parameters.json"))
SHAPES = {"layer_norm_default": [1, 1024, 2048], "permute_3": [1, 4, 1, 128, 256], "reshape_8": [1, 4, 1, 256, 128],
          "mul_10": [1, 4, 1, 64, 256, 256], "submodule_8": [1, 1024, 4096], "layer_norm_default_1": [1, 1024, 2048],
          "permute_1": [1, 4, 1, 64, 256, 64], "permute_2": [1, 4, 1, 128, 256], "reshape_12": [1, 4, 1, 256, 4096],
          "reshape_14": [1, 4, 1, 128, 4096]}
SINGLE = [("_5", "_kernel0", "layer_norm_default_1"), ("_4", "_kernel0", "submodule_8"), ("_1", "", "permute_3"),
          ("_2", "", "reshape_8"), ("_3", "_kernel0", "mul_10")]
MULTI = [("_1", "", "permute_2"), ("", "_kernel0", "permute_1"), ("_2", "_kernel0", "reshape_12"), ("_3", "_kernel0", "reshape_14")]
ORDER = ["single_5", "multi_1", "single_4", "single_1", "single_2", "multi", "multi_2", "single_3", "multi_3"]
RESHAPE = {}   # node -> IO_RESHAPE_LAST_DIM of its key (the tile's row width)
if MODEL == "llama_prefill":
    TWO_PASS = "/nonexistent"   # no saved two-pass RTL outputs for Llama
    SHAPES = {"layer_norm_default": [1, 512, 2048], "submodule_4": [1, 32, 512, 64], "submodule_40": [1, 512, 8192],
              "dropout_1": [1, 1, 512, 512], "reshape_1": [1, 32, 512, 64], "transpose_3": [1, 32, 64, 512]}
    RESHAPE = {"submodule_4": 128}   # one 64-value block per row: two rows per 128-wide row (the template needs >= 2 blocks)
    SINGLE = [("", "_kernel0", "layer_norm_default"), ("_1", "_kernel0", "submodule_4"), ("_5", "_kernel0", "submodule_40"),
              ("_7", "_kernel0", "dropout_1")]
    MULTI = [("", "_kernel0", "reshape_1"), ("_1", "_kernel0", "transpose_3")]
    ORDER = ["single", "single_1", "single_5", "single_7", "multi", "multi_1"]


def runs():
    out = {}
    for sfx, ksfx, node in SINGLE:
        layer = f"quantize_mx_default{sfx}"
        out[f"single{sfx}"] = dict(name=f"single{sfx}", kind="single", layer=layer, node=node, app="fused_e8m0_quant_single_IO",
                                   key=f"fused_e8m0_quant_single_IO_{MODEL}_{layer}{ksfx}",
                                   old_data=f"{TWO_PASS}/single{sfx}", old_scale=f"{TWO_PASS}/tree{sfx}")
    for sfx, ksfx, node in MULTI:
        layer = f"quantize_default{sfx}"
        out[f"multi{sfx}"] = dict(name=f"multi{sfx}", kind="multi", layer=layer, node=node, app="fused_e8m0_quant_multi_IOs",
                                  key=f"fused_e8m0_quant_multi_IOs_{MODEL}_{layer}{ksfx}", calc=f"calculate_mx_qparam_default{sfx}",
                                  old_data=f"{TWO_PASS}/multi{sfx}", old_scale=f"{TWO_PASS}/accum{sfx}")
    return [out[n] for n in ORDER]


RUNS = {r["name"]: r for r in runs()}
bits16 = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64)
raw_be = lambda f: np.fromfile(f, dtype=">u2").astype(np.int64) if os.path.exists(f) else None


def load_hex_block0(path):  # test.py's _load_hex_txt(block_index=0)
    blocks, cur = [], []
    for line in open(path):
        if line.strip():
            cur.extend(int(v, 16) for v in line.split())
        elif cur:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    return np.array(blocks[0] if blocks else [], dtype=np.int64)


def unpack(words):  # test.py's unpack_output: low byte, then high byte
    out = np.empty(words.size * 2, dtype=np.int64)
    out[0::2], out[1::2] = words & 0xFF, (words >> 8) & 0xFF
    return out


def env(r):
    return KEYS[r["key"]]["default"]


def tile_rc(r):
    """(rows, cols) of tile 0 and of the whole [rows, last] view."""
    C = RESHAPE.get(r["node"], SHAPES[r["node"]][-1])
    R = int(np.prod(SHAPES[r["node"]])) // C
    e, rows, cols = env(r), R, C
    if e.get("IO_DIM1_TILING") == "1":
        rows //= int(e["NUM_IO_DIM1_TILING_KERNELS"])
    if e.get("IO_INNERMOST_DIM_TILING") == "1":
        cols //= int(e["NUM_IO_INNERMOST_DIM_TILING_KERNELS"])
    return rows, cols, R, C


def x_tile(r):
    rows, cols, R, C = tile_rc(r)
    return np.fromfile(B + r["node"] + ".bin", dtype=np.float32).reshape(R, C)[:rows, :cols]


def hw_model(x, axis):
    """HW MX quantizer on a [rows, cols] bf16 tile, blocks of 64 along axis (1: dim -1, single_IO; 0: dim -2, multi_IOs).
    Returns (int8 as unsigned bytes [rows, cols], scale codes [rows, cols / 64] or [rows / 64, cols])."""
    x = np.asarray(x, dtype=np.float32)
    xb = x.reshape(x.shape[0], -1, 64) if axis == 1 else x.reshape(-1, 64, x.shape[1])
    amax = np.abs(xb).max(axis=2 if axis == 1 else 1)
    ef = ((amax.view(np.uint32) >> 23) & 0xFF).astype(np.int64)
    codes = np.where(ef == 0, 127, (ef - 6) % 256)
    scale = np.exp2((codes - 127).astype(np.float64))
    sc = np.repeat(scale, 64, axis=1) if axis == 1 else np.repeat(scale, 64, axis=0)
    v = x.astype(np.float64) / sc
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int64)
    return (q % 256), codes


def explain(x, axis, sq, ss, gq, gs):
    """Attribute every RTL-vs-voyager difference to one of the 4 HW quantizer mechanisms (valid when RTL == HW model):
    1 exact exponent: the HW scale is the exact exponent - 6; voyager's floor(bf16 log2(|max|)) rounds up near 2^k, so its
      code is the HW code + 1 and the whole block differs;
    2 int8 wrap: a value that rounds to +128 wraps to -128 (voyager clips to 127);
    3 rounding half away from zero: a tie (|x / scale| = n + 1/2) rounds away from zero (voyager to even);
    4 zero / tiny blocks: an all-zero block gets code 127, a block with |max| < 2^-121 (exponent field < 6) wraps to a
      code 250..255 and quantizes to 0.
    Returns (counts by mechanism, number unexplained, summary text)."""
    x = np.asarray(x, dtype=np.float32)
    rows, cols = x.shape
    xb = x.reshape(rows, -1, 64) if axis == 1 else x.reshape(-1, 64, cols)
    ef = ((np.abs(xb).max(axis=2 if axis == 1 else 1).view(np.uint32) >> 23) & 0xFF).astype(np.int64)
    codes = np.where(ef == 0, 127, (ef - 6) % 256)
    sc = np.exp2((codes - 127).astype(np.float64))
    expand = (lambda a: np.repeat(a, 64, axis=1)) if axis == 1 else (lambda a: np.repeat(a, 64, axis=0))
    v = x.astype(np.float64) / expand(sc)
    qpre = np.sign(v) * np.floor(np.abs(v) + 0.5)
    ss, gs = ss.reshape(codes.shape), gs.reshape(codes.shape)
    sq, gq = sq.reshape(rows, cols), gq.reshape(rows, cols)
    # scale differences, per block
    sdiff = ss != gs
    m4b = sdiff & (ef < 6)
    m1b = sdiff & ~m4b & (gs == ss + 1)
    un_b = sdiff & ~m4b & ~m1b
    # int8 differences, per value
    qdiff = sq != gq
    in1, in4 = expand(m1b) & qdiff, expand(m4b) & qdiff
    rest = qdiff & ~expand(sdiff)
    m2 = rest & ((qpre > 127) | (qpre < -128))
    m3 = rest & ~m2 & (np.abs(v) % 1 == 0.5)
    un_q = (qdiff & expand(un_b)) | (rest & ~m2 & ~m3)
    counts = dict(scale_diff_blocks=int(sdiff.sum()), m1_blocks=int(m1b.sum()), m1_values=int(in1.sum()), m2_values=int(m2.sum()),
                  m3_values=int(m3.sum()), m4_blocks=int(m4b.sum()), m4_values=int(in4.sum()))
    unexplained = int(un_b.sum() + un_q.sum())
    text = (f"vs voyager, {int(qdiff.sum())} int8 and {int(sdiff.sum())} scale differences: (1) exact exponent {counts['m1_blocks']} blocks "
            f"/ {counts['m1_values']} int8; (2) wrap {counts['m2_values']}; (3) half-away ties {counts['m3_values']}; (4) zero/tiny blocks "
            f"{counts['m4_blocks']} / {counts['m4_values']} int8; UNEXPLAINED {unexplained}")
    return counts, unexplained, text


def voyager_golds(r):
    """Voyager's int8 (unsigned bytes) and scale codes for tile 0, from the tensor files, as the parser builds them.
    multi_IOs: the quantize layer and its calculate_mx_qparam (blocks along rows); single_IO: the quantize_mx layer's
    outputs <layer>_1 (int8) and <layer>_0 (scale; blocks along the row)."""
    rows, cols, R, C = tile_rc(r)
    if r["kind"] == "single":
        q = np.fromfile(B + r["layer"] + "_1.bin", dtype=np.float32).reshape(R, C)[:rows, :cols]
        s = np.fromfile(B + r["layer"] + "_0.bin", dtype=np.float32).reshape(R, C // 64)[:rows, :cols // 64]
    else:
        q = np.fromfile(B + r["layer"] + ".bin", dtype=np.float32).reshape(R, C)[:rows, :cols]
        s = np.fromfile(B + r["calc"] + ".bin", dtype=np.float32).reshape(R // 64, C)[:rows // 64, :cols]
    codes = ((s.view(np.uint32) >> 23) & 0xFF).astype(np.int64)
    if env(r).get("E8M0_ZERO_BLOCK_CODE"):
        codes[codes == 0] = int(env(r)["E8M0_ZERO_BLOCK_CODE"])
    return q.astype(np.int64) % 256, codes


def old_overlap(r, vals, what):
    """A two-pass output/gold (flat values) restricted to this run's tile 0: a row prefix, or a column slice."""
    rows, cols, R, C = tile_rc(r)
    if r["kind"] == "single":
        n = rows * cols if what == "data" else rows * cols // 64
        return vals[:n] if vals.size >= n else None
    # the two-pass accum / multi_IOs tile 0 (their keys were removed on 2026-11-22): (row tiles, column tiles)
    nrow, ncol = {"quantize_default": (16, 1), "quantize_default_1": (1, 1), "quantize_default_2": (1, 16),
                  "quantize_default_3": (1, 8)}[r["layer"]]
    ocols, orows = C // ncol, R // nrow
    if what == "scale":
        orows, rows = orows // 64, rows // 64
    if vals.size < orows * ocols:
        return None
    return vals[:orows * ocols].reshape(orows, ocols)[:rows, :cols].reshape(-1)


def verdict(sim, gold):
    """test.py's integer check on unsigned values: <= 15 off by > 1 and <= 20% off by 1."""
    d = np.abs(sim - gold)
    hard, soft = int((d > 1).sum()), float((d == 1).mean())
    return ("PASS" if hard <= 15 and soft <= 0.2 else "FAIL"), hard, soft


def signed_hard(sim, gold):
    s8 = lambda a: np.where(a > 127, a - 256, a)
    return int((np.abs(s8(sim) - s8(gold)) > 1).sum())


def host_check():
    sys.path.insert(0, "/aha/voyager/scripts/aha_flow")
    import parse_dnnLayer_tensors as P
    for r in runs():
        e = env(r)
        for v in [k for k in os.environ if "TILING" in k or k in ("IO_FLATTEN_TO_3D", "E8M0_ZERO_BLOCK_CODE", "IO_RESHAPE_LAST_DIM")]:
            os.environ.pop(v)
        for k, v in e.items():
            if "TILING" in k or k in ("IO_FLATTEN_TO_3D", "E8M0_ZERO_BLOCK_CODE", "IO_RESHAPE_LAST_DIM"):
                os.environ[k] = v
        zw = P.parse_zircon_workarounds()
        d = tempfile.mkdtemp(dir=os.environ.get("SCRATCH", "/tmp"))
        rows, cols, R, C = tile_rc(r)
        x = x_tile(r)
        P.parse_input_bf16_cgra(B, {"node": r["node"], "shape": SHAPES[r["node"]]}, d, zw, MODEL, r["layer"])
        got = raw_be(d + "/hw_input_stencil.raw")
        msg = [f"input == tile 0 of {r['node']} [{rows} x {cols}]: {np.array_equal(got, bits16(x).reshape(-1))}"]
        q_model, s_model = hw_model(x, 1 if r["kind"] == "single" else 0)
        if r["kind"] == "multi":
            sshape = SHAPES[r["node"]][:-2] + [SHAPES[r["node"]][-2] // 64, SHAPES[r["node"]][-1]]
            gq = np.array(P.parse_output(MODEL, r["layer"], B, {"node": r["layer"], "shape": SHAPES[r["node"]], "dtype": "int8"}, zw))
            gs = np.array(P.parse_output(MODEL, r["layer"], B, {"node": r["calc"], "shape": sshape, "dtype": "fp8_e8m0"}, zw))
            vq, vs = voyager_golds(r)
            msg.append(f"int8 gold == quantize tile ({gq.size}): {np.array_equal(gq, vq.reshape(-1))}; "
                       f"scale gold == calc tile ({gs.size}): {np.array_equal(gs, vs.reshape(-1))}")
            og = os.path.join(r["old_data"], "compare", "gold_activation.txt")
            os_ = os.path.join(r["old_scale"], "compare", "gold_activation.txt")
        elif MODEL != "mamba2_prefill":
            # no two-pass golds: the parser's golds (as the flow builds them) vs voyager's tensors
            sshape = SHAPES[r["node"]][:-1] + [SHAPES[r["node"]][-1] // 64]
            gq = np.array(P.parse_output(MODEL, r["layer"], B, {"node": r["layer"] + "_1", "shape": SHAPES[r["node"]], "dtype": "int8"}, zw))
            gs = np.array(P.parse_output(MODEL, r["layer"], B, {"node": r["layer"] + "_0", "shape": sshape, "dtype": "fp8_e8m0"}, zw))
            vq, vs = voyager_golds(r)
            msg.append(f"int8 gold == {r['layer']}_1 tile ({gq.size}): {np.array_equal(gq, vq.reshape(-1))}; "
                       f"scale gold == {r['layer']}_0 tile ({gs.size}): {np.array_equal(gs, vs.reshape(-1))}")
            og = os_ = "/nonexistent"
        else:
            gq = gs = None
            og = os.path.join(r["old_data"], "compare", "gold_activation.txt")
            os_ = os.path.join(r["old_data"], "compare", "gold_scale.txt")
            if not os.path.exists(os_):
                os_ = os.path.join(r["old_scale"], "compare", "gold_scale.txt")
        for what, path, new in (("data", og, gq), ("scale", os_, gs)):
            if not os.path.exists(path):
                if MODEL == "mamba2_prefill":
                    msg.append(f"two-pass {what} gold: missing")
                continue
            old = old_overlap(r, load_hex_block0(path), what)
            if new is None:   # single_IO: the two-pass gold on the overlap is the expected gold
                new = old
                if what == "data":
                    gq = old
                else:
                    gs = old
            msg.append(f"two-pass {what} gold on the overlap == new gold: {old is not None and np.array_equal(old, new)}")
        pq = verdict(q_model.reshape(-1), gq.reshape(-1)) if gq is not None else None
        ps = verdict(s_model.reshape(-1), gs.reshape(-1)) if gs is not None else None
        msg.append(f"HW-model prediction vs voyager: int8 {pq[0]} ({pq[1]} off by >1, {100 * pq[2]:.2f}% off by 1, signed {signed_hard(q_model.reshape(-1), gq.reshape(-1))} off by >1); "
                   f"scale {ps[0]} ({ps[1]} off by >1, {100 * ps[2]:.2f}% off by 1)")
        print(f"{r['name']} ({r['key']}): " + "; ".join(msg), flush=True)


def check(name, rundir):
    r = RUNS[name]
    x = x_tile(r)
    want = bits16(x).reshape(-1)
    d = f"{APPS}/{r['app']}"
    a, b = raw_be(f"{d}/hw_input_stencil.raw"), raw_be(f"{d}/bin/input_bf_act_host_stencil.raw")
    msg = [f"inputs (app dir, bin/) == tile 0 of {r['node']} ({want.size}): {a is not None and np.array_equal(a, want)}, "
           f"{b is not None and np.array_equal(b, want)}"]
    q_model, s_model = hw_model(x, 1 if r["kind"] == "single" else 0)
    q_model, s_model = q_model.reshape(-1), s_model.reshape(-1)
    gq_p, gs_p = f"{rundir}/compare/gold_activation.txt", f"{rundir}/compare/gold_scale.txt"
    gq = load_hex_block0(gq_p) if os.path.exists(gq_p) else None
    gs = load_hex_block0(gs_p) if os.path.exists(gs_p) else None
    if r["kind"] == "multi" or MODEL != "mamba2_prefill":
        vq, vs = voyager_golds(r)
        msg.append(f"golds: int8 == voyager tile: {gq is not None and np.array_equal(gq, vq.reshape(-1))}, "
                   f"scale == voyager tile: {gs is not None and np.array_equal(gs, vs.reshape(-1))}")
    sq_p, ss_p = f"{rundir}/hw_output_mxint8_act.txt", f"{rundir}/hw_scale_packed_output.txt"
    if not os.path.exists(sq_p):
        msg.append("no RTL output saved; EXPLAINED BY THE HW MECHANISMS: NO")
        print(f"  {name} check: " + "; ".join(msg)); return 3
    sq = unpack(load_hex_block0(sq_p))[:q_model.size]
    ss = unpack(load_hex_block0(ss_p))[:s_model.size] if os.path.exists(ss_p) else None
    msg.append(f"RTL == HW model: int8 {100 * np.mean(sq == q_model):.3f}% of {q_model.size}, "
               f"scale {'-' if ss is None else f'{100 * np.mean(ss == s_model):.3f}%'} of {s_model.size}")
    # the two-pass RTL outputs on the overlap
    if r["kind"] == "single":
        od, os_ = f"{r['old_data']}/hw_output_mxint8_act.txt", f"{r['old_data']}/hw_scale_packed_output.txt"
        oscale = (lambda p: unpack(load_hex_block0(p))) if os.path.exists(os_) else None
        if not os.path.exists(os_):
            os_ = f"{r['old_scale']}/hw_scale_output.txt"
            oscale = load_hex_block0
    else:
        od, os_ = f"{r['old_data']}/hw_output_mxint8_act.txt", f"{r['old_scale']}/hw_output.txt"
        oscale = lambda p: unpack(load_hex_block0(p))
    if os.path.exists(od):
        old = old_overlap(r, unpack(load_hex_block0(od)), "data")
        if old is not None:
            line = f"RTL int8 == two-pass RTL int8 on the overlap: {100 * np.mean(sq == old):.3f}%"
            if r["kind"] == "multi" and ss is not None:
                # the two-pass multi_IOs was fed voyager's scale: compare only blocks where the HW scale == voyager's
                vq, vs = voyager_golds(r)
                rows, cols, _, _ = tile_rc(r)
                same = np.repeat((ss.reshape(rows // 64, cols) == vs), 64, axis=0).reshape(-1)
                line += (f" (blocks with HW scale == voyager scale: {100 * np.mean(sq[same] == old[same]):.3f}% of {int(same.sum())}; "
                         f"{100 * (1 - same.mean()):.2f}% of values in blocks whose scales differ)")
            msg.append(line)
    if os.path.exists(os_) and ss is not None:
        old = old_overlap(r, oscale(os_), "scale")
        if old is not None:
            msg.append(f"RTL scale == two-pass RTL scale on the overlap: {100 * np.mean(ss == old):.3f}%")
    if gq is not None:
        v = verdict(sq, gq[:sq.size])
        msg.append(f"vs voyager (test.py view): int8 {v[0]} ({v[1]} off by >1, {100 * v[2]:.2f}% off by 1; signed {signed_hard(sq, gq[:sq.size])} off by >1)")
    if gs is not None and ss is not None:
        v = verdict(ss, gs[:ss.size])
        msg.append(f"scale {v[0]} ({v[1]} off by >1, {100 * v[2]:.2f}% off by 1)")
    # the user's acceptance rule for the quantize layers: not a test.py PASS, but RTL explained by the 4 HW mechanisms
    inputs_ok = a is not None and b is not None and np.array_equal(a, want) and np.array_equal(b, want)
    model_ok = ss is not None and np.array_equal(sq, q_model) and np.array_equal(ss, s_model)
    unexplained = None
    if gq is not None and gs is not None and ss is not None and gq.size >= sq.size and gs.size >= ss.size:
        _, unexplained, text = explain(x, 1 if r["kind"] == "single" else 0, sq, ss, gq[:sq.size], gs[:ss.size])
        msg.append(text)
    ok = inputs_ok and model_ok and unexplained == 0
    msg.append(f"EXPLAINED BY THE HW MECHANISMS: {'YES' if ok else 'NO'} (inputs {inputs_ok}, RTL == HW model {model_ok}, unexplained {unexplained})")
    print(f"  {name} check: " + "; ".join(msg))
    return 0 if ok else 3


# ---------------------------------------------------------------- report
strip = lambda s: re.sub(r"\x1b\[[0-9;]*m", "", s)
readf = lambda p: strip(open(p, errors="replace").read()) if os.path.exists(p) else ""


def report(logdir, out_md):
    st = readf(f"{logdir}/quantize_fused.status")
    lines = ["| Run | App | Layer | Tile (rows x cols) | test.py exit; explained | test.py comparison | RTL cycles | Ideal (values / 32) | Utilization | Staged |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    details = []
    for r in runs():
        rows, cols, _, _ = tile_rc(r)
        ideal = rows * cols // 32
        m = re.search(rf"^{r['name']} EXIT (\d+)(?: It takes ([0-9.]+) ns)?", st, re.M)
        lg = readf(f"{logdir}/{r['name']}.log")
        comp = [l.strip() for l in lg.splitlines() if re.search(r"Integer.*(passed|failed|tolerance)", l) and "rv_comparison" not in l]
        cyc = int(float(m.group(2))) if m and m.group(2) else None
        res = "NOT RUN" if not m else ("PASS" if m.group(1) == "0" else f"FAIL (exit {m.group(1)})")
        staged = "yes" if re.search(rf"STAGED {MODEL}-{r['layer']},,,{r['key']}_RV_E64_MB", st) else "no"
        ex = re.search(rf"^\s*{r['name']} check:.*EXPLAINED BY THE HW MECHANISMS: (YES|NO)", st, re.M)
        res += f"; explained: {ex.group(1) if ex else '-'}"
        lines.append(f"| {r['name']} | `{r['app']}` | `{r['layer']}` | {rows} x {cols} | {res} | {'<br>'.join(comp[-3:]) or '-'} | "
                     f"{cyc or '-'} | {ideal} | {f'{100 * ideal / cyc:.1f}%' if cyc else '-'} | {staged} |")
        chk = [l.strip() for l in st.splitlines() if l.strip().startswith(f"{r['name']} check:")]
        details.append(f"### {r['name']} (`{r['key']}`)\n" + "\n".join(f"- {x}" for x in chk) + "\n")
    text = "\n".join(lines + ["", "## Per-run checks", ""] + details) + "\n"
    open(out_md, "w").write(text)
    print(text)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for r in runs():
            print(r["name"], r["layer"], r["key"], r["app"])
    elif cmd == "host-check":
        host_check()
    elif cmd == "check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
    elif cmd == "report":
        report(sys.argv[2], sys.argv[3])
