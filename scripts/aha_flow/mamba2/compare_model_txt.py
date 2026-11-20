"""Compare two mamba2 model.txt files op by op: names, targets, tensor shapes and dtypes.

Usage: python compare_model_txt.py <old model.txt> <new model.txt>

Prints every tensor whose dtype changed (grouped by old -> new dtype and by the target of the op
that reads it), and any difference in op names, op targets or tensor shapes. Memory addresses are
not compared.
"""
import importlib.util, sys
from collections import Counter
from google.protobuf import text_format

spec = importlib.util.spec_from_file_location(
    "param_pb2", "/aha/voyager/quantized-training/src/quantized_training/codegen/param_pb2.py")
param_pb2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(param_pb2)


def load(path):
    model = param_pb2.Model()
    text_format.Parse(open(path).read(), model)
    ops, tensors = [], {}
    for t in list(model.inputs) + list(model.parameters):
        tensors[t.node] = (tuple(t.shape), t.dtype)
    for entry in model.ops:
        outs = [entry.output] if entry.HasField("output") else list(entry.outputs.tensors)
        for t in outs:
            tensors[t.node] = (tuple(t.shape), t.dtype)
        sub = [entry.op] if entry.WhichOneof("op_type") == "op" else list(entry.fused_op.op_list)
        for op in sub:
            args = {}
            for i, a in enumerate(op.args):
                if a.WhichOneof("arg_type") == "tensor":
                    args[f"arg{i}"] = (a.tensor.node, tuple(a.tensor.shape), a.tensor.dtype)
            for k, v in op.kwargs.items():
                if v.WhichOneof("arg_type") == "tensor":
                    args[k] = (v.tensor.node, tuple(v.tensor.shape), v.tensor.dtype)
            ops.append((op.name, op.target, args))
    return ops, tensors


old_ops, old_t = load(sys.argv[1])
new_ops, new_t = load(sys.argv[2])
print(f"ops: old {len(old_ops)}, new {len(new_ops)}; tensors: old {len(old_t)}, new {len(new_t)}")

if [o[:2] for o in old_ops] != [o[:2] for o in new_ops]:
    print("OP NAMES/TARGETS DIFFER")
    for a, b in zip(old_ops, new_ops):
        if a[:2] != b[:2]:
            print("  ", a[:2], "->", b[:2])
            break
else:
    print("op names and targets: identical")

if set(old_t) != set(new_t):
    print("TENSOR NAMES DIFFER:", sorted(set(old_t) ^ set(new_t))[:20])
shape_diffs = [n for n in old_t if n in new_t and old_t[n][0] != new_t[n][0]]
print(f"output tensor shape differences: {len(shape_diffs)}", shape_diffs[:10])
dtype_out = Counter((old_t[n][1], new_t[n][1]) for n in old_t if n in new_t and old_t[n][1] != new_t[n][1])
print("output tensor dtype changes:", dict(dtype_out))

arg_changes = Counter()
other_arg_diffs = []
for (name, target, a_old), (_, _, a_new) in zip(old_ops, new_ops):
    for k in a_old:
        if k not in a_new:
            other_arg_diffs.append((name, k, "missing"))
            continue
        (n0, s0, d0), (n1, s1, d1) = a_old[k], a_new[k]
        if n0 != n1 or s0 != s1:
            other_arg_diffs.append((name, k, a_old[k], a_new[k]))
        if d0 != d1:
            arg_changes[(target, k, d0, d1)] += 1
print("op-argument dtype changes (reader target, argument, old -> new): count")
for key, c in sorted(arg_changes.items()):
    print("  ", key, c)
print(f"op-argument node/shape differences: {len(other_arg_diffs)}", other_arg_diffs[:10])
