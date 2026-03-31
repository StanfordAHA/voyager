import math
import os
import struct
import numpy as np

import torch
import torch.nn.functional as F


import argparse
import json

from parse_dnnLayer_tensors import float32_to_bfloat16_bits
from parse_dnnLayer_tensors import write_list_to_hex
from parse_dnnLayer_tensors import read_tensor


def verify_bert_gelu():
    # Read input tensor
    input_path = "/aha/voyager/test/compiler/networks/bert/MXINT8/tensor_files/linear_mx_default_4.bin"
    input_shape = (128, 3072)
    input_tensor = read_tensor(input_path, input_shape)

    # Apply GELU on input_tensor using sigmoid approximation
    output_bf16_approx = torch.zeros_like(input_tensor, dtype=torch.bfloat16)
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            x = input_tensor[i, j].item()
            sigmoid_approx = x * (1 / (1 + math.exp(-1.702 * x)))
            output_bf16_approx[i, j] = torch.tensor(sigmoid_approx, dtype=torch.bfloat16)

    # Convert output to float32 so can use float32_to_bfloat16_bits
    output_bf16_approx = output_bf16_approx.to(torch.float32)
    output_bf16_approx = float32_to_bfloat16_bits(output_bf16_approx)
    output_bf16_approx = output_bf16_approx.flatten().tolist()

    # Write output tensor to hex file
    write_list_to_hex(output_bf16_approx, '/aha/voyager/gold_activation.txt', "", add_metadata=False)


def verify_llama_prefill_silu():
    # Read input tensor
    input_path = "/aha/voyager/test/compiler/networks/llama/prefill/MXINT8/tensor_files/linear_mx_default_4.bin"
    input_shape = (512, 8192)
    input_tensor = read_tensor(input_path, input_shape)

    # Apply SiLU on input_tensor using sigmoid approximation
    output_bf16_approx = torch.zeros_like(input_tensor, dtype=torch.bfloat16)
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            x = input_tensor[i, j].item()
            sigmoid_approx = x * (1 / (1 + math.exp(-1* x)))
            output_bf16_approx[i, j] = torch.tensor(sigmoid_approx, dtype=torch.bfloat16)

    # Convert output to float32 so can use float32_to_bfloat16_bits
    output_bf16_approx = output_bf16_approx.to(torch.float32)
    output_bf16_approx = float32_to_bfloat16_bits(output_bf16_approx)
    output_bf16_approx = output_bf16_approx.flatten().tolist()

    # Write output tensor to hex file
    write_list_to_hex(output_bf16_approx, '/aha/voyager/gold_activation.txt', "", add_metadata=False)


def verify_bert_tanh():
    # Read gold tensor
    gold_path = "/aha/voyager/test/compiler/networks/bert/MXINT8/tensor_files/tanh.bin"
    gold_shape = (1, 768)
    gold_tensor = read_tensor(gold_path, gold_shape)

    gold_tensor_bf16 = float32_to_bfloat16_bits(gold_tensor)
    gold_tensor_bf16 = gold_tensor_bf16.flatten().tolist()

    # Write output tensor to hex file
    write_list_to_hex(gold_tensor_bf16, '/aha/voyager/gold_activation.txt', "", add_metadata=False)

def verify_resnet18_submodule():
    # Read gold tensor
    gold_path = "/aha/voyager/test/compiler/networks/resnet18/MXINT8/tensor_files/submodule_1.bin"
    gold_shape = (1, 12544, 64)
    gold_tensor = read_tensor(gold_path, gold_shape)

    x_dim_host_tiling = "ZIRCON_GEMM_X_DIM_HOST_TILING" in os.environ and os.environ["ZIRCON_GEMM_X_DIM_HOST_TILING"] == "1"

    if x_dim_host_tiling:
        if "X_DIM_HOST_TILING_SLICE_LENGTH" in os.environ:
            x_dim_host_tiling_slice_length = int(os.environ.get("X_DIM_HOST_TILING_SLICE_LENGTH"))

        if "X_DIM_HOST_TILING_SLICE_OFFSET" in os.environ:
            x_dim_host_tiling_slice_offset = int(os.environ.get("X_DIM_HOST_TILING_SLICE_OFFSET"))

        if "NUM_X_HOST_TILING_KERNELS" in os.environ:
            num_x_host_tiling_kernels = int(os.environ.get("NUM_X_HOST_TILING_KERNELS"))

        if "X_DIM_HOST_TILING_KERNEL_IDX" in os.environ:
            x_dim_host_tiling_kernel_idx = int(os.environ.get("X_DIM_HOST_TILING_KERNEL_IDX"))

        assert (x_dim_host_tiling_slice_length is not None and x_dim_host_tiling_slice_offset is not None) or (num_x_host_tiling_kernels is not None and x_dim_host_tiling_kernel_idx is not None), "Either X_DIM_HOST_TILING_SLICE_LENGTH and X_DIM_HOST_TILING_SLICE_OFFSET or NUM_X_HOST_TILING_KERNELS and X_DIM_HOST_TILING_KERNEL_IDX environment variables must be set for ZIRCON_GEMM_X_DIM_HOST_TILING"

        if len(gold_tensor.shape) == 3:
            if x_dim_host_tiling_slice_offset is not None and x_dim_host_tiling_slice_length is not None:
                gold_tensor = gold_tensor[:, x_dim_host_tiling_slice_offset:x_dim_host_tiling_slice_offset + x_dim_host_tiling_slice_length, :]
            else:
                gold_tensor = gold_tensor.reshape((gold_tensor.shape[0], num_x_host_tiling_kernels, gold_tensor.shape[1] // num_x_host_tiling_kernels, gold_tensor.shape[2]))
                gold_tensor = gold_tensor.permute(1, 0, 2, 3)
                gold_tensor = gold_tensor[x_dim_host_tiling_kernel_idx]
        # Error out becuase it's not supported yet
        else:
            raise NotImplementedError("X dimension host tiling is not supported for non 3-D tensors yet.")

    gold_tensor_bf16 = float32_to_bfloat16_bits(gold_tensor)
    gold_tensor_bf16 = gold_tensor_bf16.flatten().tolist()

    # Write output tensor to hex file
    write_list_to_hex(gold_tensor_bf16, '/aha/voyager/gold_activation.txt', "", add_metadata=False)

def verify_fakegemm_linear():
    # Read gold tensor
    gold_path = "/aha/voyager/test/compiler/networks/fakegemm/MXINT8/tensor_files/dequantize_default.bin"
    gold_shape = (3364, 32)
    gold_tensor = read_tensor(gold_path, gold_shape)

    gold_tensor_bf16 = float32_to_bfloat16_bits(gold_tensor)
    gold_tensor_bf16 = gold_tensor_bf16.flatten().tolist()

    # Write output tensor to hex file
    write_list_to_hex(gold_tensor_bf16, '/aha/voyager/gold_activation.txt', "", add_metadata=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom validation scripts for AHA Voyager flows.")
    # Add model and layer args
    parser.add_argument("--model", required=True, help="Model name (e.g., bert, resnet18)")
    parser.add_argument("--layer", required=True, help="Layer name (e.g., linear_mx_default_4)")

    args = parser.parse_args()
    model = args.model
    layer = args.layer

    if model == "bert" and (layer == "linear_mx_default_4" or layer == "gelu"):
        verify_bert_gelu()
    elif model == "llama_prefill" and (layer == "linear_mx_default_4" or layer == "silu"):
        verify_llama_prefill_silu()
    elif model == "bert" and layer == "tanh":
        verify_bert_tanh()
    elif model == "fakegemm" and layer == "linear_default_1":
        verify_fakegemm_linear()
    elif model == "resnet18" and layer == "submodule":
        verify_resnet18_submodule()
    else:
        raise NotImplementedError(f"No custom validation script for model {model} layer {layer}")

