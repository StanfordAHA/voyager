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


def verify_bert_tanh():
    # Read gold tensor
    gold_path = "/aha/voyager/test/compiler/networks/bert/MXINT8/tensor_files/tanh.bin"
    gold_shape = (1, 768)
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
    elif model == "bert" and layer == "tanh":
        verify_bert_tanh()
    else:
        raise NotImplementedError(f"No custom validation script for model {model} layer {layer}")

