import argparse
import os
import re

def adjust_gold_for_k_tiling(input_file, output_file, channel_size, total_num_kernels, kernel_idx, output_tensor_k_dim_tiling=False):
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f]

    result = []
    micro_kernel_size = channel_size // total_num_kernels

    # Safety check
    if channel_size % total_num_kernels != 0:
        raise ValueError("channel_size must be divisible by total_num_kernels")

    for i, val in enumerate(lines):
        # Position within current channel_size group
        group_pos = i % channel_size

        # Which micro-kernel is this line in?
        micro_kernel_idx = group_pos // micro_kernel_size

        # Keep only the desired kernel, zero out others
        if micro_kernel_idx == kernel_idx:
            result.append(val)
        elif not(output_tensor_k_dim_tiling):
            result.append("0000")

    with open(output_file, "w") as f:
        f.write("\n".join(result))

def adjust_gold_for_zircon_conv1(input_file, output_file, slice_offset, out_img, n_oc):
    tile_num_pixels = out_img * out_img * n_oc
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f]

    result = []
    startpoint = slice_offset * n_oc
    endpoint = startpoint + tile_num_pixels

    # Shrink the gold file to only include the specified slice
    for i, val in enumerate(lines):
        if startpoint <= i < endpoint:
            result.append(val)

    with open(output_file, "w") as f:
        f.write("\n".join(result))

def adjust_gold_for_bert_up_proj_gelu(input_file, output_file, gold_channel_trimming_workaround_num_kernels, gold_channel_trimming_workaround_kernel_idx):
    MU_OC0 = 32
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f]

    result = []

    gelu_gold_channel_start = gold_channel_trimming_workaround_kernel_idx * (MU_OC0 // gold_channel_trimming_workaround_num_kernels)
    gelu_gold_channel_end = gelu_gold_channel_start + (MU_OC0 // gold_channel_trimming_workaround_num_kernels)

    for i, val in enumerate(lines):
        is_in_gelu_gold_region = gelu_gold_channel_start <= (i % MU_OC0) < gelu_gold_channel_end

        # Keep only the desired channels
        if is_in_gelu_gold_region:
            result.append(val)

    with open(output_file, "w") as f:
        f.write("\n".join(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero out all micro-kernels except the selected one.")

    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output file")

    args = parser.parse_args()

    k_dim_host_tiling = "K_DIM_HOST_TILING" in os.environ and os.environ["K_DIM_HOST_TILING"] == "1"
    # This means the OUTPUT tensor is too large to fit in GLB. So we tile along K dim to fit it in GLB.
    # In regular k_dim_host_tiling, it is assumed that the OUTPUT tesnor can fit in the GLB. The tiling is due to input tensors being too large to fit in GLB.
    output_tensor_k_dim_tiling = "OUTPUT_TENSOR_K_DIM_TILING" in os.environ and os.environ["OUTPUT_TENSOR_K_DIM_TILING"] == "1"
    zircon_conv1_gold = "ZIRCON_CONV1_GOLD" in os.environ and os.environ["ZIRCON_CONV1_GOLD"] == "1"
    GOLD_CHANNEL_TRIMMING_WORKAROUND = "GOLD_CHANNEL_TRIMMING_WORKAROUND" in os.environ and os.environ["GOLD_CHANNEL_TRIMMING_WORKAROUND"] == "1"


    if k_dim_host_tiling:
        assert "NUM_K_HOST_TILING_KERNELS" in os.environ, "NUM_K_HOST_TILING_KERNELS environment variable must be set for K_DIM_HOST_TILING"
        assert "K_DIM_HOST_TILING_IDX" in os.environ, "K_DIM_HOST_TILING_IDX environment variable must be set for K_DIM_HOST_TILING"
        num_k_host_tiling_kernels = int(os.environ["NUM_K_HOST_TILING_KERNELS"])
        k_host_tiling_idx = int(os.environ["K_DIM_HOST_TILING_IDX"])

        assert "HALIDE_GEN_ARGS" in os.environ, "HALIDE_GEN_ARGS environment variable must be set for K_DIM_HOST_TILING"
        HALIDE_GEN_ARGS = os.environ["HALIDE_GEN_ARGS"]
        n_oc_match = re.search(r'n_oc=(\d+)', HALIDE_GEN_ARGS)
        assert n_oc_match, "No n_oc in HALIDE_GEN_ARGS!"
        n_oc = int(n_oc_match.group(1))


        adjust_gold_for_k_tiling(
            args.input,
            args.output,
            n_oc,
            num_k_host_tiling_kernels,
            k_host_tiling_idx,
            output_tensor_k_dim_tiling=output_tensor_k_dim_tiling
        )


    elif zircon_conv1_gold:
        assert "X_DIM_HOST_TILING_SLICE_OFFSET" in os.environ, "X_DIM_HOST_TILING_SLICE_OFFSET environment variable must be set for ZIRCON_CONV1_GOLD"

        assert "HALIDE_GEN_ARGS" in os.environ, "HALIDE_GEN_ARGS environment variable must be set for ZIRCON_CONV1_GOLD"
        HALIDE_GEN_ARGS = os.environ["HALIDE_GEN_ARGS"]
        n_oc_match = re.search(r'n_oc=(\d+)', HALIDE_GEN_ARGS)
        assert n_oc_match, "No n_oc in HALIDE_GEN_ARGS!"
        n_oc = int(n_oc_match.group(1))

        out_img = re.search(r'out_img=(\d+)', HALIDE_GEN_ARGS)
        assert out_img, "No out_img in HALIDE_GEN_ARGS!"
        out_img = int(out_img.group(1))

        x_dim_host_tiling_slice_offset = int(os.environ["X_DIM_HOST_TILING_SLICE_OFFSET"])

        adjust_gold_for_zircon_conv1(
            args.input,
            args.output,
            x_dim_host_tiling_slice_offset,
            out_img,
            n_oc
        )


    if GOLD_CHANNEL_TRIMMING_WORKAROUND:
        assert "GOLD_CHANNEL_TRIMMING_WORKAROUND_NUM_KERNELS" in os.environ, "GOLD_CHANNEL_TRIMMING_WORKAROUND_NUM_KERNELS environment variable must be set for GOLD_CHANNEL_TRIMMING_WORKAROUND"
        gold_channel_trimming_workaround_num_kernels = int(os.environ["GOLD_CHANNEL_TRIMMING_WORKAROUND_NUM_KERNELS"])
        assert "GOLD_CHANNEL_TRIMMING_WORKAROUND_KERNEL_IDX" in os.environ, "GOLD_CHANNEL_TRIMMING_WORKAROUND_KERNEL_IDX environment variable must be set for GOLD_CHANNEL_TRIMMING_WORKAROUND"
        gold_channel_trimming_workaround_kernel_idx = int(os.environ["GOLD_CHANNEL_TRIMMING_WORKAROUND_KERNEL_IDX"])

        adjust_gold_for_bert_up_proj_gelu(
            args.output,
            args.output,
            gold_channel_trimming_workaround_num_kernels,
            gold_channel_trimming_workaround_kernel_idx
        )
