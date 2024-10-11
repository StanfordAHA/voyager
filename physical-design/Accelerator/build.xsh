#!/bin/env xonsh

import argparse
import os
import sys

build_params = {
    "datatype": "INT8",
    "ic_dimension": 16,
    "oc_dimension": 16,
    "clock_period": 1,
    "input_buffer_size": 1024,
    "weight_buffer_size": 1024,
    "accum_buffer_size": 1024,
    "technology": "intel16",
}

sim_params = {
    "waveform": True,
    "sims": "accelerator,systemc",
    "network": "resnet18",  # default network
    "layer": "quantize_symmetric",  # default layer
}

sweep_params = {
    # mflowgen param sweep function is limited, use a script to sweep hardware params instead.
    # "datatypes": ["E4M3", "BF16", "P8_1"],
    # "dimensions": [[16, 16], [16, 32], [32, 32]],
    # "dimensions": [16, 32],
    "tests": {
    #     "resnet18": [
    #         "quantize_symmetric",
    #         "submodule_0",
    #         "max_pool2d",
    #         "quantize_symmetric_1",
    #         "submodule_1",
    #         "submodule_2",
    #         "quantize_symmetric_3",
    #         "submodule_3",
    #         "submodule_4",
    #         "submodule_5",
    #         "conv2d_6",
    #         "submodule_6",
    #         "quantize_symmetric_7",
    #         "submodule_7",
    #         "submodule_8",
    #         "submodule_9",
    #         "conv2d_11",
    #         "submodule_10",
    #         "quantize_symmetric_11",
    #         "submodule_11",
    #         "submodule_12",
    #         "submodule_13",
    #         "conv2d_16",
    #         "submodule_14",
    #         "quantize_symmetric_15",
    #         "submodule_15",
    #         "submodule_16",
    #         "adaptive_avg_pool2d",
    #         "linear",
    #     ],
    #     "mobilebert_encoder": [
    #         "quantize_symmetric",
    #         "submodule_0",
    #         "submodule_1",
    #         "quantize_symmetric_1",
    #         "submodule_2",
    #         "submodule_3",
    #         "submodule_4",
    #         "submodule_15",
    #         "softmax_1",
    #         "quantize_symmetric_21",
    #         "matmul_6",
    #         "submodule_16",
    #         "softmax_2",
    #         "quantize_symmetric_22",
    #         "matmul_7",
    #         "submodule_17",
    #         "softmax_3",
    #         "quantize_symmetric_23",
    #         "matmul_8",
    #         "submodule_18",
    #         "softmax_4",
    #         "quantize_symmetric_24",
    #         "matmul_9",
    #         "submodule_20",
    #         "submodule_5",
    #         "add_4",
    #         "quantize_symmetric_7",
    #         "submodule_6",
    #         "submodule_7",
    #         "add_6",
    #         "quantize_symmetric_9",
    #         "submodule_8",
    #         "submodule_9",
    #         "add_8",
    #         "quantize_symmetric_11",
    #         "submodule_10",
    #         "submodule_11",
    #         "add_10",
    #         "quantize_symmetric_13",
    #         "submodule_12",
    #         "quantize_symmetric_14",
    #         "submodule_13",
    #         "submodule_19",
    #         "submodule_14",
    #         "add_14",
    #     ],
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="", help="Directory to create the build. Default is used if omitted")
    parser.add_argument("--datatype", type=str, default=None)
    parser.add_argument("--ic", type=int, default=None, help="IC dimension")
    parser.add_argument("--oc", type=int, default=None, help="OC dimension")
    parser.add_argument("--input", type=int, default=None, help="Input Buffer Size")
    parser.add_argument("--weight", type=int, default=None, help="Weight Buffer Size")
    parser.add_argument("--accum", type=int, default=None, help="Accum Buffer Size")
    parser.add_argument("--clock_period", type=float, default=None)

    args = parser.parse_args()

    # mflowgen doesn't support passing command line arugments as paramters
    # workaround by writing out the param.py file
    print(f"Writing params...")
    # Modify params based on command line arguments
    build_params["datatype"] = args.datatype if args.datatype else build_params["datatype"]
    build_params["ic_dimension"] = args.ic if args.ic else build_params["ic_dimension"]
    build_params["oc_dimension"] = args.oc if args.oc else build_params["oc_dimension"]
    build_params["clock_period"] = args.clock_period if args.clock_period else build_params["clock_period"]
    build_params["input_buffer_size"] = args.input if args.input else build_params["input_buffer_size"]
    build_params["weight_buffer_size"] = args.weight if args.weight else build_params["weight_buffer_size"]
    build_params["accum_buffer_size"] = args.accum if args.accum else build_params["accum_buffer_size"]

    # Read to get all layers
    # for network in ["resnet18", "mobilebert_encoder"]:
    for network in ["resnet18", "resnet50", "mobilebert_encoder"]:
        with open(f"{os.path.dirname(__file__)}/../accel-src/test/compiler/networks/{network}/{build_params['datatype']}/layers.txt") as f:
            sweep_params["tests"][network] = f.read().splitlines()

    with open(f"{os.path.dirname(__file__)}/params.py", "w") as f:
        f.write(f"build_params = {build_params}\n")
        f.write(f"sim_params = {sim_params}\n")
        f.write(f"sweep_params = {sweep_params}\n")

    dirname = args.dir if args.dir else f"build-{build_params['datatype']}-{build_params['ic_dimension']}x{build_params['oc_dimension']}-{build_params['input_buffer_size']}x{build_params['weight_buffer_size']}x{build_params['accum_buffer_size']}-{build_params['clock_period']}ns"
    print(f"Creating directory \"{dirname}\"...")
    mkdir @(dirname)

    print(f"Initialize mflowgen...")
    cd @(dirname)
    mflowgen run --design ../design

    # Modify clock period  for hls stage if it's 1ns
    # this assumes hls is step 6, unfortunately mflowgen doesn't support explicit step name in this command
    if build_params['clock_period'] == 1:
        mflowgen param update -k clock_period -v 1.1 -s 6
        print(f"Step 6 clock period updated. Please make sure it's the hls stage.")

