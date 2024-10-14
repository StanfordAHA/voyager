#!/bin/env xonsh

import argparse
import os
import sys
from pprint import pprint
from collect_layers import collect_layers

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
    "tests": {}
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
    parser.add_argument("--clock", type=float, default=None)

    args = parser.parse_args()

    # mflowgen doesn't support passing command line arugments as paramters
    # workaround by writing out the param.py file
    print(f"Writing params...")
    # Modify params based on command line arguments
    build_params["datatype"] = args.datatype if args.datatype else build_params["datatype"]
    build_params["ic_dimension"] = args.ic if args.ic else build_params["ic_dimension"]
    build_params["oc_dimension"] = args.oc if args.oc else build_params["oc_dimension"]
    build_params["clock_period"] = args.clock if args.clock else build_params["clock_period"]
    build_params["input_buffer_size"] = args.input if args.input else build_params["input_buffer_size"]
    build_params["weight_buffer_size"] = args.weight if args.weight else build_params["weight_buffer_size"]
    build_params["accum_buffer_size"] = args.accum if args.accum else build_params["accum_buffer_size"]

    # Get the unique layers for each network for parameter sweep
    print(f"Collecting layers...")
    networks = ["resnet18", "resnet50", "mobilebert_encoder"]
    for network in networks:
      # Make sure dataset is generated
      if not os.path.exists(f"../accel-src/test/compiler/networks/{network}/{build_params['datatype']}"):
          print(f"Generating dataset for {network} {build_params['datatype']}...")
          pushd ../accel-src/
          make network-proto NETWORK=@(network) DATATYPE=@(build_params['datatype'])
          popd

    layers = collect_layers(networks, build_params["datatype"])
    for network in networks:
        sweep_params["tests"][network] = list(layers[network]) # only the keys - layer names

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
    # this assumes hls is step 7, unfortunately mflowgen doesn't support explicit step name in this command
    if build_params['clock_period'] == 1:
        mflowgen param update -k clock_period -v 1.1 -s 7
        print(f"Step 7 clock period updated. Please make sure it's the hls stage.")

