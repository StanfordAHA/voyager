#!/bin/env xonsh

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="",
        help="Directory to create the build. Default is used if omitted",
    )
    parser.add_argument("--technology", type=str, default=None, required=True)
    parser.add_argument("--datatype", type=str, default=None, required=True)
    parser.add_argument(
        "--clock", type=float, default=None, help="Clock Period in ns", required=True
    )

    args = parser.parse_args()

    build_params = {
        "datatype": args.datatype,
        "clock_period": args.clock,
        "technology": args.technology,
    }

    with open(f"{os.path.dirname(__file__)}/params.py", "w") as f:
        f.write(f"build_params = {build_params}\n")

    dirname = args.dir if args.dir else "build-{datatype}-{clock_period}ns-{technology}".format(**build_params)

    print(f"Creating build in {dirname}...")
    mkdir @(dirname)

    cd @(dirname)
    mflowgen run --design ../design