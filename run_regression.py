import argparse
import datetime
import multiprocessing as mp
import os
import subprocess
from collections import defaultdict
import pandas as pd
import re
import signal
import sys
from deepdiff import DeepDiff
import math

import functools
import operator

from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from quantized_training.codegen import param_pb2


def print_test_results(test_results, layers, output_folder):
    columns = ["Model", "Layer", "Status", "Runtime", "Ideal", "RuntimeType", "Count"]
    if len(test_results[0]) == 3:
        columns = columns[:3]

    # convert list of tuples to DataFrame
    df = pd.DataFrame(test_results, columns=columns)
    sorted_df = []

    # get models
    models = df["Model"].unique()

    for model in models:
        print("=" * 10 + f" {model} " + "=" * 10)

        model_df = df[df["Model"] == model]

        # sort according to order in layers
        model_df["Layer"] = pd.Categorical(model_df["Layer"], layers[model])
        model_df.sort_values("Layer", inplace=True)
        # turn categorial back to string
        model_df["Layer"] = model_df["Layer"].astype(str)
        sorted_df.append(model_df)

        passed = model_df[model_df["Status"] == True]
        failed = model_df[model_df["Status"] == False]

        print("Passed:")
        print(
            passed["Layer"].to_string(index=False) if not passed.empty else "None",
            flush=True,
        )
        print("Failed:")
        print(
            failed["Layer"].to_string(index=False) if not failed.empty else "None",
            flush=True,
        )

        if not failed.empty:
            print(f"\033[91mERROR: Some voyager tests failed in SystemC. Please see above. Check the regression_results folder for the log. \033[0m")
            # sys.exit(1)

        # if runtime column exists, print runtime of each layer
        if "Runtime" in model_df.columns:
            print("Runtime:")
            print(
                model_df[
                    ["Layer", "Runtime", "Ideal", "RuntimeType", "Count"]
                ].to_string(index=False),
                flush=True,
            )
            utilization = model_df["Ideal"].sum() / model_df["Runtime"].sum()
            matrix_runtime = model_df[model_df["RuntimeType"] == "matrix"][
                "Runtime"
            ].sum()
            matrix_ideal = model_df[model_df["RuntimeType"] == "matrix"]["Ideal"].sum()
            matri_utilization = matrix_ideal / matrix_runtime
            print(f"Utilization: {utilization:.3f}")
            print(f"Matrix Utilization: {matri_utilization:.3f}")

    # concatentate all sorted model DataFrames into a single DataFrame and save to pickle
    pd.concat(sorted_df).to_pickle(f"{output_folder}/test_results.pkl")

    # return True if all tests passed
    return len(df[df["Status"] == False]) == 0


def check_environment_vars(required_vars):
    unset_vars = [var for var in required_vars if var not in os.environ]
    if len(unset_vars) > 0:
        raise ValueError(f"Please set {', '.join(unset_vars)} environment variables")


def run_gold_model_unit_test(model, layer, output_folder):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["CLOCK_PERIOD"] = "1"
    env_vars["SIMS"] = "gold,pytorch"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            subprocess.run(
                ["make", "sim"],
                env=env_vars,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=5 * 60,
            )
        except subprocess.TimeoutExpired:
            print(f"Test {model}_{layer} timed out")
            stdout_file.write("Test timed out")

    # search if the test passed
    p = subprocess.Popen(
        ["grep", "Error count: 0", f"{output_folder}/{model}_{layer}.log"],
        stdout=subprocess.PIPE,
    )
    p.communicate()

    return (model, layer, p.returncode == 0)


def run_gold_model_tests(layers, num_processes, results_folder):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])

    # Build TestRunner binary
    # subprocess.run(["make", "clean"], env=env_vars)

    with open(f"{results_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "TestRunner"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    def signal_handler(signum, frame):
        print(f"Receive signal {signum}, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_gold_model_unit_test,
                args=(model, test, results_folder),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers, results_folder)


def run_systemc_unit_test(model, layer, output_folder, fast, scale_down_operation):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["CLOCK_PERIOD"] = "1"
    env_vars["SIMS"] = "gold,accelerator"
    env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"

    if scale_down_operation:
        env_vars["SCALE_DOWN_OPERATION"] = "1"

    with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
        try:
            process = subprocess.Popen(
                ["make", "fast-sim" if fast else "sim"],
                env=env_vars,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in process.stdout:
                print(line, end="")        # print to terminal
                stdout_file.write(line)    # write to file

            process.wait(timeout=1 * 60 * 60)

        except subprocess.TimeoutExpired:
            print(f"Test {model}_{layer} timed out")
            stdout_file.write("Test timed out\n")
            process.kill()

    # search if the test passed
    p = subprocess.Popen(
        ["grep", "Error count: 0", f"{output_folder}/{model}_{layer}.log"],
        stdout=subprocess.PIPE,
    )
    p.communicate()

    return (model, layer, p.returncode == 0)


def run_systemc_tests(layers, condensed_models, num_processes, results_folder, fast):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])
    env_vars = os.environ.copy()
    env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"

    # Build TestRunner binary
    subprocess.run(["make", "clean"], env=os.environ)

    with open(f"{results_folder}/build.log", "w") as stdout_file:
        process = subprocess.Popen(
            ["make", "-j", "TestRunner-fast" if fast else "TestRunner"],
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in process.stdout:
            print(line, end="")        # show on terminal
            stdout_file.write(line)    # also save to log

        process.wait()

    pool = mp.Pool(num_processes)

    def signal_handler(signum, frame):
        print(f"Receive signal {signum}, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_systemc_unit_test,
                args=(
                    model,
                    test,
                    results_folder,
                    fast,
                    model in condensed_models if condensed_models else False,
                ),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers, results_folder)


def run_rtl_test(model, layer, layer_count, output_folder, scale_down_operation):
    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model
    env_vars["TESTS"] = layer
    env_vars["SIMS"] = "gold,accelerator"

    if scale_down_operation:
        env_vars["SCALE_DOWN_OPERATION"] = "1"

    # Workaround: vcs/catapult don't support GLIBCXX_3.4.30 in their libstdc++, and the tools hardcode the linker libraries in such an
    # order that their libs are used over the user specified ones. We need the newer version in order to run dependencies installed from conda.
    env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"
    if "INPUT_BUFFER_SIZE" not in env_vars:
        env_vars["INPUT_BUFFER_SIZE"] = "1024"
    if "WEIGHT_BUFFER_SIZE" not in env_vars:
        env_vars["WEIGHT_BUFFER_SIZE"] = "1024"
    if "ACCUM_BUFFER_SIZE" not in env_vars:
        env_vars["ACCUM_BUFFER_SIZE"] = "1024"
    if "DOUBLE_BUFFERED_ACCUM_BUFFER" not in env_vars:
        env_vars["DOUBLE_BUFFERED_ACCUM_BUFFER"] = "false"

    # we occasionally see the test fail due to filesystem issues ("no rule to make target", but the target exists), so we retry up to 3 times
    for attempt in range(3):
        with open(f"{output_folder}/{model}_{layer}.log", "w") as stdout_file:
            try:
                subprocess.run(
                    ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "sim"],
                    cwd=f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}_{env_vars['INPUT_BUFFER_SIZE']}x{env_vars['WEIGHT_BUFFER_SIZE']}x{env_vars['ACCUM_BUFFER_SIZE']}_{env_vars['DOUBLE_BUFFERED_ACCUM_BUFFER']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
                    env=env_vars,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    timeout=8 * 60 * 60,
                )
            except subprocess.TimeoutExpired:
                print(f"Test {model}_{layer} timed out")
                stdout_file.write("Test timed out")
                break

        with open(f"{output_folder}/{model}_{layer}.log", "r") as logfile:
            text = logfile.read()
            if "No rule to make target" not in text:
                break

    # search if the test passed
    p = subprocess.Popen(
        ["grep", "Error count: 0", f"{output_folder}/{model}_{layer}.log"],
        stdout=subprocess.PIPE,
    )
    p.communicate()
    success = p.returncode == 0

    if success:
        # capture number after "Runtime: " in the log file
        p = subprocess.Popen(
            [
                "grep",
                "-oP",
                "(?<=Runtime: ).\d+",
                f"{output_folder}/{model}_{layer}.log",
            ],
            stdout=subprocess.PIPE,
        )
        runtime = int(p.communicate()[0].decode("utf-8").strip())

        # capture ideal runtime number in the log file
        # can either be "Matrix unit ideal runtime: " or "Vector unit ideal runtime: "
        p = subprocess.Popen(
            [
                "grep",
                "-oP",
                "(?<=matrix unit ideal runtime: ).\d+",
                f"{output_folder}/{model}_{layer}.log",
            ],
            stdout=subprocess.PIPE,
        )

        match = p.communicate()[0]

        if match:
            runtime_type = "matrix"
        else:
            p = subprocess.Popen(
                [
                    "grep",
                    "-oP",
                    "(?<=vector unit ideal runtime: ).\d+",
                    f"{output_folder}/{model}_{layer}.log",
                ],
                stdout=subprocess.PIPE,
            )
            match, error = p.communicate()
            assert not error

            runtime_type = "vector"

        ideal = int(match.decode("utf-8").strip())
    else:
        runtime = 0
        ideal = 0
        runtime_type = ""

    return (model, layer, success, runtime, ideal, runtime_type, layer_count)


def run_rtl_tests(
    layers,
    layer_counts,
    condensed_models,
    num_processes,
    results_folder,
    keep_build=False,
):
    check_environment_vars(
        ["DATATYPE", "IC_DIMENSION", "OC_DIMENSION", "TECHNOLOGY", "CLOCK_PERIOD"]
    )

    # clean old build
    if not keep_build:
        subprocess.run(["make", "clean-catapult"], env=os.environ)

    # generate RTL
    with open(f"{results_folder}/rtl_generation.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "rtl"],
            env=os.environ,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    model, (test, *_) = next(iter(layers.items()))
    print(f"Running {model} {test}")

    # build VCS simulation binary
    with open(f"{results_folder}/vcs_build.log", "w") as stdout_file:
        env_vars = os.environ.copy()
        env_vars["NETWORK"] = model
        env_vars["TESTS"] = test
        env_vars["SIMS"] = "gold,accelerator"
        env_vars["LD_PRELOAD"] = env_vars["CONDA_PREFIX"] + "/lib/libstdc++.so.6"

        if "INPUT_BUFFER_SIZE" not in env_vars:
            env_vars["INPUT_BUFFER_SIZE"] = "1024"
        if "WEIGHT_BUFFER_SIZE" not in env_vars:
            env_vars["WEIGHT_BUFFER_SIZE"] = "1024"
        if "ACCUM_BUFFER_SIZE" not in env_vars:
            env_vars["ACCUM_BUFFER_SIZE"] = "1024"
        if "DOUBLE_BUFFERED_ACCUM_BUFFER" not in env_vars:
            env_vars["DOUBLE_BUFFERED_ACCUM_BUFFER"] = "false"

        subprocess.run(
            ["make", "-f", "scverify/Verify_concat_sim_rtl_v_vcs.mk", "build"],
            cwd=f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}_{env_vars['INPUT_BUFFER_SIZE']}x{env_vars['WEIGHT_BUFFER_SIZE']}x{env_vars['ACCUM_BUFFER_SIZE']}_{env_vars['DOUBLE_BUFFERED_ACCUM_BUFFER']}/Catapult/{env_vars['TECHNOLOGY']}/clock_{env_vars['CLOCK_PERIOD']}/Accelerator/Accelerator.v1",
            env=env_vars,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    pool = mp.Pool(num_processes)

    def signal_handler(signum, frame):
        print(f"Receive signal {signum}, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    test_results = []

    for model, tests in layers.items():
        for test in tests:
            pool.apply_async(
                run_rtl_test,
                args=(
                    model,
                    test,
                    layer_counts[model][test],
                    results_folder,
                    model in condensed_models if condensed_models else False,
                ),
                callback=test_results.append,
            )

    pool.close()
    pool.join()

    return print_test_results(test_results, layers, results_folder)


ACCURACY_RESULTS = {
    "resnet18": {
        "E4M3": 70.8,
        "CFLOAT": 70.8,
        "INT8": 69.7,
        "MXINT8": 71.0,
        "P8_1": 69.5,
    },
    "resnet50": {
        "E4M3": 67.7,
        "CFLOAT": 71.2,
        "INT8": 69.1,
        "MXINT8": 69.5,
        "P8_1": 69.6,
    },
    "mobilebert": {
        "E4M3": 90.6,
        "CFLOAT": 90.83,
        "INT8": 90.37,
        "MXINT8": 91.0,
        "P8_1": 90.37,
    },
}


def run_accuracy(model, dataset, num_processes, output_folder):
    check_environment_vars(["DATATYPE", "IC_DIMENSION", "OC_DIMENSION"])

    if len(model) > 1:
        print(f"Only testing accuracy for the first model: {model[0]}")
    model = model[0]

    env_vars = os.environ.copy()
    env_vars["NETWORK"] = model

    if "INPUT_BUFFER_SIZE" not in env_vars:
        env_vars["INPUT_BUFFER_SIZE"] = "1024"
    if "WEIGHT_BUFFER_SIZE" not in env_vars:
        env_vars["WEIGHT_BUFFER_SIZE"] = "1024"
    if "ACCUM_BUFFER_SIZE" not in env_vars:
        env_vars["ACCUM_BUFFER_SIZE"] = "1024"
    if "DOUBLE_BUFFERED_ACCUM_BUFFER" not in env_vars:
        env_vars["DOUBLE_BUFFERED_ACCUM_BUFFER"] = "false"

    # Build AccuracyTester binary
    subprocess.run(["make", "clean"], env=env_vars)

    with open(f"{output_folder}/build.log", "w") as stdout_file:
        subprocess.run(
            ["make", "-j", "AccuracyTester"],
            env=env_vars,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    # Generate input samples from dataset
    if dataset == "imagenet":
        imagenet_path = "/sim2/shared/MINOTAUR/nn_data/imagenet_1000/data/"
        output_data_dir = "data/imagenet"
        subprocess.run(
            [
                "python",
                "test/script/dump_resnet_dataset.py",
                "--data_dir",
                imagenet_path,
                "--output_dir",
                output_data_dir,
                "--num_samples",
                "1000",
            ]
        )
    elif dataset == "sst2":
        output_data_dir = "data/sst2"
        subprocess.run(
            [
                "python",
                "test/script/dump_bert_dataset.py",
                "--dataset",
                "sst2",
                "--model_name_or_path",
                "models/mobilebert/mobilebert-tiny-sst2-bf16/",
                "--output_dir",
                output_data_dir,
            ]
        )
    elif dataset == "squad":
        output_data_dir = "data/squad"
        subprocess.run(
            [
                "python",
                "test/script/dump_bert_dataset.py",
                "--dataset",
                "squad",
                "--model_name_or_path",
                "models/mobilebert/mobilebert-tiny-squad-bf16/",
                "--output_dir",
                output_data_dir,
            ]
        )
    else:
        raise ValueError("Invalid dataset")

    # Dump model parameters
    if model == "resnet18":
        model_path = "models/resnet/resnet18_mp2_p8_qat.pth"
    elif model == "resnet50":
        model_path = "models/resnet/resnet50.pth"
    elif model == "mobilebert" and dataset == "sst2":
        model_path = "models/mobilebert/mobilebert-tiny-sst2-bf16/"
    elif model == "mobilebert" and dataset == "squad":
        model_path = "models/mobilebert/mobilebert-tiny-squad-bf16/"
    else:
        raise ValueError("Invalid model")

    block_size = max(os.environ["OC_DIMENSION"], os.environ["IC_DIMENSION"])

    if env_vars["DATATYPE"] == "E4M3":
        quantization_args = [
            "--activation",
            "fp8_e4m3",
            "--weight",
            "fp8_e4m3",
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "INT8":
        quantization_args = [
            "--activation",
            "int8,qs=per_tensor_symmetric",
            "--weight",
            "int8,qs=per_tensor_symmetric",
            "--bias",
            "int24",
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "P8_1":
        quantization_args = [
            "--activation",
            "posit8_1",
            "--weight",
            "posit8_1",
            "--bf16",
        ]
    elif env_vars["DATATYPE"] == "CFLOAT":
        quantization_args = []
    elif env_vars["DATATYPE"] == "MXINT8":
        quantization_args = [
            "--force_scale_power_of_two",
            "--activation",
            "int8,qs=microscaling,bs=" + block_size,
            "--weight",
            "int8,qs=microscaling,bs=" + block_size,
            "--bf16",
        ]
    else:
        raise ValueError("Invalid datatype")

    with open(f"{output_folder}/{model}_{dataset}_compiler.log", "w") as stdout_file:
        subprocess.run(
            [
                "python",
                "quantized-training/test/test_codegen.py",
                model,
                "--model_name_or_path",
                "--transpose_weight",
                model_path,
                *quantization_args,
                "--output_dir",
                "test/compiler/networks/" + model + "/" + env_vars["DATATYPE"],
            ],
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    with open(f"{output_folder}/{model}_{dataset}_tiler.log", "w") as stdout_file:
        subprocess.run(
            [
                "python",
                "test/compiler/run_tiler.py",
                "--codegen_dir",
                f"test/compiler/networks/{model}/{env_vars['DATATYPE']}",
                "--IC_dimension",
                env_vars["IC_DIMENSION"],
                "--OC_dimension",
                env_vars["OC_DIMENSION"],
                "--input_buffer_size",
                env_vars["INPUT_BUFFER_SIZE"],
                "--weight_buffer_size",
                env_vars["WEIGHT_BUFFER_SIZE"],
                "--accum_buffer_size",
                env_vars["ACCUM_BUFFER_SIZE"],
            ],
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
        )

    # Run accuracy test
    additional_args = []
    if dataset == "squad":
        additional_args = ["1000"]  # limit number of samples to 1000 for squad dataset
    with open(f"{output_folder}/{model}_{dataset}.log", "w") as stdout_file:
        try:
            subprocess.run(
                [
                    f"build/{env_vars['DATATYPE']}_{env_vars['IC_DIMENSION']}x{env_vars['OC_DIMENSION']}_{env_vars['INPUT_BUFFER_SIZE']}x{env_vars['WEIGHT_BUFFER_SIZE']}x{env_vars['ACCUM_BUFFER_SIZE']}_{env_vars['DOUBLE_BUFFERED_ACCUM_BUFFER']}/cc/AccuracyTester",
                    model,
                    output_data_dir,
                    str(num_processes),
                    *additional_args,
                ],
                env=env_vars,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                timeout=3 * 60 * 60,
            )
        except subprocess.TimeoutExpired:
            print(f"Test {model}_{dataset} timed out")
            stdout_file.write("Test timed out")
            return False

    # Extract accuracy from log file
    accuracy_regex = "Accuracy: \d+\/\d+ \((\d+\.+\d+)%\)"
    with open(f"{output_folder}/{model}_{dataset}.log", "r") as logfile:
        text = logfile.read()
    final_accuracy = float(re.findall(accuracy_regex, text)[-1])

    print(f"Final accuracy: {final_accuracy}%")

    # save results to dataframe
    df = pd.DataFrame(
        [(model, dataset, final_accuracy)], columns=["Model", "Dataset", "Accuracy"]
    )

    # dump dataframe to pickle
    df.to_pickle(f"{output_folder}/test_results.pkl")

    gold_accuracy = ACCURACY_RESULTS[model][env_vars["DATATYPE"]]
    return abs(final_accuracy - gold_accuracy) < 1


def add_layers(network, layers, layer_counts, uniquify):
    all_layers = []

    layers[network] = []
    layer_counts[network] = {}

    if not uniquify:
        with open(
            f"test/compiler/networks/{network}/{os.environ['DATATYPE']}/layers.txt",
            "r",
        ) as f:
            layers[network] = f.read().splitlines()
            layer_counts[network] = {layer: 1 for layer in layers[network]}
    else:
        # open the proto file
        with open(
            f"test/compiler/networks/{network}/{os.environ['DATATYPE']}/model.txt",
            "r",
        ) as f:
            contents = f.read()
        params = param_pb2.Model()
        text_format.Parse(contents, params)

        # convert to json
        params_dict = MessageToDict(params, preserving_proto_field_name=True)

        def delete_nested_keys(data, key):
            # if the current element is a dict
            if isinstance(data, dict):
                # use list() to create a copy, avoiding modifying the dictionary while iterating
                for k in list(data.keys()):
                    if k == key:
                        del data[k]
                    else:
                        # recursively check the value of the current key
                        delete_nested_keys(data[k], key)

            # if the current element is a list
            elif isinstance(data, list):
                for item in data:
                    delete_nested_keys(item, key)

        unique_layers = {}
        for op in params_dict["ops"]:
            # skip nop layers
            if "op" in op and op["op"]["op"] == "nop":
                continue

            name = op["op"]["name"] if "op" in op else op["fused_op"]["name"]

            # remove the name, memory, and node fields from the op
            delete_nested_keys(op, "name")
            delete_nested_keys(op, "memory")
            delete_nested_keys(op, "node")

            is_unique_layer = True
            for op_name, op_val in unique_layers.items():
                if not DeepDiff(op, op_val, ignore_order=True):
                    layer_counts[network][op_name] += 1
                    is_unique_layer = False
                    break

            if is_unique_layer:
                unique_layers[name] = op
                layers[network].append(name)
                layer_counts[network][name] = 1


def append_glb_base_addresses(tensor_metadata, kwargs, mu_glb_base_address, is_gemm=False):
    zircon_fx_fy_stride_workaround = "ZIRCON_FX_FY_STRIDE_WORKAROUND" in os.environ and os.environ["ZIRCON_FX_FY_STRIDE_WORKAROUND"] == "1"
    zircon_input_act_padding_workaround = "ZIRCON_INPUT_ACT_PADDING_WORKAROUND" in os.environ and os.environ["ZIRCON_INPUT_ACT_PADDING_WORKAROUND"] == "1"
    if zircon_input_act_padding_workaround:
        assert "ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE" in os.environ, "ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE environment variable must be set for ZIRCON_INPUT_ACT_PADDING_WORKAROUND"
        zircon_input_act_padding_workaround_size = int(os.environ.get("ZIRCON_INPUT_ACT_PADDING_WORKAROUND_SIZE", 0))
        assert "ZIRCON_INPUT_ACT_PADDING_WORKAROUND_STRIDE" in os.environ, "ZIRCON_INPUT_ACT_PADDING_WORKAROUND_STRIDE environment variable must be set for ZIRCON_INPUT_ACT_PADDING_WORKAROUND"
        zircon_input_act_padding_workaround_stride = int(os.environ.get("ZIRCON_INPUT_ACT_PADDING_WORKAROUND_STRIDE", 0))
        pad_dim = zircon_input_act_padding_workaround_size * zircon_input_act_padding_workaround_stride
    k_dim_host_tiling = "K_DIM_HOST_TILING" in os.environ and os.environ["K_DIM_HOST_TILING"] == "1"
    if k_dim_host_tiling:
        assert "NUM_K_HOST_TILING_KERNELS" in os.environ, "NUM_K_HOST_TILING_KERNELS environment variable must be set for K_DIM_HOST_TILING"
        num_k_host_tiling_kernels = int(os.environ.get("NUM_K_HOST_TILING_KERNELS", 1))

    zircon_gemm_reduction_tiling_workaround = "ZIRCON_GEMM_REDUCTION_TILING_WORKAROUND" in os.environ and os.environ["ZIRCON_GEMM_REDUCTION_TILING_WORKAROUND"] == "1"
    if zircon_gemm_reduction_tiling_workaround:
        assert "NUM_PSUMS" in os.environ, "NUM_PSUMS environment variable must be set for ZIRCON_GEMM_REDUCTION_TILING_WORKAROUND"
        num_psums = int(os.environ.get("NUM_PSUMS", 1))

    x_dim_host_tiling = "ZIRCON_GEMM_X_DIM_HOST_TILING" in os.environ and os.environ["ZIRCON_GEMM_X_DIM_HOST_TILING"] == "1"
    if x_dim_host_tiling:
        x_dim_host_tiling_slice_length = None
        num_x_host_tiling_kernels = None

        if "X_DIM_HOST_TILING_SLICE_LENGTH" in os.environ:
            x_dim_host_tiling_slice_length = int(os.environ.get("X_DIM_HOST_TILING_SLICE_LENGTH"))

        if "NUM_X_HOST_TILING_KERNELS" in os.environ:
            num_x_host_tiling_kernels = int(os.environ.get("NUM_X_HOST_TILING_KERNELS"))


        assert x_dim_host_tiling_slice_length is not None or num_x_host_tiling_kernels is not None, "Either X_DIM_HOST_TILING_SLICE_LENGTH or NUM_X_HOST_TILING_KERNELS environment variable must be set for ZIRCON_GEMM_X_DIM_HOST_TILING"

    # Append GLB base addresses to the kwargs for input, weight, bias, inputScale, and weightScale tensors
    input_base_address = mu_glb_base_address
    curr_addr_pointer = input_base_address

    # multiply all element in kwargs['input']['tensor']['shape'] together and add to input_base_address
    input_num_elements = functools.reduce(operator.mul, kwargs['input']['tensor']['shape'], 1)
    # TODO: Refine this
    if x_dim_host_tiling:
        if len(kwargs['input']['tensor']['shape']) == 3:
            if x_dim_host_tiling_slice_length is not None:
                input_num_elements = kwargs['input']['tensor']['shape'][0] * x_dim_host_tiling_slice_length * (kwargs['input']['tensor']['shape'][2])
            else:
                input_num_elements = kwargs['input']['tensor']['shape'][0] * (kwargs['input']['tensor']['shape'][1] // num_x_host_tiling_kernels) * (kwargs['input']['tensor']['shape'][2])
        # Error out because it's not supported yet
        else:
            raise NotImplementedError("X dimension host tiling is not supported for non 3-D tensors yet.")
    if zircon_input_act_padding_workaround:
        input_shape = kwargs['input']['tensor']['shape']
        input_num_elements = input_shape[0] * (input_shape[1] + pad_dim) * (input_shape[2] + pad_dim) * input_shape[3]
    if zircon_gemm_reduction_tiling_workaround:
        input_num_elements = input_num_elements // num_psums
    curr_addr_pointer = input_base_address + math.ceil(input_num_elements/32) * 32 # take math.ceil(/32) * 32 to align to 32 bytes in MU-GLB address space

    if 'input_scale' in kwargs and 'tensor' in kwargs['input_scale'] and 'shape' in kwargs['input_scale']['tensor']:
        inputScale_num_elements = functools.reduce(operator.mul, kwargs['input_scale']['tensor']['shape'], 1)

        if x_dim_host_tiling:
            if len(kwargs['input_scale']['tensor']['shape']) == 3:
                if x_dim_host_tiling_slice_length is not None:
                    inputScale_num_elements = kwargs['input_scale']['tensor']['shape'][0] * x_dim_host_tiling_slice_length * (kwargs['input_scale']['tensor']['shape'][2])
                else:
                    inputScale_num_elements = kwargs['input_scale']['tensor']['shape'][0] * (kwargs['input_scale']['tensor']['shape'][1] // num_x_host_tiling_kernels) * (kwargs['input_scale']['tensor']['shape'][2])
            # Error out because it's not supported yet
            else:
                raise NotImplementedError("X dimension host tiling is not supported for non 3-D tensors yet.")
        if zircon_input_act_padding_workaround:
            inputScale_shape = kwargs['input_scale']['tensor']['shape']
            inputScale_num_elements = inputScale_shape[0] * (inputScale_shape[1] + pad_dim) * (inputScale_shape[2] + pad_dim) * inputScale_shape[3]
        if zircon_gemm_reduction_tiling_workaround:
            inputScale_num_elements = inputScale_num_elements // num_psums
        inputScale_base_address = curr_addr_pointer
        curr_addr_pointer += math.ceil(inputScale_num_elements/32) * 32 # take math.ceil(/32) * 32 to align to 32 bytes in MU-GLB address space


    if is_gemm:
        if 'other' in kwargs and not 'weight' in kwargs:
            kwargs['weight'] = kwargs['other']  # rename 'other' to 'weight' for consistency
            del kwargs['other']  # remove 'other' from kwargs
    if 'weight' in kwargs:
        weight_num_elements = functools.reduce(operator.mul, kwargs['weight']['tensor']['shape'], 1)
        if k_dim_host_tiling:
            weight_num_elements = weight_num_elements // num_k_host_tiling_kernels
        if zircon_gemm_reduction_tiling_workaround:
            weight_num_elements = weight_num_elements // num_psums
        if zircon_fx_fy_stride_workaround:
            weight_num_elements *= 3 * 3
        weight_base_address = curr_addr_pointer
        curr_addr_pointer += math.ceil(weight_num_elements/32) * 32 # take math.ceil(/32) * 32 to align to 32 bytes in MU-GLB address space

    if 'weight_scale' in kwargs and 'tensor' in kwargs['weight_scale'] and 'shape' in kwargs['weight_scale']['tensor']:
        weightScale_num_elements = functools.reduce(operator.mul, kwargs['weight_scale']['tensor']['shape'], 1)
        if zircon_gemm_reduction_tiling_workaround:
            weightScale_num_elements = weightScale_num_elements // num_psums
        if k_dim_host_tiling:
            weightScale_num_elements = weightScale_num_elements // num_k_host_tiling_kernels
        if zircon_fx_fy_stride_workaround:
            weightScale_num_elements *= 3 * 3
        weightScale_base_address = curr_addr_pointer
        curr_addr_pointer += math.ceil(weightScale_num_elements/32) * 32 # take math.ceil(/32) * 32 to align to 32 bytes in MU-GLB address space

    if 'bias' in kwargs:
        bias_base_address = curr_addr_pointer

    if 'input' in kwargs:
        kwargs['input']['tensor']['glb_base_address'] = input_base_address
        tensor_metadata["has_input"] = True
    if 'input_scale' in kwargs:
        kwargs['input_scale']['tensor']['glb_base_address'] = inputScale_base_address
        tensor_metadata["has_input_scale"] = True
    # Handle CGRA-only layers with scale inputs
    if 'scale' in kwargs:
        num_elements = functools.reduce(operator.mul, kwargs['scale']['tensor']['shape'], 1)
        if num_elements > 1:
            tensor_metadata["has_input_scale"] = True
    if 'weight' in kwargs:
        kwargs['weight']['tensor']['glb_base_address'] = weight_base_address
        tensor_metadata["has_weight"] = True
    if 'weight_scale' in kwargs:
        kwargs['weight_scale']['tensor']['glb_base_address'] = weightScale_base_address
        tensor_metadata["has_weight_scale"] = True
    if 'bias' in kwargs:
        kwargs['bias']['tensor']['glb_base_address'] = bias_base_address
        tensor_metadata["has_bias"] = True


def create_tensor_metadata_json(layer, params_dict):
    # create tensor_metadata.json file, needed by aha flow
    tensor_metadata = {
        "layer_name": layer,
        "has_input": True,
        "has_input_scale": False,
        "has_weight": False,
        "has_weight_scale": False,
        "has_bias": False,
        "has_residual": False,
        "mu_glb_base_address": 0,
        "ops": [],
        "outputs": [],
    }

    # All tensors (input, weight, bias, inputScale, weightScale) are stored in the GLB memory contiguously from this base address.
    mu_glb_base_address = 0
    if "MU_GLB_BASE_ADDR" in os.environ:
        mu_glb_base_address = int(os.environ["MU_GLB_BASE_ADDR"])

    is_standalone_cgra_app = "VOYAGER_STANDALONE_CGRA_APP" in os.environ and os.environ["VOYAGER_STANDALONE_CGRA_APP"] == "1"
    tensor_metadata["mu_glb_base_address"] = mu_glb_base_address

    match = False
    for op in params_dict["ops"]:
        if 'op' in op and op["op"]["name"] == layer:
            match = True
            op_dict = {}
            op_dict["name"] = op["op"]["name"]
            op_dict["kwargs"] = op["op"]["kwargs"]
            # Should be if "conv2d" or if "matmul", etc. Generalize this in the future
            if "conv2d" in op_dict["name"] or "matmul" in op_dict["name"] or "linear" in op_dict["name"] or "quantize" in op_dict["name"] or "layer_norm" in op_dict["name"]:
                is_gemm = "matmul" in op_dict["name"] or ("linear" in op_dict["name"] and not(is_standalone_cgra_app))
                append_glb_base_addresses(tensor_metadata, op_dict["kwargs"], mu_glb_base_address, is_gemm=is_gemm)
            for arg_key in op_dict["kwargs"]:
                    arg = op_dict["kwargs"][arg_key]
                    tensor = arg.get("tensor")
                    if isinstance(tensor, dict):
                        if "memory" in tensor:
                            del tensor["memory"]  # Remove memory field from voyager compiler (mapping to GLB memory is different and handled in the aha flow)
            tensor_metadata["ops"].append(op_dict)

        elif 'fused_op' in op and op["fused_op"]["name"] == layer:
            match = True
            for fused_op in op["fused_op"]["op_list"]:
                fused_op_dict = {}
                fused_op_name = fused_op["name"]
                fused_op_dict["name"] = fused_op_name
                if "add" in fused_op_name:
                    tensor_metadata["has_residual"] = True
                fused_op_dict["kwargs"] = fused_op["kwargs"]
                # Should be if "conv2d" or if "matmul", etc. Generalize this in the future
                if "conv2d" in fused_op_dict["name"] or "matmul" in fused_op_dict["name"] or "linear" in fused_op_dict["name"] or "quantize" in fused_op_dict["name"] or "layer_norm" in fused_op_dict["name"]:
                    is_gemm = "matmul" in fused_op_dict["name"] or ("linear" in fused_op_dict["name"] and not(is_standalone_cgra_app))
                    append_glb_base_addresses(tensor_metadata, fused_op_dict["kwargs"], mu_glb_base_address, is_gemm=is_gemm)
                for arg_key in fused_op_dict["kwargs"]:
                    arg = fused_op_dict["kwargs"][arg_key]
                    tensor = arg.get("tensor")
                    if isinstance(tensor, dict):
                        if "memory" in tensor:
                            del tensor["memory"]  # Remove memory field from voyager compiler (mapping to GLB memory is different and handled in the aha flow)

                tensor_metadata["ops"].append(fused_op_dict)

        if match:
            if 'outputs' in op:
                tensor_metadata["outputs"].append(op["outputs"])
            elif 'output' in op:
                tensor_metadata["outputs"].append(op["output"])

            for output in tensor_metadata["outputs"]:
                # Remove memory field from voyager compiler (mapping to GLB memory is different and handled in the aha flow
                if isinstance(output, dict):
                    output.pop("memory", None)
            # if we found the layer, we can stop searching
            break


    conv1_bias_hack = "CONV1_BIAS_HACK" in os.environ and os.environ["CONV1_BIAS_HACK"] == "1"
    if conv1_bias_hack:
        tensor_metadata["ops"][0]["kwargs"]["bias"] = tensor_metadata["ops"][1]["kwargs"]["other"]
        del tensor_metadata["ops"][1]
        tensor_metadata["has_residual"] = False
        append_glb_base_addresses(tensor_metadata, tensor_metadata["ops"][0]["kwargs"], mu_glb_base_address)

    with open(f"tensor_metadata.json", "w") as f:
        import json

        json.dump(tensor_metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        required=True,
        help="Model(s) to test for regression (resnet18, mobilebert)",
    )
    parser.add_argument(
        "--condensed_models",
        required=False,
        help="Model(s) to test for regression, but are first condensed by shrinking larger layers",
    )
    parser.add_argument(
        "--uniquify_layers",
        action="store_true",
        help="Remove duplicated layers in the model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Dataset to use for accuracy test (imagenet, sst2)",
    )
    parser.add_argument(
        "--sims",
        choices=["gold_model", "systemc", "fast-systemc", "rtl", "accuracy"],
        required=True,
        help="Simulation to run (gold_model, systemc, rtl, accuracy)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        required=True,
        help="Number of processes to run in parallel",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help="Comma separated list of tests to run (e.g. test1,test2)",
    )
    parser.add_argument(
        "--keep_build",
        action="store_true",
        help="Keep the generated rtl and use it to run rtl tests",
    )
    args = parser.parse_args()

    args.models = [s.strip() for s in args.models.split(",")]

    # Create directory with current time
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_folder = "regression_results/" + current_time
    os.makedirs(results_folder)
    # create softlink to latest results (delete old if exists)
    os.system("rm -f regression_results/latest")
    os.system(f"cd regression_results && ln -sf {current_time} latest")

    layers = {}
    layer_counts = {}

    # Add codegen layers
    layers = {}
    if args.tests is None:
        all_models = []
        if args.models is not None:
            all_models.extend(args.models)
        if args.condensed_models is not None:
            all_models.extend(args.condensed_models)

        for network in all_models:
            env_vars = os.environ.copy()
            env_vars["NETWORK"] = network
            subprocess.run(["make", "network-proto"], env=env_vars)
            add_layers(network, layers, layer_counts, args.uniquify_layers)
    else:
        assert (
            len(args.models) == 1
        ), "Only one model can be specified when using --tests"
        env_vars = os.environ.copy()
        env_vars["NETWORK"] = args.models[0]
        subprocess.run(["make", "network-proto"], env=env_vars)
        layers[args.models[0]] = args.tests.split(",")
        layer_counts[args.models[0]] = {layer: 1 for layer in layers[args.models[0]]}

        # Create tensor_metadata.json file, needed by aha flow
        # open the proto file
        with open(
            f"test/compiler/networks/{args.models[0]}/{os.environ['DATATYPE']}/model.txt",
            "r",
        ) as f:
            contents = f.read()
        params = param_pb2.Model()
        text_format.Parse(contents, params)

        # convert to json
        params_dict = MessageToDict(params, preserving_proto_field_name=True)
        for layer in layers[args.models[0]]:
            create_tensor_metadata_json(layer, params_dict)

        # tanh not yet supported
        if args.models[0] == 'bert' and layer == 'tanh':
            exit(0)

    if args.sims == "systemc" or args.sims == "fast-systemc":
        success = run_systemc_tests(
            layers,
            args.condensed_models,
            args.num_processes,
            results_folder,
            args.sims == "fast-systemc",
        )
    elif args.sims == "rtl":
        success = run_rtl_tests(
            layers,
            layer_counts,
            args.condensed_models,
            args.num_processes,
            results_folder,
            args.keep_build,
        )
    elif args.sims == "gold_model":
        success = run_gold_model_tests(layers, args.num_processes, results_folder)
    elif args.sims == "accuracy":
        success = run_accuracy(
            args.models, args.dataset, args.num_processes, results_folder
        )
    else:
        raise ValueError("Invalid simulation type")

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
