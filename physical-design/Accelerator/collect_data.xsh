#!/bin/env xonsh
import re
import os
import subprocess
import csv
from pprint import pprint as pp

$XONSH_SHOW_TRACEBACK = True

power_cond = "0p850v,25c"

multiplier = {
  "resnet18": {
    "conv1": 1,
    "layer1_0_conv1": 1,
    "layer1_0_conv2": 1,
    "layer1_1_conv1": 1,
    "layer1_1_conv2": 1,
    "layer2_0_downsample": 1,
    "layer2_0_conv1": 1,
    "layer2_0_conv2": 1,
    "layer2_1_conv1": 1,
    "layer2_1_conv2": 1,
    "layer3_0_downsample": 1,
    "layer3_0_conv1": 1,
    "layer3_0_conv2": 1,
    "layer3_1_conv1": 1,
    "layer3_1_conv2": 1,
    "layer4_0_downsample": 1,
    "layer4_0_conv1": 1,
    "layer4_0_conv2": 1,
    "layer4_1_conv1": 1,
    "layer4_1_conv2": 1,
    "fc": 1,
  },
  "mobilebert": {
    "bottleneck_input_dense": 2 * 21,
    "bottleneck_input_LayerNorm": 5 * 21,
    "attention_self_query_layer": 2 * 21,
    "attention_self_value_layer": 1 * 21,
    "attention_self_attention_scores_2": 4 * 21,
    "attention_self_attention_probs_0": 4 * 21,
    "attention_self_context_layer_0": 1 * 21,
    "attention_self_context_layer_1": 1 * 21,
    "attention_self_context_layer_2": 1 * 21,
    "attention_self_context_layer_3": 1 * 21,
    "attention_output_dense": 1 * 21,
    "ffn_0_output_dense": 2 * 21,
    "intermediate_dense": 2 * 21,
    "output_bottleneck_dense": 1 * 21,
    "output_bottleneck_LayerNorm": 1 * 21,
    "classifier": 1 * 21,
  }
}


def get_sim_results(build_dir, clock_period, design_name="Accelerator"):
    # list out all simulation nodes 
    sim_dirs = os.listdir(build_dir)
    # exclude the rtl-sim namemap run
    sim_dirs = [f"{build_dir}/{dir}" for dir in sim_dirs if re.match(r".*-sim(?!-namemap)", dir)]

    tests = {}
    for node in sim_dirs:
        # get the test name
        config = f"{node}/configure.yml"
        p = subprocess.Popen(
            ["grep", "-oP", "(?<=network: ).*", config],
            stdout=subprocess.PIPE,
        )
        network = p.communicate()[0].decode("utf-8").strip()
        p = subprocess.Popen(
            ["grep", "-oP", "(?<=layer: ).*", config],
            stdout=subprocess.PIPE,
        )
        layer = p.communicate()[0].decode("utf-8").strip()
        test = {"network": network, "layer": layer}
        if network not in tests:
          tests[network] = {}
        tests[network][layer] = test

        # check if test passed
        log = f"{node}/mflowgen-run.log"
        p = subprocess.Popen(
            ["grep", "-oP", "(?<=Error count: )\\d", log],
            stdout=subprocess.PIPE,  # capture the output, not print to screen
        )
        out = p.communicate()[0]    # pattern not found
        if not p.returncode == 0:
            test.update({"status": "Abnormal"})
        else:
            count = int(out.decode('utf-8').strip())
            if count > 0:
                test.update({"status": f"Failed: {count}"})
            else:
                test.update({"status": "Passed"})

        # get the runtime
        p = subprocess.Popen(
            ["grep", "-oP", "(?<=Ideal cycles: ).\\d+", log],
            stdout=subprocess.PIPE,
        )
        ideal_cycles = p.communicate()[0].decode("utf-8").strip()
        if ideal_cycles:
            ideal_cycles = int(ideal_cycles)
        p = subprocess.Popen(
            ["grep", "-oP", "(?<=Runtime: ).\\d+", log],
            stdout=subprocess.PIPE,
        )
        runtime = p.communicate()[0].decode("utf-8").strip()
        if runtime:
            runtime = int(runtime)
        test.update({"ideal cycles": ideal_cycles, "runtime (ns)": runtime, "utilization": ideal_cycles/(runtime/clock_period)})

        # Workaround: add the power fields so that they always exist as a key for all entries
        # so that the csv writer won't error out
        # test.update({f"rtl power (W@{power_cond})": "", f"syn power (W@{power_cond})": ""})

        # get the power numbers
        # for level in ["rtl", "syn"]:
        for level in ["rtl"]:
          ptpx_dir = gf`{build_dir}/*ptpx-{level}-network-{network}-layer-{layer}`
          if not ptpx_dir:
            continue
          ptpx_dir = ptpx_dir[0]
          rpt = f"{ptpx_dir}/reports/{design_name}.power.rpt"
          if os.path.exists(rpt):
            total_power = float($(grep -P "Total Power" @(rpt)).split()[3])
            mem_power = float($(grep -P "memory" @(rpt)).split()[4])
            test.update({f"{level} power (W@{power_cond})": total_power, "mem power": mem_power})
          rpt = f"{ptpx_dir}/reports/{design_name}.power.systolic_array.rpt"
          if os.path.exists(rpt):
            array_power = float($(grep -P "Totals" @(rpt)).split()[-2])
            test.update({"array power": array_power})
          rpt = f"{ptpx_dir}/reports/{design_name}.power.matrix_glue.rpt"
          if os.path.exists(rpt):
            matrix_glue_power = float($(grep -P "Totals" @(rpt)).split()[-2])
            test.update({"matrix glue power": matrix_glue_power})
    return tests


if __name__ == "__main__":
  results = {}

  # get the list of builds
  build_dirs = g`build-*/`
  # extract metadata from the build dir name
  for build in build_dirs:
    build = build[:-1] # remove the trailing slash
    _, datatype, dimension, clock_period = build.split("-")
    clock_period = clock_period[:-2] # remove the trailing unit
    # get area numbers
    rpt = gf`{build}/*synopsys-dc-synthesis/reports/Accelerator.mapped.area.rpt`
    if rpt:
      total_area = int(float($(grep -oP "(?<=Total cell area:).*" @(rpt[0])).strip()))
      mem_area = int(float($(grep -oP "(?<=Macro/Black Box area:).*" @(rpt[0])).strip()))
      try:
        array_area = int(float($(grep -P ".*systolicArray " @(rpt[0])).split()[1]))
      except:
        # right now some builds are still flattening
        array_area = "N/A"

      area = {"total_area (um2)": total_area, "mem_area": mem_area, "array_area": array_area}
    # get simulation results
    tests = get_sim_results(build, float(clock_period))
    results.update({build: {"datatype": datatype, "dimension": dimension, "clock_period (ns)": clock_period,
                            "area": area, "tests": tests }})

  def get_nested_keys(d):
    return list(d[list(d)[0]].keys())

  def get_nested_values(d, key=None):
    if key is None:
      key = get_nested_keys(d)[0]
    return d[list(d)[0]][key]

  # write to csv
  with open(f'simulations.csv', 'w') as f:
    # get the headers in a list
    header = list(results[list(results)[0]].keys())
    header = get_nested_keys(results)
    fields_to_remove = ["tests", "area"]
    for entry in fields_to_remove:
      header.remove(entry)
    networks_dict = get_nested_values(results, "tests")
    header+=list(get_nested_values(networks_dict))
    dw = csv.DictWriter(f, fieldnames=header)
    dw.writeheader()
    for entry in results.values():
      flatten = entry.copy()
      for field in fields_to_remove:
        del flatten[field]

      for network in entry["tests"].values():
        for layer in network.values():
          flatten.update(layer)
          dw.writerow(flatten)

  with open(f'summary.csv', 'w') as f:
    # get the headers in a list
    header = list(results[list(results)[0]].keys())
    header = get_nested_keys(results)
    header += list(get_nested_values(results, "area").keys())
    header += ["network", "status", "ideal cycles", "runtime (ns)", "utilization", "energy (mJ)"]
    fields_to_remove = ["tests", "area"]
    for entry in fields_to_remove:
      header.remove(entry)
    dw = csv.DictWriter(f, fieldnames=header)
    dw.writeheader()

    # Calculate total energy and runtime for a network
    for entry in results.values():
      data = entry.copy()
      for field in fields_to_remove:
        del data[field]
      data.update(entry["area"])
      for network, layers in entry["tests"].items():
        checklist = list(multiplier[network])
        status = "Complete"
        ideal_cycles = 0
        runtime = 0
        energy = 0
        for layer, layer_data in layers.items():
          # some reports may not be generated yet.
          try:
            ideal_cycles += layer_data["ideal cycles"] * multiplier[network][layer]
            runtime += layer_data["runtime (ns)"] * multiplier[network][layer]
            energy += layer_data[f"rtl power (W@{power_cond})"] * layer_data["runtime (ns)"] * 1e-6 * multiplier[network][layer]
          except:
            status = "Incomplete"
          checklist.remove(layer)
        utilization = ideal_cycles/(runtime/float(entry["clock_period (ns)"]))
        # some layers are missing
        if checklist:
          status = "Incomplete"
        data.update({"network": network, "status": status, "ideal cycles": ideal_cycles, "runtime (ns)": runtime, "utilization": utilization, "energy (mJ)": energy})
        dw.writerow(data)
    
