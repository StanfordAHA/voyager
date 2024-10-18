#!/bin/env xonsh
import re
import os
import subprocess
import csv
import json
import copy
from pprint import pprint
from collect_layers import collect_layers, delete_nested_keys

$XONSH_SHOW_TRACEBACK = True

power_cond = "0p850v,25c"

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
            ["grep", "-oP", "-m", "1", "(?<=network: ).*", config],
            stdout=subprocess.PIPE,
        )
        network = p.communicate()[0].decode("utf-8").strip()
        p = subprocess.Popen(
            ["grep", "-oP", "-m", "1","(?<=layer: ).*", config],
            stdout=subprocess.PIPE,
        )
        layer = p.communicate()[0].decode("utf-8").strip()
        test = {"network": network, "layer": layer}
        if network not in tests:
          tests[network] = {}
        tests[network][layer] = test

        # print("\tParsing", network, layer)

        # check if test passed
        log = f"{node}/mflowgen-run.log"
        p = subprocess.Popen(
            ["grep", "-oP", "-m", "1", "(?<=Error count: )\\d+", log],
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
            ["grep", "-oP", "-m", "1", "(?<=Ideal runtime: )\\d+", log],
            stdout=subprocess.PIPE,
        )
        ideal_runtime = p.communicate()[0].decode("utf-8").strip()
        if ideal_runtime:
            ideal_runtime = int(ideal_runtime)
        p = subprocess.Popen(
            ["grep", "-oP", "-m", "1", "(?<=Runtime: ).\\d+", log],
            stdout=subprocess.PIPE,
        )
        runtime = p.communicate()[0].decode("utf-8").strip()
        util = ""
        if runtime:
            runtime = int(runtime)
            util = ideal_runtime/(runtime/clock_period)

        test.update({"ideal runtime (ns)": ideal_runtime, "runtime (ns)": runtime, "utilization": util})

        # Workaround: add the power fields so that they always exist as a key for all entries
        # so that the csv writer won't error out
        # test.update({f"rtl power (W@{power_cond})": "", f"syn power (W@{power_cond})": ""})

        # get the power numbers
        # for level in ["rtl", "syn"]:
        for level in ["rtl"]:
          test[f'{level}_power'] = {}
          ptpx_dir = gf`{build_dir}/*ptpx-{level}-network-{network}-layer-{layer}`
          if not ptpx_dir:
            continue
          ptpx_dir = ptpx_dir[0]
          rpt = f"{ptpx_dir}/reports/{design_name}.power.rpt"
          if os.path.exists(rpt):
            total_power = float($(grep -P -m 1 "Total Power" @(rpt)).split()[3])
            mem_power = float($(grep -P -m 1 "^memory" @(rpt)).split()[4])

          rpt = f"{ptpx_dir}/reports/InputBuffer.power.rpt"
          if os.path.exists(rpt):
            input_buf_power = float($(grep -P -m 1 "Totals" @(rpt)).split()[6])

          rpt = f"{ptpx_dir}/reports/WeightBuffer.power.rpt"
          if os.path.exists(rpt):
            weight_buf_power = float($(grep -P -m 1 "Totals" @(rpt)).split()[6])

          rpt = f"{ptpx_dir}/reports/AccumBuffer.power.rpt"
          if os.path.exists(rpt):
            accum_buf_power = float($(grep -P -m 1 "Totals" @(rpt)).split()[6])

          rpt = f"{ptpx_dir}/reports/SystolicArray.power.rpt"
          if os.path.exists(rpt):
            array_power = float($(grep -P -m 1 "Totals" @(rpt)).split()[6])

          rpt = f"{ptpx_dir}/reports/VectorUnit.power.rpt"
          if os.path.exists(rpt):
            vector_power = float($(grep -P -m 1 "Totals" @(rpt)).split()[6])
          
          try:
            test[f'{level}_power'].update({"total": total_power,
                                          "mem": mem_power,
                                          "input_buffer": input_buf_power, 
                                          "weight_buffer": weight_buf_power,
                                          "accum_buffer": accum_buf_power,
                                          "array": array_power, 
                                          "vector": vector_power}
                                          )
          except:
            pass

    return tests


def flatten_dict(d, parent_key='', sep='.'):
  items = {}
  for k, v in d.items():
    # Construct the new key by combining parent key and current key
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, dict):
      # If the value is a dictionary, recursively flatten it
      items.update(flatten_dict(v, new_key, sep=sep))
    else:
      # If it's not a dictionary, just set the value
      items[new_key] = v
  return items

if __name__ == "__main__":
  results = {}

  # get the list of builds
  build_dirs = g`build-*/`

  # extract metadata from the build dir name
  for build in build_dirs:
    build = build[:-1] # remove the trailing slash
    print("Parsing", build)
    _, datatype, dimension, buffer_sizes, clock_period = build.split("-")
    clock_period = clock_period[:-2] # remove the trailing unit
    input_buf_size, weight_buf_size, accum_buf_size = buffer_sizes.split("x")
    area = {}

    # get area numbers
    rpt = gf`{build}/*synopsys-dc-synthesis/reports/Accelerator.mapped.area.rpt`
    if rpt:
      total_area = int(float($(grep -m 1 -oP "(?<=Total cell area:).*" @(rpt[0])).strip()))
      total_area_comb = int(float($(grep -m 1 -oP "(?<=Combinational area:).*" @(rpt[0])).strip()))
      total_area_seq = int(float($(grep -m 1 -oP "(?<=Noncombinational area:).*" @(rpt[0])).strip()))
      mem_area = int(float($(grep -m 1 -oP "(?<=Macro/Black Box area:).*" @(rpt[0])).strip()))

      accum_buf_area = int(float($(grep -m 1 -P "while_accumulation_buffer_value_.*_rsc_comp " @(rpt[0])).split()[5])) # column 5 is the macro size
      # accum_buf_area = int(float($(grep -P ".*_matrixProcessor " @(rpt[0])).split()[5])) # column 5 is the macro size
      input_buf_area = int(float($(grep -m 1 -P ".*_inputBuffer " @(rpt[0])).split()[5]))
      weight_buf_area = int(float($(grep -m 1 -P ".*_weightBuffer " @(rpt[0])).split()[5]))

      area.update({"total": total_area,
                   "comb": total_area_comb,
                   "seq": total_area_seq,
                   "mem": {
                      "total": mem_area,
                      "input": input_buf_area,
                      "weight": weight_buf_area,
                      "accum": accum_buf_area
                    },
                 })

    rpt = gf`{build}/*synopsys-dc-synthesis/reports/SystolicArray.mapped.area.rpt`
    if rpt:
      array_area = int(float($(grep -m 1 -oP "(?<=Total cell area:).*" @(rpt[0])).strip()))
      array_area_comb = int(float($(grep -m 1 -oP "(?<=Combinational area:).*" @(rpt[0])).strip()))
      array_area_seq = int(float($(grep -m 1 -oP "(?<=Noncombinational area:).*" @(rpt[0])).strip()))
      area.update({"array": {
                   "total": array_area,
                   "comb": array_area_comb,
                   "seq": array_area_seq
                   }
                 })

    rpt = gf`{build}/*synopsys-dc-synthesis/reports/VectorUnit.mapped.area.rpt`
    if rpt:
      vector_area = int(float($(grep -oP "(?<=Total cell area:).*" @(rpt[0])).strip()))
      vector_area_comb = int(float($(grep -oP "(?<=Combinational area:).*" @(rpt[0])).strip()))
      vector_area_seq = int(float($(grep -oP "(?<=Noncombinational area:).*" @(rpt[0])).strip()))
      area.update({"vector": {
                    "total": vector_area,
                    "comb": vector_area_comb,
                    "seq": vector_area_seq
                   }
                 })

    # get simulation results
    tests = get_sim_results(build, float(clock_period))
    results.update({build: {"datatype": datatype,
                            "dimension": dimension,
                            "input_buf_size": input_buf_size,
                            "weight_buf_size": weight_buf_size,
                            "accum_buf_size": accum_buf_size,
                            "clock_period (ns)": clock_period,
                            "area": area, "tests": tests }})


  json.dump(results, open("results.json", "w"), indent=2)

  # write to csv
  with open(f'simulations.csv', 'w') as f:
    result_ = copy.deepcopy(results)
    # area is reported in summary report, tests need to be flattened
    delete_nested_keys(result_, "area")
    delete_nested_keys(result_, "tests")
    
    # Flatten the tests dict, store as a list of a flat dict
    for build in results.keys():
      result_[build]["tests"] = []
      for network, layer_dict in results[build]["tests"].items():
        for layer, layer_data in layer_dict.items():
          result_[build]["tests"].append(
            {"network": network, "layer": layer, **flatten_dict(layer_data)}
          )

    # Get the headers, for tests, need to get its first-level keys
    header = list(result_[list(result_)[0]].keys())
    header.remove("tests")
    try:
      header += list(result_[list(result_)[0]]["tests"][0])
    except:
      pass

    dw = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
    dw.writeheader()
    for entry in result_.values():
      flatten = entry.copy()
      flatten.pop("tests")

      for values in entry["tests"]:
        flatten.update(values)
        dw.writerow(flatten)

  with open(f'summary.csv', 'w') as f:
    # get the headers in a list
    results_ = copy.deepcopy(results)
    delete_nested_keys(results_, "tests")
    
    # Flatten area dict 
    for build in results.keys():
      results_[build] = flatten_dict(results_[build])

    # Flatten the tests dict, store as a list of a flat dict after accumulating values
    for build in results.keys():
      # get the required layers and multipliers for the datatype
      networks = ["resnet18", "resnet50", "mobilebert_encoder"]
      layers_to_run = collect_layers(networks, results[build]["datatype"], verbose=False)
      results_[build]["tests"] = []
      for network, layer_dict in results[build]["tests"].items():
        # tracking whether all required layers are completed
        checklist = list(layers_to_run[network])
        status = "Passed"
        power_status = "Complete"
        ideal_runtime = runtime = energy = mem_energy = input_buf_energy = weight_buf_energy = accum_buf_energy = arr_energy = vec_energy = 0
        for layer, layer_data in layer_dict.items():
          try:
            ideal_runtime += layer_data["ideal runtime (ns)"] * layers_to_run[network][layer]
            runtime += layer_data["runtime (ns)"] * layers_to_run[network][layer]
            status = "Failed" if layer_data["status"] != "Passed" else status
          except:
            print("\tIncomplete simulation data for", build, network, layer)
            status = "Abnormal"
            
          try:
            # put energy at the back so that if energy is not available, the rest of the data is still there
            energy += layer_data["rtl_power"]["total"] * layer_data["runtime (ns)"] * 1e-6 * layers_to_run[network][layer]
            mem_energy += layer_data["rtl_power"]["mem"] * layer_data["runtime (ns)"] * 1e-6 * layers_to_run[network][layer]
            input_buf_energy += layer_data["rtl_power"]["input_buffer"] * layer_data["runtime (ns)"] * 1e-6 * layers_to_run[network][layer]
            weight_buf_energy += layer_data["rtl_power"]["weight_buffer"] * layer_data["runtime (ns)"] * 1e-6 * layers_to_run[network][layer]
            accum_buf_energy += layer_data["rtl_power"]["accum_buffer"] * layer_data["runtime (ns)"] * 1e-6 * layers_to_run[network][layer]
            arr_energy += layer_data["rtl_power"]["array"] * layer_data["runtime (ns)"] * 1e-6 * layers_to_run[network][layer]
            vec_energy += layer_data["rtl_power"]["vector"] * layer_data["runtime (ns)"] * 1e-6 * layers_to_run[network][layer]
          except:
            print("\tIncomplete power data for", build, network, layer)
            power_status = "Incomplete"

          checklist.remove(layer)
        
        # multiply by 21
        if network == "mobilebert_encoder":
          ideal_runtime *= 21
          runtime *= 21
          mem_energy *= 21
          input_buf_energy *= 21
          weight_buf_energy *= 21
          accum_buf_energy *= 21
          arr_energy *= 21
          vec_energy *= 21
          energy *= 21

        # Calculate utilization
        try:
          utilization = ideal_runtime/float(runtime)
        except:
          # if runtime is 0, means tests likely failed, so cannot divide
          utilization = "N/A"

        # if power data is incomplete, mark the build as incomplete
        if power_status == "Incomplete":
          status += " - Incomplete"
        # if some layers are missing, then both simulation and power data are not trustworthy
        if checklist:
          status = "Incomplete"

        results_[build]["tests"].append(
          {"network": network,
          "status": status,
          "ideal runtime (ns)": ideal_runtime,
          "runtime (ns)": runtime,
          "utilization": utilization,
          "energy (mJ)": energy,
          "mem energy": mem_energy, 
          "input buffer energy": input_buf_energy, 
          "weight buffer energy": weight_buf_energy,
          "accum buffer energy": accum_buf_energy,
          "array energy": arr_energy,
          "vector energy": vec_energy}
        )
    
    header = list(results_[list(results_)[0]].keys())
    header.remove("tests")
    try:
      header += list(results_[list(results_)[0]]["tests"][0])
    except:
      pass
    dw = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
    dw.writeheader()

    for entry in results_.values():
      flatten = entry.copy()
      flatten.pop("tests")

      for values in entry["tests"]:
        flatten.update(values)
        dw.writerow(flatten)
