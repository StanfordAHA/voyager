#! /usr/bin/env python
# =========================================================================
# construct.py
# =========================================================================

import os
import sys
import yaml

from mflowgen.components import Graph, Step

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for params
from params import build_params, sim_params, sweep_params

# load technology specific parameters
with open(
    os.path.join(
        os.environ["PROJECT_ROOT"],
        "physical-design",
        "technology",
        build_params["technology"] + ".yml",
    ),
    "r",
) as f:
    tech_params = yaml.safe_load(f)


def construct():
    g = Graph()

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------

    design_name = "Accelerator"

    parameters = {
        "construct_path": __file__,
        "design_name": design_name,
        "adk": tech_params["adk_name"],
        "adk_view": tech_params["adk_view"],
        "nthreads": 16,
        # Synthesis
        "flatten_effort": 2,  # honors manual no-ungrouping
        "topographical": True,  # get spef from synthesis
        # Hold fixing
        "signoff_engine": True,
        "hold_target_slack": 0.100,
        # Power
        "saif_instance": "sc_main/harness/accelerator/ccs_rtl/dut_inst",
        "technology": build_params["technology"],
    }

    # -----------------------------------------------------------------------
    # Create nodes
    # -----------------------------------------------------------------------

    this_dir = os.path.dirname(os.path.abspath(__file__))
    # ADK
    g.sys_path.append(
        os.path.join(
            os.environ["PROJECT_ROOT"], "physical-design", "technology", "adks"
        )
    )
    g.set_adk(tech_params["adk_name"])
    adk = g.get_adk_step()

    # Steps
    info = Step("info", default=True)
    # RTL and Library
    hls = Step(this_dir + "/../../common/hls")
    constraints = Step(this_dir + "/../../common/constraints")
    memories = Step(
        os.path.join(
            os.environ["PROJECT_ROOT"],
            "physical-design",
            "technology",
            "memories",
            build_params["technology"],
        )
    )
    synthesis = Step(this_dir + "/../../common/synopsys-dc-synthesis")
    plugin_dc = Step(this_dir + "/../../common/plugin-dc")

    # TODO: Add power analysis step

    # -----------------------------------------------------------------------
    # Graph -- Add nodes
    # -----------------------------------------------------------------------

    g.add_step(info)
    g.add_step(hls)
    g.add_step(constraints)
    g.add_step(memories)
    g.add_step(synthesis)
    g.add_step(plugin_dc)

    # -----------------------------------------------------------------------
    # Graph -- Add edges
    # -----------------------------------------------------------------------

    # Connect ADK to required nodes
    g.connect_by_name(adk, synthesis)

    synthesis.extend_inputs(plugin_dc.all_outputs())
    g.connect_by_name(plugin_dc, synthesis)

    synthesis.extend_inputs(memories.all_outputs())
    g.connect_by_name(memories, synthesis)

    g.connect_by_name(hls, synthesis)
    g.connect_by_name(constraints, synthesis)

    # Remove unwanted postconditions
    conditions = synthesis.get_postconditions()
    new_conditions = conditions[0:2]  # remove search for error line
    new_conditions = new_conditions + conditions[3:8]
    synthesis.set_postconditions(new_conditions)

    # -----------------------------------------------------------------------
    # Parameterize
    # -----------------------------------------------------------------------
    # Set custom parameters in params.py
    g.update_params(build_params)
    g.update_params(parameters)

    if build_params["technology"] == "tsmc40":
        # remove the memory module definitions from the generated RTL
        hls.extend_commands(
            [
                "sed -i '/module TSDN/,/endmodule/d;/module TS1N40LPB1024X128M4FWBA /,/endmodule/d' outputs/design.v"
            ]
        )

    synthesis.update_params(
        {
            "typ_dbs": tech_params["typ_dbs"],
            "ref_libs": tech_params["ref_libs"],
            "target_dbs": tech_params["target_dbs"],
            "additional_search_path": tech_params["additional_search_path"],
        },
        allow_new=True,
    )

    synthesis.extend_commands(["cp reports/Accelerator.mapped.area.rpt ../"])

    return g


if __name__ == "__main__":
    g = construct()
    g.plot()
