#! /usr/bin/env python
# =========================================================================
# construct.py
# =========================================================================

import os
import sys

from mflowgen.components import Graph, Step

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # for params
from params import build_params, sim_params, sweep_params


def construct():
    g = Graph()

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------

    adk_name = "intel16-adk"
    adk_view = "multivt"

    design_name = "Accelerator"

    parameters = {
        "construct_path": __file__,
        "design_name": design_name,
        "adk": adk_name,
        "adk_view": adk_view,
        "adk_stdcell": "b15_7t_108pp",
        "adk_libmodel": "nldm",
        "nthreads": 16,
        # Synthesis
        "flatten_effort": 2,  # honors manual no-ungrouping
        "topographical": True, # get spef from synthesis
        # Hold fixing
        "signoff_engine": True,
        "hold_target_slack": 0.100,
        # Power
        "saif_instance": "sc_main/harness/accelerator/ccs_rtl/dut_inst",
    }

    # -----------------------------------------------------------------------
    # Create nodes
    # -----------------------------------------------------------------------

    this_dir = os.path.dirname(os.path.abspath(__file__))
    # ADK
    g.set_adk(adk_name)
    adk = g.get_adk_step()

    # Steps
    info = Step("info", default=True)
    # RTL and Library
    hls = Step(this_dir + "/hls")
    constraints = Step(this_dir + "/constraints")
    sram = Step(this_dir + "/../../common/memory")
    # Simulation
    # codegen = Step(this_dir + "/codegen")
    vcs_build = Step(this_dir + "/vcs-build")
    rtl_vcs_build = vcs_build.clone()
    rtl_vcs_build.set_name( 'rtl-vcs-build')
    syn_vcs_build = vcs_build.clone()
    syn_vcs_build.set_name( 'syn-vcs-build')
    sim = Step(this_dir + "/sim")
    rtl_sim = sim.clone()
    rtl_sim.set_name( 'rtl-sim')
    # this is a hack for param sweeping for RTL sim. We only feed the namemap to Synth from this step
    rtl_sim_namemap = sim.clone()
    rtl_sim_namemap.set_name('rtl-sim-namemap')
    syn_sim = sim.clone()
    syn_sim.set_name( 'syn-sim' )
    # Synthesis
    # synth = Step(this_dir + "/cadence-genus-synthesis")
    synth = Step(this_dir + "/../../common/synopsys-dc-synthesis")
    # Power
    ptpx = Step(this_dir + "/../../common/synopsys-ptpx")
    ptpx_rtl = ptpx.clone()
    ptpx_syn = ptpx.clone()
    ptpx_rtl.set_name('ptpx-rtl')
    ptpx_syn.set_name('ptpx-syn')
    
    # Customization
    custom_hack_synth_sdc = Step(this_dir + "/custom-hack-synth-sdc")
    custom_dc_synth = Step(this_dir + "/custom-dc-synthesis")
    custom_ptpx = Step(this_dir + "/custom-ptpx")

    # -----------------------------------------------------------------------
    # Node modifications
    # -----------------------------------------------------------------------

    # Customize for each sram type
    sram_rf = sram.clone()
    sram_rf.set_name("sram-rf")
    sram_sp = sram.clone()
    sram_sp.set_name("sram-sp")

    mem_nodes = {
        sram_rf: "ip224rfsbhpm1r1w1024x64m2",
        sram_sp: "ip224uhdlp1p11rf_1024x64m4b2c1s1_t0r0p0d0a1m1h"
    }

    all_mem_types = set()
    for mem in mem_nodes.values():
        all_mem_types.add(mem)

    for node, mem in mem_nodes.items():
        node.extend_outputs(
            [
                f"{mem}.sp",
                f"{mem}.oas",
                f"{mem}.v",
                f"{mem}.lef",
                f"{mem}-typical.lib",
                f"{mem}-typical.db",
                f"{mem}-bc.lib",
                f"{mem}-bc.db",
                f"{mem}-wc.lib",
                f"{mem}-wc.db",
            ]
        )

    for mem in all_mem_types:
        for step in [
            ptpx_rtl,
            ptpx_syn,
        ]:
            step.extend_inputs([f"{mem}-typical.db", f"{mem}-wc.db", f"{mem}-bc.db"])


        for step in [
            synth,
        ]:
            step.extend_inputs(
                # [f"{mem}-typical.lib", f"{mem}-wc.lib", f"{mem}-bc.lib", f"{mem}.lef"]
                [f"{mem}-typical.db", f"{mem}-wc.db", f"{mem}-bc.db", f"{mem}.lef"]
            )

        for step in [
            hls
        ]:
            step.extend_inputs(
                [f"{mem}.v"]
            )

        # signoff.extend_inputs([f"{mem}.oas"])
        # lvs.extend_inputs([f"{mem}.sp"])

    synth.extend_outputs(["design.spef.gz"])
    synth.extend_inputs(custom_dc_synth.all_outputs())

    ptpx_rtl.extend_inputs(custom_ptpx.all_outputs())
    ptpx_syn.extend_inputs(custom_ptpx.all_outputs())

    # Add extra input edges to innovus steps that need custom tweaks
    # init.extend_inputs(custom_init.all_outputs())
    # init.extend_inputs(["io_pad_placement.tcl"])
    # power.extend_inputs(custom_power.all_outputs())
    # route.extend_inputs(pre_route.all_outputs())
    # signoff.extend_inputs(
    #     [
    #         "change-names.tcl",
    #         "bump-util.tcl",
    #         "create-physical-pin.tcl",
    #         "full-chip-delete-vias.tcl",
    #     ]
    # )

    # Add merge files
    # merge_modules = [f"{module}.oas" for module in list(all_mem_types)]
    # merge.extend_inputs(merge_modules)

    # Add LVS overrides
    # lvs.extend_inputs(lvs_overrides.all_outputs())

    # -----------------------------------------------------------------------
    # Graph -- Add nodes
    # -----------------------------------------------------------------------

    g.add_step(info)
    g.add_step(hls)
    # g.add_step(codegen)
    g.add_step(rtl_vcs_build)
    g.add_step(rtl_sim)
    g.add_step(rtl_sim_namemap)
    for mem in mem_nodes.keys():
        g.add_step(mem)
    g.add_step(constraints)
    g.add_step(custom_dc_synth)
    g.add_step(custom_ptpx)
    g.add_step(synth)
    g.add_step(custom_hack_synth_sdc)
    g.add_step(syn_vcs_build)
    g.add_step(syn_sim)
    g.add_step(ptpx_rtl)
    g.add_step(ptpx_syn)

    # -----------------------------------------------------------------------
    # Graph -- Add edges
    # -----------------------------------------------------------------------

    # Connect ADK to required nodes
    g.connect_by_name(adk, synth)
    g.connect_by_name(adk, syn_vcs_build)
    g.connect_by_name(adk, ptpx_rtl)
    g.connect_by_name(adk, ptpx_syn)


    # Connect memory to required nodes
    for mem in mem_nodes.keys():
        g.connect_by_name(mem, hls)
        g.connect_by_name(mem, synth)
        g.connect_by_name(mem, rtl_vcs_build)
        g.connect_by_name(mem, syn_vcs_build)
        g.connect_by_name(mem, ptpx_rtl)
        g.connect_by_name(mem, ptpx_syn)


    g.connect_by_name(hls, synth)
    g.connect(hls.o("design.v"), rtl_vcs_build.i("design.v"))
    g.connect_by_name(hls, rtl_vcs_build)
    g.connect_by_name(rtl_vcs_build, rtl_sim)
    g.connect_by_name(rtl_vcs_build, rtl_sim_namemap)
    g.connect_by_name(rtl_sim_namemap, synth)  # run.saif to generate namemap
    g.connect_by_name(constraints, synth)
    g.connect_by_name(custom_dc_synth, synth) # overwrite scripts
    g.connect_by_name(synth, custom_hack_synth_sdc)
    g.connect_by_name(synth, syn_vcs_build)
    g.connect(hls.o("build"), syn_vcs_build.i("build"))
    g.connect_by_name(syn_vcs_build, syn_sim)
    g.connect_by_name(rtl_sim, ptpx_rtl)
    g.connect_by_name(custom_ptpx, ptpx_rtl)
    g.connect(synth.o("design.v"), ptpx_rtl.i("design.v"))
    g.connect(synth.o("design.spef.gz"), ptpx_rtl.i("design.spef.gz"))
    g.connect(synth.o("design.namemap"), ptpx_rtl.i("design.namemap"))
    g.connect_by_name(custom_hack_synth_sdc, ptpx_rtl)
    g.connect_by_name(custom_ptpx, ptpx_syn)
    g.connect_by_name(syn_sim, ptpx_syn)
    g.connect(synth.o("design.v"), ptpx_syn.i("design.v"))
    g.connect(synth.o("design.spef.gz"), ptpx_syn.i("design.spef.gz"))
    g.connect_by_name(custom_hack_synth_sdc, ptpx_syn)

    # Remove unwanted postconditions
    conditions = synth.get_postconditions()
    new_conditions = conditions[0:2] #remove search for error line
    new_conditions = new_conditions + conditions[3:8]
    synth.set_postconditions(new_conditions)

    # -----------------------------------------------------------------------
    # Parameterize
    # -----------------------------------------------------------------------
    # Set custom parameters in params.py
    parameters.update(build_params)
    parameters.update(sim_params)

    g.update_params(parameters)

    # Customize parameters per node
    for node, mem in mem_nodes.items():
        node.update_params({"memory_name": mem})

    # RTL simulation
    rtl_vcs_build.update_params( {"sim_level": "rtl"} )
    rtl_sim.update_params( {"sim_level": "rtl"} )
    rtl_sim_namemap.update_params( {"sim_level": "rtl"} )

    # Post-syn simulation
    syn_vcs_build.update_params( {"sim_level": "syn"} )
    syn_sim.update_params( {"sim_level": "syn"} )

    ptpx_rtl.update_params( { 'corner': 'typical', "default_clock_buffer": "lib224_b15_7t_108pp_clk_lvt_tttt_0p800v_25c_tttt_ctyp_nldm/b15cbf000ah1n08x5"} )
    ptpx_syn.update_params( { 'corner': 'typical', "default_clock_buffer": "lib224_b15_7t_108pp_clk_lvt_tttt_0p800v_25c_tttt_ctyp_nldm/b15cbf000ah1n08x5"} )

    # Parameter Sweep
    # Sweep over design configs
    
    # NOTE: mflowgen has a bug. If two inputs of a node are independent branches off a node we swept,
    #       the parameter space gets cross multipled even though we only swept a single parameter. The
    #       correct behavior is to only take the upstream with the same parameter value.
    #       Even if this is resolved, getting the now-renamed downstream node is also a hacky task.
    #       This issue would not arise (possibly) if we didn't have to create a dedicated rtl-sim-namemap node
    #       so that sweeping rtl-sim wouldn't cause DC to run multiple times (which makes sense and probably must be done).

    # Cannot really think of any easy way to solve this, unless we don't sweep tests and run simulation in one node, but ptpx will be difficult
    # We have to use sweep_params["tests"] to determine the different combinations, and
    # sweep (or maybe clone) ptpx using that. Explicitly change the input it's looking for to the specific fsdb name. 
    # The only thing is parameter doesn't support dict. If we want to make this work, we need a dumber 
    # ways to specify the sweep parameters (tests).

    # codegen: need to generate data for all networks involved
    # codegen.update_params( { 'network': list(sweep_params["tests"]) } )

    # Sweep over tests
    networks = list(sweep_params["tests"])
    # for sim in [rtl_sim, syn_sim]:
    for sim in [rtl_sim]:
        parameterized_step = g.param_space(sim, "network", networks)
        for step in parameterized_step:
            network = step.get_param("network")
            layers = sweep_params["tests"][network]
            g.param_space(step, "layer", layers)

    return g


if __name__ == "__main__":
    g = construct()
    g.plot()

