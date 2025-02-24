#=========================================================================
# designer-interface.tcl
#=========================================================================
# The designer_interface.tcl file is the first script run by PT 
# and sets up ASIC design kit variables and inputs.
#
# Author : Christopher Torng
# Date   : May 20, 2019


#-------------------------------------------------------------------------
# Parameters
#-------------------------------------------------------------------------

set ptpx_design_name        		$::env(design_name)

# The strip path must be defined!
#
#   export strip_path = th/dut
#
# There must _not_ be any quotes, or read_saif will fail. This fails:
#
#   export strip_path = "th/dut"
#

set ptpx_strip_path         		$::env(saif_instance)

set ptpx_analysis_mode				  $::env(analysis_mode)
set ptpx_zero_delay_simulation	$::env(zero_delay_simulation)
set ptpx_op_condition				    $::env(lib_op_condition)
set ptpx_clock_buffer           $::env(default_clock_buffer)

#-------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------

set adk_dir                       inputs/adk

set ptpx_additional_search_path   $adk_dir
set ptpx_target_libraries        [join "
                                    [lsort [glob -nocomplain inputs/adk/stdcells*-$::env(corner).db]]
                                 "]
set ptpx_extra_link_libraries    [join " 
                                    [lsort [glob -nocomplain inputs/*-$::env(corner).db]]
                                    [lsort [glob -nocomplain inputs/adk/*io*-$::env(corner).db]]
                                 "]


#-------------------------------------------------------------------------
# Inputs
#-------------------------------------------------------------------------

# set ptpx_gl_netlist         [join "
#                                      [lsort [glob -nocomplain inputs/*.v]]
#                                      [lsort [glob -nocomplain inputs/adk/*.v]]
#                             "]
set ptpx_gl_netlist         [lsort [glob -nocomplain inputs/*.v]]
set ptpx_sdc                inputs/design.sdc
set ptpx_spef               inputs/design.spef.gz
set ptpx_saif               inputs/run.saif
set ptpx_vcd				        inputs/run.vcd
set ptpx_namemap			      inputs/design.namemap

#-------------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------------

set ptpx_reports_dir	   	reports
set ptpx_logs_dir	   	  	logs
set ptpx_outputs_dir	  	outputs


