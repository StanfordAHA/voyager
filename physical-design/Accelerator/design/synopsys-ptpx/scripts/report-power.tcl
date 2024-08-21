#=========================================================================
# report-power.tcl
#=========================================================================
# This script checks for potential errors, performs power analysis and
# reports the power of the design
#
# Author : Maximilian Koschay
# Date   : 05.03.2021


# Check for potential problems for power analysis

check_power > $ptpx_reports_dir/$ptpx_design_name.power.check.rpt

# Set power analysis options

if {$ptpx_analysis_mode == "time_based"} {
	
	# Set some analysis options for peak power analysis over time:
	# - include all hierarchy cells, except for leaf cells
	# - report worst peak power phases in final report
	# - save waveform as FSDB in the reports directory 
	set_power_analysis_options 	-include all_without_leaf \
								-npeak 10 -peak_power_instances \
								-npeak_out $ptpx_reports_dir/$ptpx_design_name \
								-waveform_output $ptpx_reports_dir/$ptpx_design_name
} 

# Apply activiy annotations and calculate power values
# estimate clock tree
if {${ptpx_clock_buffer} != "undefined"} {
  estimate_clock_network_power ${ptpx_clock_buffer}
}
update_power > $ptpx_logs_dir/$ptpx_design_name.power.update.rpt

#-------------------------------------------------------------------------
# Power Reports
#-------------------------------------------------------------------------

# Report switching activity
report_switching_activity \
  > $ptpx_reports_dir/$ptpx_design_name.activity.post.rpt

# Group-based power report
report_power -nosplit -verbose \
  > $ptpx_reports_dir/$ptpx_design_name.power.rpt

# # Clock network estimate power report
# report_power -nosplit -verbose -include_estimated_clock_network \
#   > $ptpx_reports_dir/$ptpx_design_name.power.clock_tree.rpt

# Custom reporting
report_power -cell_power [get_cells -hier *systolicArray] \
  > $ptpx_reports_dir/$ptpx_design_name.power.systolic_array.rpt

# Deal with unpredictable ungrouping behavior
set matrix_glue [get_cells *matrixProcessor/* -filter {is_memory_cell != true && base_name !~ *systolicArray*} -quiet]
append_to_collection matrix_glue [get_cells */matrixProcessor* -filter {is_memory_cell != true && base_name !~ *systolicArray*} -quiet]
append_to_collection matrix_glue [get_cells */matrixProcessor/* -filter {is_memory_cell != true && base_name !~ *systolicArray*} -quiet]
report_power -cell_power $matrix_glue \
  > $ptpx_reports_dir/$ptpx_design_name.power.matrix_glue.rpt

# Cell hierarchy power report
report_power -nosplit -hierarchy -verbose \
  > $ptpx_reports_dir/$ptpx_design_name.power.hier.rpt

report_clock_gate_savings > $ptpx_reports_dir/$ptpx_design_name.clock-gate-savings.rpt

# Get Leakage for each used library and count of cells

foreach_in_collection l [get_libs] {
	if {[get_attribute [get_lib $l] default_threshold_voltage_group] == ""} {
	    set libname [get_object_name [get_lib $l]]
	    set_user_attribute [get_lib $l] default_threshold_voltage_group $libname -class lib
	}
}
report_power -threshold_voltage_group > $ptpx_reports_dir/$ptpx_design_name.power.leakage-per-lib.rpt
report_threshold_voltage_group > $ptpx_reports_dir/$ptpx_design_name.power.cells-per-vth-group.rpt
