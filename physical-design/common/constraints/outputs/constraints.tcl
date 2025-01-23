#=========================================================================
# Design Constraints File
#=========================================================================

set_units -time ns -capacitance pF

#=========================================================================
# General
#=========================================================================

# -- main clock
set main_clock_net  clk
set main_clock_name ideal_clock
# We use ns in all other places
set main_clock_period $clock_period
create_clock -name ${main_clock_name} \
             -period ${main_clock_period} \
             [get_ports ${main_clock_net}]


# -- load and drive
set_load -pin_load ${ADK_TYPICAL_ON_CHIP_LOAD} [all_outputs]
set_driving_cell -no_design_rule -lib_cell ${ADK_DRIVING_CELL} [all_inputs]

# -- fanout
set_max_fanout 20 ${design_name}

# -- transition
set_max_transition 0.05 ${design_name}

#=========================================================================
# Clock Constraints
#=========================================================================

set all_clock_setup_uncertainty  0.100
set all_clock_hold_uncertainty   0.100

set_clock_uncertainty -setup $all_clock_setup_uncertainty [get_clocks ${main_clock_name}]
set_clock_uncertainty -hold $all_clock_hold_uncertainty   [get_clocks ${main_clock_name}]

#=========================================================================
# Cycle Definition
#=========================================================================

set cycle60         [expr 0.60 * ${main_clock_period}]
set cycle50         [expr 0.50 * ${main_clock_period}]
set cycle40         [expr 0.40 * ${main_clock_period}]
set cycle30         [expr 0.30 * ${main_clock_period}]
set cycle25         [expr 0.25 * ${main_clock_period}]
set cycle20         [expr 0.20 * ${main_clock_period}]
set cycle10         [expr 0.10 * ${main_clock_period}]

#=========================================================================
# Input/Output Delays
#=========================================================================
# main clock
set_input_delay  $cycle20 -clock ${main_clock_name} [remove_from_collection [all_inputs] "rstn [get_ports ${main_clock_net}]" ]
set_output_delay $cycle40 -clock ${main_clock_name} [all_outputs]

# rstn is asynchronous
# set_input_delay $cycle30 -clock ${main_clock_name} [get_ports "rstn"]
