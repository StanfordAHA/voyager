#!/bin/bash

tt=tttt_0.85v_25c
ff=pfff_0.89v_-40c
ss=psss_0.765v_125c

cp /intel16/ip/mem/${memory_name}/spice/${memory_name}.sp          outputs/${memory_name}.sp
cp /intel16/ip/mem/${memory_name}/oasis/${memory_name}.oas         outputs/${memory_name}.oas
cp /intel16/ip/mem/${memory_name}/rtl/${memory_name}_fast_func.sv  outputs/${memory_name}.v
cp /intel16/ip/mem/${memory_name}/lef/${memory_name}.lef           outputs/${memory_name}.lef
cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${tt}.lib     outputs/${memory_name}-typical.lib
cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${tt}.db      outputs/${memory_name}-typical.db
cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ff}.lib     outputs/${memory_name}-bc.lib
cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ff}.db      outputs/${memory_name}-bc.db
cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ss}.lib     outputs/${memory_name}-wc.lib
cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ss}.db      outputs/${memory_name}-wc.db

sed --in-place '/^\s*SITE.*\;/d' outputs/*.lef
