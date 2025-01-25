#!/bin/bash

tt=tttt_0.85v_25c
ff=pfff_0.89v_-40c
ss=psss_0.765v_125c

memory_names=(
"ip224rfsbhpm1r1w2048x32m4"
"ip224rfsbhpm1r1w1024x64m2"
"ip224rfsbhpm1r1w512x64m1"
"ip224rfsbhpm1r1w256x64m1"
"ip224rfsbhpm1r1w128x64m1"
"ip224uhdlp1p11rf_512x32m4b2c1s0_t0r0p0d0a1m1h"
"ip224uhdlp1p11rf_1024x64m4b2c1s0_t0r0p0d0a1m1h"
"ip224uhdlp1p11rf_2048x64m4b2c1s0_t0r0p0d0a1m1h"
"ip224uhdlp1p11rf_4096x64m4b2c1s1_t0r0p0d0a1m1h"
"ip224uhdlp1p11rf_8192x8m8b2c1s0_t0r0p0d0a1m1h"
)

mkdir -p outputs/memories

for memory_name in ${memory_names[@]}; do
  cp /intel16/ip/mem/${memory_name}/spice/${memory_name}.sp          outputs/memories/${memory_name}.sp
  cp /intel16/ip/mem/${memory_name}/oasis/${memory_name}.oas         outputs/memories/${memory_name}.oas
  cp /intel16/ip/mem/${memory_name}/rtl/${memory_name}_fast_func.sv  outputs/memories/${memory_name}.v
  cp /intel16/ip/mem/${memory_name}/lef/${memory_name}.lef           outputs/memories/${memory_name}.lef
  cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${tt}.lib     outputs/memories/${memory_name}-typical.lib
  cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${tt}.db      outputs/memories/${memory_name}-typical.db
  cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ff}.lib     outputs/memories/${memory_name}-bc.lib
  cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ff}.db      outputs/memories/${memory_name}-bc.db
  cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ss}.lib     outputs/memories/${memory_name}-wc.lib
  cp /intel16/ip/mem/${memory_name}/lib/${memory_name}_${ss}.db      outputs/memories/${memory_name}-wc.db
done

sed --in-place '/^\s*SITE.*\;/d' outputs/memories/*.lef
