#!/bin/bash

mkdir -p outputs/memories

tt=tt1p1v25c
ff=ff1p21vm40c
ss=ss0p99v125c

for memory in ts*; do
    cp ${memory}/VERILOG/${memory}_${tt}.v outputs/memories/${memory}.v
    cp ${memory}/NLDM/${memory}_${tt}.lib outputs/memories/${memory}-typical.lib
    cp ${memory}/NLDM/${memory}_${tt}.db outputs/memories/${memory}-typical.db
    cp ${memory}/NLDM/${memory}_${ff}.lib outputs/memories/${memory}-bc.lib
    cp ${memory}/NLDM/${memory}_${ff}.db outputs/memories/${memory}-bc.db
    cp ${memory}/NLDM/${memory}_${ss}.lib outputs/memories/${memory}-wc.lib
    cp ${memory}/NLDM/${memory}_${ss}.db outputs/memories/${memory}-wc.db
done