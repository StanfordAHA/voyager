set -o xtrace

EXTRA_FLAGS=""
if [ "${sim_level}" == "syn" ]; then
  VLOG_NAME=${sim_level}.v
  EXTRA_FLAGS="GATE_VLOG_DEP=./${VLOG_NAME}/${VLOG_NAME}.vts STAGE=gate"
fi

# TODO: need a way to skip dump file. I think we need to remove the dump options from catapult scripts. Need to test throughly
fsdb_name=${sim_level}_${network}_${layer}
VCS_VCSSIM_OPTS="+fsdbfile+${fsdb_name}.fsdb +fsdb+all=on +fsdb+dumpon+0"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
export PROJECT_ROOT=$(pwd)

if [ ! -f inputs/build/Catapult/${fsdb_name}.fsdb ]; then
  pushd inputs/build/Catapult/Accelerator/Accelerator.v1
  # run sim
  make -f scverify/Verify_concat_sim_rtl_v_vcs.mk sim CODEGEN_DIR=inputs/build/codegen NETWORK=${network} TESTS=${layer} DATATYPE=${datatype} SIMS=${sims} CLOCK_PERIOD=${clock_period} NETLIST_LEAF=${sim_level} VCS_VCSSIM_OPTS="${VCS_VCSSIM_OPTS}" ${EXTRA_FLAGS}
  popd
fi

# Convert fsdb to saif
if [ "${gen_saif}" == "True" ]; then
  fsdb2saif inputs/build/Catapult/${fsdb_name}.fsdb -o outputs/run.saif
fi
