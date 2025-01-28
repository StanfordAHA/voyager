set -o xtrace

CATAPULT_BUILD_DIR=build/Catapult
mkdir -p ${CATAPULT_BUILD_DIR}


make network-proto NETWORK=resnet18 DATATYPE=${datatype}
make -j8 ${design_name} BUILD_DIR=build CATAPULT_BUILD_DIR=${CATAPULT_BUILD_DIR} DATATYPE=${datatype} OC_DIMENSION=16 IC_DIMENSION=16 INPUT_BUFFER_SIZE=1024 WEIGHT_BUFFER_SIZE=1024 ACCUM_BUFFER_SIZE=1024 CLOCK_PERIOD=${clock_period} TECHNOLOGY=${technology}

[[ -f ${CATAPULT_BUILD_DIR}/${design_name}/${design_name}.v1/concat_rtl.v ]] || {
  echo "Error: RTL generation failed"
  exit 1
}

cd outputs
ln -s ../build

cp build/Catapult/${design_name}/${design_name}.v1/concat_rtl.v design.v

# Renaming modules
OLD_IFS=$IFS
IFS=" "

for vname in design.v; do
  modname=$(grep -oP "(?<=module )ProcessingElement\w*" $vname | tail -1)
  sed -i "s/\<$modname\>/ProcessingElement/g" $vname
  modname=$(grep -oP "(?<=module )SystolicArray\w*" $vname | tail -1)
  sed -i "s/\<$modname\>/SystolicArray/g" $vname
  modname=$(grep -oP "(?<=module )VectorUnit\w*" $vname | tail -1)
  sed -i "s/\<$modname\>/VectorUnit/g" $vname
done

IFS=$OLD_IFS

cd ..
