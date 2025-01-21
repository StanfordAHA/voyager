set -o xtrace

# Save the current value of IFS
OLD_IFS=$IFS

# Input $network is expected to be a comma-delimited string unless there's only a single network to run
# Check if the string contains a comma
if [[ $network == *","* ]]; then
    IFS=','
fi

for n in $network; do
  make network-proto CODEGEN_DIR=inputs/build/codegen NETWORK=${n} DATATYPE=${datatype}
done

# Reset IFS to its original value
IFS=$OLD_IFS

ln -s ../inputs/build outputs/build
