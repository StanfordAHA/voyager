mflowgen-based physical design flow for [DNN accelerator](https://code.stanford.edu/tsmc40r/brainpower/accelerator).

## Instructions
1. `git submodules update --init --recursive`
2. Create and set up conda environment
  - `cd accel-src`
  - `conda env create -p .conda-env -f environment.yml`
  - `conda activate ./.conda-env`
  - `pip install -e quantized-training`
  - `pip install xonsh deepdiff graphviz`
3. Install mflowgen
  - Clone mflowgen in your machine and check out `birch-intel16`: `git clone https://github.com/mflowgen/mflowgen.git -b birch-intel16`
  - Go back to project directory and run `pip install -e <path-to-mflowgen>`
4. Set up environment
  - Go back to `accelerator-pd` root directory, open `.envrc`
  - Line 2: set `MFLOWGEN_PATH` to the directory containing the adk.
  - run `source .envrc`
5. Hardcoded paths that may need to be modified
  - `accel-src/scripts/tech/intel16.tcl`: line 1 `stdcells` path.
  - `common/memory/run.sh`: all the intel16 memory paths.
6. Create build directory
  - Go to `Accelerator` directory, run `./build.xsh` to create the mflowgen build directory.
  - Run `./build.xsh --help` to get he list of options. E.g. `./build.xsh --datatpye E4M3 --ic 8 --clock 5`
  - Hint: If you don't plan to run simulation, add `--no-sweep` to disable parameter sweeping.

