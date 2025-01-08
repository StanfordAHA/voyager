mflowgen-based physical design flow for [DNN accelerator](https://code.stanford.edu/tsmc40r/brainpower/accelerator).

## Instructions

1. `git submodule update --init --recursive`
2. Create and set up conda environment

   ```bash
   $ cd accel-src
   $ conda env create -p .conda-env -f accel-src/environment.yml
   $ conda activate ./.conda-env
   $ pip install -e accel-src/quantized-training
   $ pip install xonsh deepdiff graphviz
   $ pip install -r requirements.txt
   ```
3. Set up environment

   ```
   $ source env.sh
   ```

4. Create build directory

- Go to `Accelerator` directory, run `./build.xsh` to create the mflowgen build directory.
- Run `./build.xsh --help` to get he list of options. E.g. `./build.xsh --datatpye E4M3 --ic 8 --clock 5`
- Hint: If you don't plan to run simulation, add `--no-sweep` to disable parameter sweeping.

## Setting up a technology
1. Add a softlink to the ADK folder under `technology/adks/`
2. Create a .yml configuration file under `technology/` for your technology (if it doesn't already exist)
