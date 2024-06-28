# Overview

The E3SM-RDycore model has been tested on Perlmutter and Frontier for the RDycore's 5-day Hurricane Harvey benchmark. The E3SM-RDycore simulation uses a data-land configuration with an active river model. In an E3SM-RDycore run, RDycore can run on CPUs and GPUs. The overall workflow for an E3SM-RDycore run is as follows:

- Get the code:
    - Clone the E3SM fork from [https://github.com/rdycore/e3sm](https://github.com/rdycore/e3sm).
    - Switch to the E3SM-RDycore development branch and recursively initialize submodules.

- Create, build, and run a case
    1. Compile RDycore to generate libraries (i.e. `librdycore.a`, `librdycore_f90.a`, `libcyaml.a`, `libyaml.a`, and `libcmocka.a`)
    2. Create a E3SM case. Presently, the coupled model has been tested for a case with `--comspet RMOSGPCC --res MOS_USRDAT`.
    3. Before building the case, make the following modifications:
        - Modify the Macros file to add settings for PETSc and RDycore
        - Update the DLND streamfile (i.e `user_dlnd.streams.txt.lnd.gpcc`)
        - In `user_nl_mosart`, specify a few settings for MOSART including providing a placeholder MOSART file via `frivinp_rtm`
        - In `user_nl_dlnd`, specify a few settings for DLND including a map from the DLND mesh to the placeholder MOSART mesh
    4. Build the case
    5. Before submitting the case, do the following
        - In the rundir (`./xmlquerry RUNDIR`), copy or add symbolic links to a RDycore input YAML (as `rdycore.yaml`),
 any files specified in the RDycore's YAML file (e.g. mesh, initial condition), and map file to exchange data
 from the placeholder MOSART mesh to RDycore mesh.
        - Change the value of `run_exe` in the `env_mach_specific.xml` to include commandline options for PETSc and libCEED.
    6. Submit the case

The steps a-e have been automated via the shell script via [`e3sm_rdycore_harvey_flooding.sh`](e3sm_rdycore_harvey_flooding.sh).