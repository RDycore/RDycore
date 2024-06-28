# E3SM-RDycore

The coupled E3SM-RDycore is developed within a fork of the E3SM repository under the RDycore's Github account at [https://github.com/rdycore/e3sm](https://github.com/rdycore/e3sm). The coupled model is developed in a branch within the forked repository and the current branch is named `rdycore/mosart-rdycore/8b4c2d5df3-2024-04-05`. The `8b4c2d5df3` corresponds to the Git hash of the commit on the E3SM master branch from which the current E3SM-RDycore development branch started and `2024-04-05` corresponds to the date of that starting commit.

```text
# From E3SM repo
>git show 8b4c2d5df3
commit 8b4c2d5df3f0a53a2ea49bc0c63c2c7f07bcadd4
Merge: 3b09ee8c5c 686b5c1689
Author: James Foucar <jgfouca@sandia.gov>
Date:   Fri Apr 5 09:05:36 2024 -0600

 Merge branch 'jgfouca/rm_gnu9' into master (PR #6328)

 Remove gnu9 1-off

 This was a 1-off for mappy that needed two gnu toolchains. Now that we
 have 11.2, we don't need this.

 [BFB]

(END)
```

The E3SM-RDycore development branch is infrequently rebased on E3SM's master. After a rebase, the E3SM-RDycore development branch would named such that the new name correctly represents the starting commit hash and the commit date. The RDycore has been added in E3SM as a submodule at
`externals/rdycore`. In the current model coupling, RDycore is part of the MOSART as shown below.

![image](e3sm-rdycore-via-mosart.png)

The E3SM-RDycore model has been tested on Perlmutter and Frontier for the RDycore's 5-day Hurricane Harvey benchmark. The E3SM-RDycore simulation uses a data-land configuration with an active river model. In an E3SM-RDycore run, RDycore can run on CPUs and GPUs.

The overall workflow for an E3SM-RDycore run is as follows:

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

The steps 1-5 have been automated via