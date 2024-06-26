# Overview

## Example for Perlmutter CPU nodes

```bash
./setup_harvey_flooding_ocean_bc.sh \
 --mach pm-cpu -N 1 --project m4267 \
--rdycore-dir /global/cfs/projectdirs/m4267/gbisht/rdycore
```

## Example for Perlmutter GPU nodes

```bash
./setup_harvey_flooding_ocean_bc.sh \
--mach pm-gpu -N 1 --project m4267_g \
--rdycore-dir /global/cfs/projectdirs/m4267/gbisht/rdycore 
```

## Example for Frontier using CPUs

```bash
./setup_harvey_flooding_ocean_bc.sh \
--mach frontier --frontier-node-type cpu -N 2 \
--project cli192 \
--rdycore-dir /lustre/orion/cli192/proj-shared/gb9/rdycore/rdycore 
```

## Example for Frontier using GPUs

```bash
./setup_harvey_flooding_ocean_bc.sh \
--mach frontier --frontier-node-type gpu -N 1 \
--project cli192 \
--rdycore-dir /lustre/orion/cli192/proj-shared/gb9/rdycore/rdycore 
```
