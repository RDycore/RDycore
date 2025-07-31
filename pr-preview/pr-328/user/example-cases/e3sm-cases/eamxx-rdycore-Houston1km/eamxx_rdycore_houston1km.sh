#!/bin/sh

res=ne4_ne4
compset="2010_SCREAM_ELM%SPBC_CICE%PRES_DOCN%DOM_RDYCORE_SGLC_SWAV_SIAC_SESP"
mach=
compiler=
project_id=
ntasks=1
e3sm_dir=$PWD/../../../../../../../

display_help() {
  echo "Usage: $0 " >&2
  echo
  echo "   -h, --help                              Display this message"
  echo "   --e3sm-dir                              Path to E3SM-RDycore directory"
  echo "   --mach <pm-cpu|pm-gpu|frontier>         Supported machine name"
  echo "   --frontier-node-type <cpu|gpu>          To run on Frontier CPUs or GPUs"
  echo "   --ntasks  <NTASKS>                      Number of MPI tasks"
  echo "   --project-id <project-id>               Project ID that will charged for the job"
  return 0
}

while [ $# -gt 0 ]
do
  case "$1" in
    --mach ) mach="$2"; shift ;;
    --frontier-node-type) frontier_node_type="$2"; shift ;;
    --project-id) project_id="$2"; shift ;;
    --e3sm-dir) e3sm_dir="$2"; shift ;;
    --ntasks) ntasks="$2"; shift ;;
    -h | --help)
      display_help
      exit 0
      ;;
    -*)
      echo "Unsupported argument: $1"
      display_help
      exit 0
      ;;
    *)  break;;    # terminate while loop
  esac
  shift
done

rdycore_domain_file=domain.Houston1km_with_z_updated.nc
rdycore_file=Houston1km_with_z_updated.nc

l2r_map_file=map.ne4np4_to_Houston1km_RDycore.nc
r2l_map_file=map.Houston1km_RDycore_to_ne4np4.nc

rdycore_yaml_file=Houston1km.CriticalBC.updated.yaml
rdycore_ic_file=Houston1km.IC.dat
rdycore_mesh_file=Houston1km_with_z_updated.exo

if [ "$mach" == "pm-cpu" ]; then

  data_dir=/global/cfs/projectdirs/m4267/shared/data/harvey/Houston1km
  device="cpu"
  macros_file_in=${PWD}/gnu_pm-cpu.cmake.pm-cpu-opt-32bit-gcc-13-2-1-95934b0d393
  macros_file_out=gnu_pm-cpu.cmake
  compiler=gnu

elif [ "$mach" == "pm-gpu" ]; then

  data_dir=/global/cfs/projectdirs/m4267/shared/data/harvey/Houston1km
  device="gpu"
  macros_file_in=${PWD}/gnugpu_pm-gpu.cmake.pm-gpu-opt-32bit-gcc-13-2-1-95934b0d393
  macros_file_out=gnugpu_pm-gpu.cmake
  compiler=gnugpu

else
  echo "Unsupported machine specified via --mach $mach"
  display_help
  exit 0
fi

elm_data_dir=${data_dir}/e3sm
rdycore_data_dir=${data_dir}/rdycore

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Start creating and building the case
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Let's build RDycore
src_dir=${e3sm_dir}
rdycore_dir=$e3sm_dir/externals/rdycore/

cd $rdycore_dir

source config/set_petsc_settings.sh --mach $mach --config 3

if [ ! -d "$rdycore_dir/build-$PETSC_ARCH" ]
then
  echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  echo "The following expected RDycore build directory not found:$rdycore_dir/build-$PETSC_ARCH "
  echo "So, attempting to build RDycore."
  echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

  cmake -S . -B build-$PETSC_ARCH -DCMAKE_INSTALL_PREFIX=$PWD/build-$PETSC_ARCH -G Ninja
fi

# Build
cd build-$PETSC_ARCH
ninja -j4 install

# 2. Create an E3SM case

cd $src_dir
git_hash=`git log -n 1 --format=%h`
case_name=eamxx_rdycore.Houston1km.${mach}.${device}.${git_hash}.NTASKS_${ntasks}.${compiler}.`date "+%Y-%m-%d"`

case_dir=${src_dir}/cime/scripts

cd $case_dir

./create_newcase -case ${case_name} -res ${res} -mach ${mach} -compiler ${compiler} -compset ${compset} --project ${project_id}

cd $case_name

# Modify default settings
./xmlchange NTASKS=$ntasks
./xmlchange NTHRDS=1
./xmlchange PROJECT=$project_id

./xmlchange ATM_NCPL=96
./xmlchange ROF_NCPL=96

./xmlchange LND2ROF_FMAPNAME=$elm_data_dir/$l2r_map_file
./xmlchange ROF2LND_FMAPNAME=$elm_data_dir/$r2l_map_file

./xmlchange JOB_QUEUE=debug,JOB_WALLCLOCK_TIME=00:30:00

cat >> user_nl_rdycore << EOF
filename_rof = '${elm_data_dir}/${rdycore_file}'
EOF

./case.setup --disable-git

# Modify Macros file
cp ${macros_file_in} cmake_macros/${macros_file_out}

sed -i "s/PLACEHOLDER_E3SM_DIR/${e3sm_dir//\//\\/}/g" cmake_macros/${macros_file_out}
sed -i "s/PLACEHOLDER_PETSC_DIR/${PETSC_DIR//\//\\/}/g" cmake_macros/${macros_file_out}
sed -i "s/PLACEHOLDER_PETSC_ARCH/${PETSC_ARCH}/g" cmake_macros/${macros_file_out}

if [ "$mach" == "pm-cpu" ]; then
  ./xmlchange run_exe="\${EXEROOT}/e3sm.exe -ceed /cpu/self -log_view"
elif [ "$mach" == "pm-gpu" ]; then
  ./xmlchange run_exe="-G${ntasks} \${EXEROOT}/e3sm.exe -ceed /gpu/cuda -dm_vec_type cuda -use_gpu_aware_mpi 1 -log_view -log_view_gpu_time"
elif [ "$mach" == "frontier" ]; then
  # Make sure both CPU and GPU options were not specified
  if [ "$frontier_node_type" == "cpu" ]
  then
    ./xmlchange run_exe="\${EXEROOT}/e3sm.exe -ceed /cpu/self -log_view"
  elif [ "$frontier_node_type" == "gpu" ]
  then
    ./xmlchange run_exe="\${EXEROOT}/e3sm.exe -ceed /gpu/hip -dm_vec_type hip -use_gpu_aware_mpi 0 -log_view -log_view_gpu_time"
  fi
else
  echo "Unsupported machine specified via --mach $mach"
  display_help
  exit 0
fi

rundir=`./xmlquery RUNDIR --value`

cd $rundir

cp $rdycore_data_dir/$rdycore_yaml_file rdycore.yaml
ln -s $rdycore_data_dir/$rdycore_ic_file  .
ln -s $rdycore_data_dir/$rdycore_mesh_file .


cd ${case_dir}/${case_name}

export Kokkos_ROOT=$PETSC_DIR/$PETSC_ARCH

./case.build --ninja

