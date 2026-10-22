#!/bin/sh

res=RDY_USRDAT
compset="2000_DATM%QIA_DLND%GPCC_SICE_SOCN_RDYCORE_SGLC_SWAV"
mach=
project_id=
N=1
e3sm_dir=../../../../../../../

# will be set autoamtically
compiler=

display_help() {
  echo "Usage: $0 " >&2
  echo
  echo "   -h, --help                        Display this message"
  echo "   --e3sm-dir                        Path to E3SM-RDycore directory"
  echo "   --mach <pm-cpu|pm-gpu|frontier>   Supported machine name"
  echo "   --frontier-node-type <cpu|gpu>    To run on Frontier CPUs or GPUs"
  echo "   -N, --node  <N>                   Number of compute nodes (default = 1)"
  echo "   --project-id <project-id>         Project ID that will charged for the job"
  return 0
}  

while [ $# -gt 0 ]
do
  case "$1" in
    --mach ) mach="$2"; shift ;;
    --frontier-node-type) frontier_node_type="$2"; shift ;;
    --project-id) project_id="$2"; shift ;;
    --e3sm-dir) e3sm_dir="$2"; shift ;;
    -N | --node) N="$2"; shift ;;
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

#
# Determine the value of:
#  device
#  ntasks
#
device=""
if [ "$mach" == "pm-cpu" ]; then
  data_dir=/global/cfs/projectdirs/m4267/shared/data/harvey/Houston1km
  grfr_data_dir=/global/cfs/projectdirs/m4267/shared/data/grfr/dlnd
  device="cpu"
  ntasks=$((N*128))
  compiler=gnu
  dlnd_streams_file=$PWD/user_dlnd.streams.txt.lnd.gpcc.Perlmutter
elif [ "$mach" == "pm-gpu" ]; then
  data_dir=/global/cfs/projectdirs/m4267/shared/data/harvey/Houston1km
  grfr_data_dir=/global/cfs/projectdirs/m4267/shared/data/grfr/dlnd
  device="gpu"
  ntasks=$((N*4))
  compiler=gnugpu
  dlnd_streams_file=$PWD/user_dlnd.streams.txt.lnd.gpcc.Perlmutter
elif [ "$mach" == "frontier" ]; then

  data_dir=/lustre/orion/cli192/proj-shared/data/harvey
  compiler=gnugpu

  # Make sure both CPU and GPU options were not specified
  if [ "$frontier_node_type" == "cpu" ]
  then
    device="cpu"
    ntasks=$((N*56))
  elif [ "$frontier_node_type" == "gpu" ]
  then
    device="gpu"
    mach_prefix=".gpu"
    ntasks=$((N*8))
  else
    echo "Unknown -frontier-node-type $frontier_node_type"
    display_help
    exit 0
  fi

elif [ "$mach" == "" ]; then
  echo "No machine specified via --mach"
  display_help
  exit 0
else
  echo "Unsupported machine specified via --mach $mach"
  display_help
  exit 0
fi

elm_data_dir=${data_dir}/e3sm
rdycore_data_dir=${data_dir}/rdycore

domain_file=domain_Dlnd_2926x1_c240507.nc
rdycore_file=Houston1km_with_z_updated.nc

rdycore_yaml_file=$PWD/Houston1km.CriticalBC.updated.yaml
rdycore_ic_file=${rdycore_data_dir}//Houston1km.IC.dat
rdycore_mesh_file=${rdycore_data_dir}/Houston1km_with_z_updated.exo

domain_file=domain.Houston1km_with_z_updated.nc
domain_path=${elm_data_dir}
map_file=${elm_data_dir}/map.GRFR_to_Houston1km.v2.nc

src_dir=${e3sm_dir}
case_dir=${src_dir}/cime/scripts

# Determine the case name
cd $src_dir
git_hash=`git log -n 1 --format=%h`
case_name=Dlnd.${mach}.${device}.Houston1km.GRFR.${git_hash}.NTASKS_${ntasks}.${compiler}.`date "+%Y-%m-%d"`

# Create the case
cd cime/scripts
case_dir=${src_dir}/cime/scripts
./create_newcase -case ${case_dir}/${case_name} --res ${res} --mach ${mach} --compiler ${compiler} --compset ${compset} --project ${project_id}

# Change to case dir
cd ${case_name}

./xmlchange CALENDAR=NO_LEAP

./xmlchange LND_DOMAIN_FILE=$domain_file
./xmlchange ATM_DOMAIN_FILE=$domain_file
./xmlchange LND_DOMAIN_PATH=$domain_path
./xmlchange ATM_DOMAIN_PATH=$domain_path

./xmlchange DATM_CLMNCEP_YR_END=2001
./xmlchange DATM_CLMNCEP_YR_START=2001
./xmlchange DATM_CLMNCEP_YR_ALIGN=2001
./xmlchange DLND_CPLHIST_YR_START=2001
./xmlchange DLND_CPLHIST_YR_END=2001
./xmlchange DLND_CPLHIST_YR_ALIGN=2001
./xmlchange RUN_STARTDATE=2001-01-01

./xmlchange NTASKS=$ntasks
./xmlchange STOP_N=2,STOP_OPTION=nhours
./xmlchange PIO_NETCDF_FORMAT=classic
./xmlchange ROF_NCPL=24

cat >> user_nl_rdycore << EOF
filename_rof = '${elm_data_dir}/${rdycore_file}'
EOF

cat >> user_nl_dlnd << EOF
dtlimit=2.0e0
mapalgo = "nn"
mapread = "$map_file"
EOF

cp ${dlnd_streams_file} user_dlnd.streams.txt.lnd.gpcc

./case.setup

if [ "$mach" == "pm-cpu" ]; then
  ./xmlchange run_exe="\${EXEROOT}/e3sm.exe -ceed /cpu/self -log_view"
elif [ "$mach" == "pm-gpu" ]; then
  ./xmlchange run_exe="-G4 \${EXEROOT}/e3sm.exe -ceed /gpu/cuda -dm_vec_type cuda -use_gpu_aware_mpi 1 -log_view -log_view_gpu_time"
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

cp $rdycore_yaml_file    rdycore.yaml
ln -s $rdycore_ic_file   .
ln -s $rdycore_mesh_file .

cd ${case_dir}/${case_name}
./case.build --ninja

