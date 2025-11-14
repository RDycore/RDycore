#!/bin/sh

res=ELM_USRDAT
compset="2000_DATM%QIA_ELM%SPBC_SICE_SOCN_RDYCORE_SGLC_SWAV_SIAC_SESP"
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



elm_surfdata_file=surfdata_Houston1km_c250708.nc
elm_domain_file=domain_mid_atlantic_270x471_c250310.nc

elm_domain_file=domain_Dlnd_2926x1_c240507.nc
atm_domain_file=$elm_domain_file

rdycore_domain_file=domain.Houston1km_with_z_updated.nc
rdycore_file=Houston1km_with_z_updated.nc

l2r_map_file=map.Houston1km.ELM_to_RDycore.nc
r2l_map_file=map.Houston1km.RDycore_to_ELM.nc

rdycore_yaml_file=$PWD/Houston1km.CriticalBC.updated.yaml
rdycore_ic_file=Houston1km.IC.dat
rdycore_mesh_file=Houston1km_with_z_updated.exo

if [ "$mach" == "pm-cpu" ]; then 

  data_dir=/global/cfs/projectdirs/m4267/shared/data/harvey/Houston1km
  device="cpu"
  compiler=gnu

elif [ "$mach" == "pm-gpu" ]; then

  data_dir=/global/cfs/projectdirs/m4267/shared/data/harvey/Houston1km
  device="gpu"
  compiler=gnugpu

else
  echo "Unsupported machine specified via --mach $mach"
  display_help
  exit 0
fi

elm_data_dir=${data_dir}/e3sm
rdycore_data_dir=${data_dir}/rdycore

yr_start="2011"
yr_end="2011"
start_date="2011-08-26"

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Start creating and building the case
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 1. Create an E3SM case
src_dir=${e3sm_dir}

cd $src_dir
git_hash=`git log -n 1 --format=%h`
case_name=elm_rdycore.Houston1km.${mach}.${device}.${git_hash}.NTASKS_${ntasks}.${compiler}.`date "+%Y-%m-%d"`

case_dir=${src_dir}/cime/scripts

cd $case_dir

./create_newcase -case ${case_name} -res ${res} -mach ${mach} -compiler ${compiler} -compset ${compset} --project ${project-id}

cd $case_name

# Modify default settings
./xmlchange LND_DOMAIN_FILE=${elm_domain_file}
./xmlchange ATM_DOMAIN_FILE=${atm_domain_file}
./xmlchange LND_DOMAIN_PATH=${elm_data_dir}
./xmlchange ATM_DOMAIN_PATH=${elm_data_dir}

./xmlchange DATM_CLMNCEP_YR_END=1979
./xmlchange DATM_CLMNCEP_YR_START=1979
./xmlchange DATM_CLMNCEP_YR_ALIGN=1979

./xmlchange NTASKS=$ntasks
./xmlchange NTHRDS=1
./xmlchange PROJECT=$project_id

./xmlchange ATM_NCPL=96
./xmlchange ROF_NCPL=96

./xmlchange LND2ROF_FMAPNAME=$elm_data_dir/$l2r_map_file
./xmlchange ROF2LND_FMAPNAME=$elm_data_dir/$r2l_map_file

./xmlchange STOP_N=1,STOP_OPTION=nhours
./xmlchange JOB_QUEUE=debug,JOB_WALLCLOCK_TIME=00:30:00

cat >> user_nl_elm << EOF
fsurdat = '$elm_data_dir/$elm_surfdata_file'
EOF

cat >> user_nl_rdycore << EOF
filename_rof = '${elm_data_dir}/${rdycore_file}'
EOF

./case.setup --disable-git

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

cp $rdycore_yaml_file rdycore.yaml
ln -s $rdycore_data_dir/$rdycore_ic_file  .
ln -s $rdycore_data_dir/$rdycore_mesh_file .


cd ${case_dir}/${case_name}
./case.build --ninja

