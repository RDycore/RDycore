#!/bin/sh

res=ELM_USRDAT
compset=IELMBC
mach=
compiler=
project_id=
N=1
e3sm_dir=$PWD/../../../../../../../

display_help() {
  echo "Usage: $0 " >&2
  echo
  echo "   -h, --help                              Display this message"
  echo "   --e3sm-dir                              Path to E3SM-RDycore directory"
  echo "   --mach <pm-cpu|pm-gpu|frontier>         Supported machine name"
  echo "   --frontier-node-type <cpu|gpu>          To run on Frontier CPUs or GPUs"
  echo "   -N, --node  <N>                         Number of nodes (default = 1)"
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


elm_delaware_id=270x471

elm_surfdata_file=surface_dataset_mid_atlantic_270x471_c250310.nc
elm_domain_file=domain_mid_atlantic_270x471_c250310.nc
mosart_file=MOSART_delaware_113505_c250311.nc
e2m_map_file=map_ELM_to_MOSART_delaware_113505_c250311.nc
m2e_map_file=map_MOSART_to_ELM_delaware_113505_c250311.nc

m2r_map_file=map_MOSART_to_RDycore_delaware_113505.int32.bin
rdycore_yaml_file=Delaware.CriticalOutflowBC.yaml
rdycore_ic_file=delaware.ic.int32.bin
rdycore_mesh_file=delaware.exo

m2r_map_file=map_MOSART_to_RDycore_delaware_30m_113505.int32.bin
rdycore_yaml_file=Delaware_30m.OceanDirichletBC.yaml
rdycore_ic_file=delaware_30m.ic.int32.bin
rdycore_mesh_file=delaware_30m.v3.1.0.h5
rdycore_mannings_file=delaware_30m_manning.int32.bin

if [ "$mach" == "pm-cpu" ]; then 

  data_dir=/global/cfs/projectdirs/m4267/shared/data/irene/delaware
  device="cpu"
  ntasks=$((N*128))
  macros_file_in=${PWD}/../harvey-flooding/gnu_pm-cpu.cmake.pm-cpu-opt-32bit-gcc-11-2-0-fc2888174f5
  macros_file_out=gnu_pm-cpu.cmake
  compiler=gnu

elif [ "$mach" == "pm-gpu" ]; then

  data_dir=/global/cfs/projectdirs/m4267/shared/data/irene/delaware
  device="gpu"
  ntasks=$((N*4))
  macros_file_in=${PWD}/../harvey-flooding/gnugpu_pm-gpu.cmake.pm-gpu-opt-32bit-gcc-11-2-0-fc2888174f5
  macros_file_out=gnugpu_pm-gpu.cmake
  compiler=gnugpu

else
  echo "Unsupported machine specified via --mach $mach"
  display_help
  exit 0
fi

elm_data_dir=${data_dir}/e3sm/elm-mosart/${elm_delaware_id}
frivinp_rtm=${elm_data_dir}/$mosart_file

datm_cplhist_dir=${data_dir}/e3sm/icom-data/t44_v1.2-NATL-F2010C5-v2_L71_forJonDarren.2011082600.ens012
map_dir=${data_dir}/e3sm/icom-data/icom-data/datm_Jim_ens012

rdycore_data_dir=${data_dir}/rdycore

yr_start="2011"
yr_end="2011"
start_date="2011-08-26"

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

  cmake -S . -B build-$PETSC_ARCH -DCMAKE_INSTALL_PREFIX=$PWD/build-$PETSC_ARCH
fi

# Build 
cd build-$PETSC_ARCH
make -j4 install

# 2. Create an E3SM case

cd $src_dir
git_hash=`git log -n 1 --format=%h`
case_name=delaware_${elm_delaware_id}.${mach}.${device}.${git_hash}.NTASKS_${ntasks}.${compiler}.`date "+%Y-%m-%d"`

case_dir=${src_dir}/cime/scripts

cd $case_dir

./create_newcase -case ${case_name} -res ${res} -mach ${mach} -compiler ${compiler} -compset ${compset} --project ${project-id}

cd $case_name

# Modify default settings
./xmlchange LND_DOMAIN_FILE=${elm_domain_file}
./xmlchange ATM_DOMAIN_FILE=${elm_domain_file}
./xmlchange LND_DOMAIN_PATH=${elm_data_dir}
./xmlchange ATM_DOMAIN_PATH=${elm_data_dir}

./xmlchange DATM_MODE=CPLHIST
./xmlchange DATM_CPLHIST_YR_ALIGN=${yr_start}
./xmlchange DATM_CPLHIST_YR_START=${yr_start}
./xmlchange DATM_CPLHIST_YR_END=${yr_start}
./xmlchange DATM_CPLHIST_DIR=${datm_cplhist_dir}
./xmlchange DATM_CPLHIST_DOMAIN_FILE=${elm_data_dir}/${elm_domain_file}

./xmlchange PIO_VERSION=2
./xmlchange PIO_TYPENAME=pnetcdf
./xmlchange PIO_NETCDF_FORMAT=64bit_offset
./xmlchange NTASKS=$ntasks
./xmlchange NTHRDS=1
./xmlchange PROJECT=$project_id

./xmlchange ATM_NCPL=96
./xmlchange ROF_NCPL=96

./xmlchange LND2ROF_FMAPNAME=$elm_data_dir/$e2m_map_file
./xmlchange ROF2LND_FMAPNAME=$elm_data_dir/$m2e_map_file

cat >> user_nl_elm << EOF
fsurdat = '$elm_data_dir/$elm_surfdata_file'
use_top_solar_rad  = .true.
use_modified_infil = .true.
hist_empty_htapes=.TRUE.
hist_fincl1 = 'QOVER','QRUNOFF','RAIN','FSAT','FH2OSFC','QDRAI'
hist_nhtfrq = -1
hist_mfilt = 24
EOF

cat << EOF >> user_nl_datm
  dtlimit = 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30
  mapalgo = "nn", "nn", "nn", "nn", "nn", "nn"
  mapread = "map.${elm_delaware_id}.Solar.nc", "map.${elm_delaware_id}.nonSolarFlux.nc", "map.${elm_delaware_id}.State3hr.nc", "map.${elm_delaware_id}.State1hr.nc", "map.${elm_delaware_id}.presaero.nc", "map.${elm_delaware_id}.topo.nc"
EOF

cat >> user_nl_mosart << EOF
&mosart_inparm
 barrier_timers = .false.
 coupling_period = 900
 data_bgc_fluxes_to_ocean_flag = .false.
 delt_mosart = 1800
 dlevelh2r = 5
 dlevelr = 3
 do_budget = 0
 do_rtm = .true.
 frivinp_rtm  = '$frivinp_rtm'
 decomp_option  = 'rdycore'
 routingmethod  = 1
 frivinp_mesh   = ''
 heatflag = .false.
 ice_runoff = .true.
 inundflag = .false.
 ngeom = 50
 nlayers = 30
 rinittemp = 283.15
 routingmethod = 1
 rstraflag = .false.
 rtmhist_ndens = 2
 sediflag = .false.
 smat_option = 'Xonly'
 wrmflag = .false. 
/
EOF

cp ${data_dir}/e3sm/icom-data/datm_Jim_ens012/user_datm.* ./

./case.setup

# Modify Macros file
cp ${macros_file_in} cmake_macros/${macros_file_out}

sed -i "s/PLACEHOLDER_E3SM_DIR/${e3sm_dir//\//\\/}/g" cmake_macros/${macros_file_out}
sed -i "s/PLACEHOLDER_PETSC_DIR/${PETSC_DIR//\//\\/}/g" cmake_macros/${macros_file_out}
sed -i "s/PLACEHOLDER_PETSC_ARCH/${PETSC_ARCH}/g" cmake_macros/${macros_file_out}

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

cp $rdycore_data_dir/inputdeck/$rdycore_yaml_file rdycore.yaml
ln -s $rdycore_data_dir/ic/$rdycore_ic_file  .
ln -s $rdycore_data_dir/mesh/$rdycore_mesh_file .
ln -s $rdycore_data_dir/mannings/$rdycore_mannings_file .
ln -s $elm_data_dir/$m2r_map_file map_MOSART_to_RDycore.bin

cp $elm_data_dir/map.${elm_delaware_id}.*.nc

cd ${case_dir}/${case_name}
./case.build

