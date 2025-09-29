#!/bin/sh

res=ELM_USRDAT
compset=IELMBC
mach=
ros_event=
deltaT=
N=1
e3sm_dir=../../../../../../../../
project=m3520

# will be set autoamtically
compiler=
rdycore_dir=
atm_forcing_name=
atm_forcing_domain_file=

display_help() {
  echo "Usage: $0 " >&2
  echo
  echo "   -h, --help                              Display this message"
  echo "   --e3sm-dir                              Path to E3SM-RDycore directory"
  echo "   --mach <pm-cpu|pm-gpu|frontier>         Supported machine name"
  echo "   --frontier-node-type <cpu|gpu>          To run on Frontier CPUs or GPUs"
  echo "   -N, --node  <N>                         Number of nodes (default = 1)"
  echo "   --project-id <project-id>               Project ID that will charged for the job"
  echo "   --ros_event <1996PacN,1996MidA,2017CA>  Supported dataset name (i.e. daymet|imerg|mrms|mswep|nldas)"
  echo "   --deltaT <value>                        Temperature anomaly"
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
    --ros_event) ros_event="$2"; shift ;;
    --deltaT) deltaT="$2"; shift ;;
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

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Check command line arguments and set default values
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Check "--deltaT <value>"
if [ "$deltaT" = "" ]; then
    echo "Please specify the temperature anomaly using --deltaT <value>"
    display_help
    exit 0
elif [ "$deltaT" = "0" ]; then
    # Supported
    echo "Supported deltaT = " $deltaT
else
    echo "Unsupported deltaT = " $deltaT
    display_help
    exit 0
fi

#
# Determine the value of:
#  device
#  ntasks
#
device=""
if [ "$mach" == "pm-cpu" ]; then

  data_dir=/global/cfs/projectdirs/m4267/shared/data/ros
  device="cpu"
  ntasks=$((N*128))
  macros_file_in=${PWD}/../harvey-flooding/gnu_pm-cpu.cmake.pm-cpu-opt-32bit-gcc-11-2-0-fc2888174f5
  macros_file_out=gnu_pm-cpu.cmake
  compiler=gnu
  nldas_dir="/global/cfs/cdirs/e3sm/inputdata/atm/datm7/NLDAS"
  domain_dir="/global/cfs/cdirs/e3sm/inputdata/share/domains/domain.clm"

elif [ "$mach" == "pm-gpu" ]; then

  data_dir=/global/cfs/projectdirs/m4267/shared/data/ros
  device="gpu"
  ntasks=$((N*4))
  macros_file_in=${PWD}/../harvey-flooding/gnugpu_pm-gpu.cmake.pm-gpu-opt-32bit-gcc-11-2-0-fc2888174f5
  macros_file_out=gnugpu_pm-gpu.cmake
  compiler=gnugpu

elif [ "$mach" == "" ]; then

  echo "No machine specified via --mach"
  display_help
  exit 0

else

  echo "Unsupported machine specified via --mach $mach"
  display_help
  exit 0

fi

# Check if '--ros_event <name>' is supported
supported_event=0
if [ "$ros_event" = "1996PacN" ]; then
  supported_event=1

  yr_start="1996"
  yr_end="1996"
  start_date="1996-01-01"

  # E3SM-files that won't change if RDycore resolution is changed
  atm_forcing_name=atm_forcing.L15.PN.${deltaT}degree_P_2yrs
  atm_forcing_domain_file=domain_domain.lnd.nldas.PN_c231005.nc
  atm_forcing_dir="${data_dir}/e3sm/forcing_data/${atm_forcing_name}"
  domainFile=domain_ROS_1996_PN_c230427.nc
  domainPath=${data_dir}/e3sm/PN
  surfdataFile=surfdata_ROS_1996_PN_c230428_v2.nc
  finidatFile=ROS_1996_PN_FLOOD_Optimal_future_0K_P_spinup_20240909.elm.r.1996-01-01-00000.nc
  frivinp_rtm=${data_dir}/e3sm/PN1996_jigsaw_1km_90m/MOSART_PN1996_c241106.nc
  LND2ROF=${data_dir}/e3sm/PN1996_jigsaw_1km_90m/map_ELM_to_MOSART_PN1996.nc
  ROF2LND=${LND2ROF}

  # The following files will have to updated if RDycore resolution is changed
  RDYCORE_YAML_FILE=${data_dir}/rdycore/PN1996_jigsaw_1km_90m/PN1996_jigsaw_1km_90m.CriticalOutFlowBC.yaml
  RDYCORE_IC_FILE=${data_dir}/rdycore/PN1996_jigsaw_1km_90m/PN1996_jigsaw_1km_90m_manning.int32.bin
  RDYCORE_MESH_FILE=${data_dir}/rdycore/PN1996_jigsaw_1km_90m/PN1996_jigsaw_1km_90m.exo
  RDYCORE_BIN_MAP=${data_dir}/rdycore/PN1996_jigsaw_1km_90m/map_MOSART_to_RDycore_PN1996_jigsaw_1km_90m.int32.bin
fi

if [ "$supported_event" -eq 0 ]; then
    echo "The following ROS event is not supported: " $ros_event
    display_help
    exit 0
fi


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
case_name=ELM-RDycore.${mach}.${device}.${ros_event}_${deltaT}_${git_hash}.NTASKS_${ntasks}.${compiler}.`date "+%Y-%m-%d"`

# Create the case
cd ${src_dir}/cime/scripts
case_dir=${src_dir}/cime/scripts
./create_newcase -case ${case_dir}/${case_name} \
--res ${res} --mach ${mach} --compiler ${compiler} --compset ${compset} --project ${project_id}

# Change to case dir
cd ${case_dir}/${case_name}

# Modify default settings
./xmlchange LND_DOMAIN_FILE=${domainFile}
./xmlchange ATM_DOMAIN_FILE=${domainFile}
./xmlchange LND_DOMAIN_PATH=${domainPath}
./xmlchange ATM_DOMAIN_PATH=${domainPath}

./xmlchange NTASKS=$ntasks
./xmlchange DATM_MODE=CLMMOSARTTEST
./xmlchange DATM_CLMNCEP_YR_START=${yr_start}
./xmlchange DATM_CLMNCEP_YR_END=${yr_end}
./xmlchange RUN_STARTDATE=${start_date}

./xmlchange ROF_NCPL=48
./xmlchange LND2ROF_FMAPNAME=$LND2ROF
./xmlchange ROF2LND_FMAPNAME=$ROF2LND

./xmlchange JOB_QUEUE=debug
./xmlchange JOB_WALLCLOCK_TIME=00:30:00
./xmlchange STOP_N=1
./xmlchange STOP_OPTION=ndays

cat >> user_nl_elm << EOF
 fsurdat = '${domainPath}/${surfdataFile}'
 finidat = '${domainPath}/${finidatFile}'
EOF

cat >> user_nl_mosart << EOF

&mosart_inparm
 barrier_timers = .false.
 coupling_period = 1800
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
 rtmhist_nhtfrq = 1
 rtmhist_mfilt  = 240
 heatflag = .false.
 ice_runoff = .true.
 inundflag = .false.
 ngeom = 50
 nlayers = 30
 rinittemp = 283.15
 routingmethod = 1
 rstraflag = .false.
 rtmhist_mfilt = 240
 rtmhist_ndens = 2
 rtmhist_nhtfrq = 1
 sediflag = .false.
 smat_option = 'Xonly'
 wrmflag = .false. 
/

EOF

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


cp ./CaseDocs/datm.streams.txt.CLMMOSARTTEST ./user_datm.streams.txt.CLMMOSARTTEST
chmod +rw ./user_datm.streams.txt.CLMMOSARTTEST
sed -i "s/${domain_dir//\//\\/}/${atm_forcing_dir//\//\\/}/g" ./user_datm.streams.txt.CLMMOSARTTEST
sed -i "s/domain.lnd.nldas2_0224x0464_c110415.nc/${atm_forcing_domain_file}g" ./user_datm.streams.txt.CLMMOSARTTEST
sed -i "s/${nldas_dir//\//\\/}/${atm_forcing_dir//\//\\/}/g" ./user_datm.streams.txt.CLMMOSARTTEST
sed -i "s@clmforc.nldas/elmforc.L15.PN.${deltaT}degreeg" ./user_datm.streams.txt.CLMMOSARTTEST
sed -i '/ZBOT/d' ./user_datm.streams.txt.CLMMOSARTTEST

rundir=`./xmlquery RUNDIR --value`

cd $rundir

cp $RDYCORE_YAML_FILE    rdycore.yaml
ln -s $RDYCORE_IC_FILE   .
ln -s $RDYCORE_MESH_FILE .
ln -s $RDYCORE_BIN_MAP   map_MOSART_to_RDycore.bin


cd ${case_dir}/${case_name}
./case.build

