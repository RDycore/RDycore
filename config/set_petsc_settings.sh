#!/bin/sh

mach=
with64bit=0

display_help() {
    echo "Usage: $0 " >&2
    echo
    echo "   -h, --help             Display this message"
    echo "   --mach <machine_name>  Machine name (pm-cpu, pm-gpu, frontier)"
    echo "   --64bit                With 64bit support (optional)"
    echo
    exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

while [ $# -gt 0 ]
do
  case "$1" in
    --mach ) mach="$2"; shift ;;
    --64bit ) with64bit=1; shift ;;
    -*)
      display_help
      exit 0
      ;;
    -h | --help)
      display_help
      exit 0
      ;;
    *)  break;;    # terminate while loop
  esac
  shift
done

if [ "$mach" = "pm-cpu" ]; then

  MODULE_FILE=$DIR/modules.pm-cpu.gnu
  export PETSC_DIR=/global/cfs/projectdirs/m4267/petsc/petsc_main/
  if [ "$with64bit" -eq 0 ]; then
     export PETSC_ARCH=pm-cpu-opt-32bit-gcc-11-2-0-fc2888174f5
  else
     export PETSC_ARCH=pm-cpu-opt-64bit-gcc-11-2-0-fc2888174f5
  fi

elif [ "$mach" = "pm-gpu" ]; then

  MODULE_FILE=$DIR/modules.pm-gpu.gnugpu
  export PETSC_DIR=/global/cfs/projectdirs/m4267/petsc/petsc_main/
  if [ "$with64bit" -eq 0 ]; then
     export PETSC_ARCH=pm-gpu-opt-32bit-gcc-11-2-0-fc2888174f5
  else
     export PETSC_ARCH=pm-gpu-opt-64bit-gcc-11-2-0-fc2888174f5
  fi

elif [ "$mach" = "frontier"  ]; then

  MODULE_FILE=$DIR/modules.frontier.gnugpu
  export PETSC_DIR=/lustre/orion/cli192/proj-shared/petsc
  if [ "$with64bit" -eq 0 ]; then
     export PETSC_ARCH=frontier-gpu-opt-32bit-gcc-11-2-0-fc288817
  else
     export PETSC_ARCH=frontier-gpu-opt-32bit-gcc-11-2-0-fc288817
  fi
else

  echo "Unsupported machine."
  display_help
  exit 1
fi

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Will source the following module file and set the following PETSc settings"
echo ""
echo "  source $MODULE_FILE"
echo "  export PETSC_DIR=$PETSC_DIR"
echo "  export PETSC_ARCH=$PETSC_ARCH"
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
source $MODULE_FILE
export PETSC_DIR=$PETSC_DIR
export PETSC_ARCH=$PETSC_ARCH

