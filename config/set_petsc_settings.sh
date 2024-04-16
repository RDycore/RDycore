#!/bin/sh

pm_node=
mach=
with64bit=0
with_debugging=0

display_help() {
    echo "Usage: $0 " >&2
    echo
    echo "   -h, --help             Display this message"
    echo "   --pm <cpu|gpu>         Type of Perlmutter nodes (cpu or gpu)"
    echo "   --64bit                With 64bit support (optional)"
    echo "   --with-debugging <0|1> With or without debugging version (optional)"
    echo
    return 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

while [ $# -gt 0 ]
do
  case "$1" in
    --pm ) pm_node="$2"; shift ;;
    --64bit ) with64bit=1 ;;
    ----with-debugging) with_debugging="$2"; shift ;;
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

# Determine the system
if [ ! -z "$LMOD_SYSTEM_NAME" ]; then
  if [[ $LMOD_SYSTEM_NAME == *"perlmutter"* ]]; then
    mach=pm
  elif [[ $LMOD_SYSTEM_NAME == *"frontier"* ]]; then
    mach=frontier
  fi
fi

if [ ! -z "$HOST" ]; then
  if [[ $HOST == *"aurora"* ]]; then
     mach=aurora
  fi
fi


if [ "$mach" = "pm" ]; then

  if [[ -z "$pm_node" ]]; then
    echo ""
    echo "NOTE: The -pm <cpu|gpu> was not specified, so assuming CPU nodes are being used."
    echo ""
    pm_node=cpu
  fi

  if [ "$pm_node"  = "cpu" ]; then

    MODULE_FILE=$DIR/modules.pm-cpu.gnu
    export PETSC_DIR=/global/cfs/projectdirs/m4267/petsc/petsc_main/
    if [ "$with64bit" -eq 0 ]; then
      if [ "$with_debugging" -eq 0 ]; then
        export PETSC_ARCH=pm-cpu-opt-32bit-gcc-11-2-0-fc2888174f5
      else
        export PETSC_ARCH=pm-cpu-debug-32bit-gcc-11-2-0-fc2888174f5
      fi
    else
      if [ "$with_debugging" -eq 0 ]; then
        export PETSC_ARCH=pm-cpu-opt-64bit-gcc-11-2-0-fc2888174f5
      else
        export PETSC_ARCH=pm-cpu-debug-64bit-gcc-11-2-0-fc2888174f5
      fi
    fi

  elif [ "$pm_node" = "gpu" ]; then

    MODULE_FILE=$DIR/modules.pm-gpu.gnugpu
    export PETSC_DIR=/global/cfs/projectdirs/m4267/petsc/petsc_main/
    if [ "$with64bit" -eq 0 ]; then
      if [ "$with_debugging" -eq 0 ]; then
        export PETSC_ARCH=pm-gpu-opt-32bit-gcc-11-2-0-fc2888174f5
      else
        export PETSC_ARCH=pm-gpu-debug-32bit-gcc-11-2-0-fc2888174f5
      fi
    else
      if [ "$with_debugging" -eq 0 ]; then
        export PETSC_ARCH=pm-gpu-opt-64bit-gcc-11-2-0-fc2888174f5
      else
        export PETSC_ARCH=pm-gpu-debug-64bit-gcc-11-2-0-fc2888174f5
      fi
    fi

  else
    echo "The only supported options for -pm are cpu or gpu."
    display_help
    exit 0
  fi

elif [ "$mach" = "frontier"  ]; then

  MODULE_FILE=$DIR/modules.frontier.gnugpu
  export PETSC_DIR=/lustre/orion/cli192/proj-shared/petsc
  if [ "$with64bit" -eq 0 ]; then
    if [ "$with_debugging" -eq 0 ]; then
      export PETSC_ARCH=frontier-gpu-opt-32bit-gcc-11-2-0-fc288817
    else
      export PETSC_ARCH=frontier-gpu-debug-32bit-gcc-11-2-0-fc288817
    fi
  else
    if [ "$with_debugging" -eq 0 ]; then
      export PETSC_ARCH=frontier-gpu-opt-64bit-gcc-11-2-0-fc288817
    else
      # Got installation error
      #export PETSC_ARCH=frontier-gpu-debug-64bit-gcc-11-2-0-fc288817
       echo "On Frontier, PETSc with 64bit and debugging turned on has not been installed."
       exit 0
    fi
  fi

  if [[ ! -z "$pm_node" ]]; then
     echo "The --pm_node <$pm_node> was specified, which is applicable is only for Perlmutter,"
     echo "but the machine detected is Frontier."
     exit 0
  fi

elif [ "$mach" = "aurora"  ]; then

  MODULE_FILE=$DIR/modules.aurora.oneapi
  export PETSC_DIR=/lus/gecko/projects/CSC250STMS07_CNDA/bishtgautam/petsc
  if [ "$with64bit" -eq 0 ]; then
    if [ "$with_debugging" -eq 0 ]; then
      export PETSC_ARCH=aurora-opt-32bit-oneapi-ifx-fc288817
    else
       echo "On Aurora, --with-debugging 1 was selected, but PETSc has not been installed with debugging turned on."
       exit 0
    fi
  else
    if [ "$with_debugging" -eq 0 ]; then
      export PETSC_ARCH=aurora-opt-64bit-oneapi-ifx-fc288817
    else
       echo "On Aurora, --with-debugging 1 was selected, but PETSc has not been installed with debugging turned on."
       exit 0
    fi
  fi

  if [[ ! -z "$pm_node" ]]; then
     echo "The --pm_node <$pm_node> was specified, which is applicable is only for Perlmutter,"
     echo "but the machine detected is Aurora."
     exit 0
  fi

else
  echo "Could not determine the machine. mach=$mach"
  display_help
  exit 0
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

