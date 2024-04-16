! This module was adapted from https://github.com/RDycore/E3SM/blob/bishtgautam/mosart-rdycore/78a5ea1613-2023-08-07/components/mosart/src/rdycore/rdycoreMod.F90

module rdycoreMod

#include <petsc/finclude/petsc.h>

  use petsc
  use rdycore

  implicit none

  private

  type(RDy)          :: rdy_


  PetscInt           :: num_cells_owned
  PetscReal, pointer :: rain_data(:)
  PetscReal          :: dtime ! in units expressed within config file
  PetscInt           :: nstep

  public :: rdycore_usage
  public :: rdycore_init
  public :: rdycore_run
  public :: rdycore_final

  public :: dtime, nstep

contains

  !-----------------------------------------------------------------------
  subroutine rdycore_usage()
    print *, "test_coupling: usage:"
    print *, "test_coupling <input.yaml>"
    print *, ""
  end subroutine

  !-----------------------------------------------------------------------
  subroutine rdycore_init(config_file)
    !
    ! !DESCRIPTION:
    ! Initialize RDycore
    !
    ! !USES:
    !
    implicit none
    !
    character(len=1024), intent(in) :: config_file
    !
    !
    ! !LOCAL VARIABLES:
    character(len=1024) :: log_file = 'rof_modelio.log'
    PetscViewer         :: viewer
    PetscInt            :: num_cells_global
    PetscErrorCode      :: ierr

    ! initialize subsystems
    PetscCallA(RDyInit(ierr))

    ! create rdycore and set it up with the given file
    PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
    PetscCallA(RDySetLogFile(rdy_, log_file, ierr))
    PetscCallA(RDySetup(rdy_, ierr))

    ! allocate memory for grid-level rain data
    PetscCallA(RDyGetNumLocalCells(rdy_, num_cells_owned, ierr))
    allocate(rain_data(num_cells_owned))

    PetscCallA(RDyGetNumLocalCells(rdy_, num_cells_global, ierr))

  end subroutine rdycore_init

  !-----------------------------------------------------------------------
  subroutine rdycore_run()
    !
    ! !DESCRIPTION:
    ! Initialize RDycore
    !
    implicit none
    !
    ! !LOCAL VARIABLES:
    PetscScalar, pointer :: rain_p(:)
    PetscInt             :: t
    integer(RDyTimeUnit) :: time_unit
    PetscReal            :: time_dn, time_up, cur_time, cur_rain
    PetscErrorCode       :: ierr

    cur_time = (nstep-1)*dtime

    ! Set spatially homogeneous rainfall for all grid cells
    rain_data(:) = 0.d0
    PetscCallA(RDySetWaterSourceForLocalCells(rdy_, num_cells_owned, rain_data, ierr))

    ! Set the coupling time step
    PetscCallA(RDyGetTimeUnit(rdy_, time_unit, ierr))
    PetscCallA(RDySetCouplingInterval(rdy_, time_unit, dtime, ierr))

    ! Run the simulation to completion.
    PetscCallA(RDyAdvance(rdy_, ierr))

  end subroutine rdycore_run

  !-----------------------------------------------------------------------
  subroutine rdycore_final()
    !
    ! !DESCRIPTION:
    ! Destroy RDy object
    !
    ! !USES:
    !
    implicit none
    !
    ! !LOCAL VARIABLES:
    PetscErrorCode :: ierr

    ! deallocate memory for rain data
    deallocate(rain_data)

    ! destroy RDy object
    PetscCallA(RDyDestroy(rdy_, ierr));

    ! finalize
    PetscCallA(RDyFinalize(ierr));

  end subroutine rdycore_final

end module rdycoreMod

program test_coupling

  use rdycore
  use rdycoreMod
  use petsc

  implicit none

  character(len=1024) :: config_file

  if (command_argument_count() < 1) then
    call rdycore_usage()
  else
    ! fetch config file name
    call get_command_argument(1, config_file)
  end if

  nstep = 0
  dtime = 1 ! hours for ex2b.yaml

  call rdycore_init(config_file)
  call rdycore_run()
  call rdycore_final()

end program
