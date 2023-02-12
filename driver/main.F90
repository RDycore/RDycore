module driver
  implicit none
contains
  subroutine usage()
    print *, "rdycore_f90: usage:"
    print *, "rdycore_f90 <input.yaml>"
    print *, ""
  end subroutine
end module

program rdycore_f90
#include <petsc/finclude/petsc.h>
  use rdycore
  use driver
  use petsc

  implicit none

  character(len=1024) :: config_file
  type(RDy)           :: rdy_
  PetscErrorCode      :: ierr

  if (command_argument_count() < 1) then
    call usage()
  else
    ! fetch config file name
    call get_command_argument(1, config_file)

    ! initialize subsystems
    PetscCallA(RDyInit(ierr))

    ! create rdycore and set it up with the given file
    PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
    PetscCallA(RDySetup(rdy_, ierr))

    ! Run the simulation to completion.
    PetscCallA(RDyRun(rdy_, ierr))

    ! clean up
    PetscCallA(RDyDestroy(rdy_, ierr));
    PetscCallA(RDyFinalize(ierr));
  end if

end program
