module mms_driver
#include <petsc/finclude/petsc.h>
  use petsc
  implicit none

contains
  subroutine usage()
    print *, "rdycore_mms_f90: usage:"
    print *, "rdycore_mms_f90 <input.yaml>"
    print *, ""
  end subroutine

end module mms_driver

program mms_f90

#include <petsc/finclude/petsc.h>
#include <finclude/rdycore.h>

  use rdycore
  use mms_driver
  use petsc

  implicit none

  character(len=1024)  :: config_file
  type(RDy)            :: rdy_
  PetscErrorCode       :: ierr

  if (command_argument_count() < 1) then
    call usage()
  else
    call get_command_argument(1, config_file)

    ! initialize subsystems
    PetscCallA(RDyInit(ierr))

    if (trim(config_file) /= trim('-help')) then
      ! create rdycore and set it up with the given file
      PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
      PetscCallA(RDyMMSSetup(rdy_, ierr))

      ! run the problem according to the given configuration
      PetscCallA(RDyMMSRun(rdy_, ierr));
      PetscCallA(RDyDestroy(rdy_, ierr))
    endif
    ! shut off
    PetscCallA(RDyFinalize(ierr))
  endif

end program
