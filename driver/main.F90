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
  PetscInt            :: n
  PetscReal, pointer  :: h(:), vx(:), vy(:)

  if (command_argument_count() < 1) then
    call usage()
  else
    ! fetch config file name
    call get_command_argument(1, config_file)

    ! initialize subsystems
    PetscCallA(RDyInit(ierr))

    if (trim(config_file) /= trim('-help')) then
      ! create rdycore and set it up with the given file
      PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
      PetscCallA(RDySetup(rdy_, ierr))

      ! allocate arrays for inspecting simulation data
      PetscCallA(RDyGetNumLocalCells(rdy, n, ierr))
      allocate(h(n), vx(n), vy(n))

      ! run the simulation to completion using the time parameters in the
      ! config file
      do while (.not. RDyFinished(rdy_))
        PetscCallA(RDyAdvance(rdy_, ierr))

        PetscCallA(RDyGetHeight(rdy, h));
        PetscCallA(RDyGetXVelocity(rdy, vx));
        PetscCallA(RDyGetYVelocity(rdy, vy));
      end do

      deallocate(h, vx, vy)
      PetscCallA(RDyDestroy(rdy_, ierr));
    end if

    PetscCallA(RDyFinalize(ierr));
  end if

end program
