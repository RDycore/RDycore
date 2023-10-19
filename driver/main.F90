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
#include <finclude/rdycore.h>

  use rdycore
  use driver
  use petsc

  implicit none

  character(len=1024) :: config_file
  type(RDy)           :: rdy_
  PetscErrorCode      :: ierr
  PetscInt            :: n, step
  PetscInt            :: nbconds, ibcond, num_edges, bcond_type
  PetscReal, pointer  :: h(:), vx(:), vy(:), rain(:)
  PetscReal           :: time, time_step, prev_time, coupling_interval

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
      PetscCallA(RDyGetNumLocalCells(rdy_, n, ierr))
      allocate(h(n), vx(n), vy(n), rain(n))

      ! get information about boundary conditions
      PetscCallA(RDyGetNumBoundaryConditions(rdy_, nbconds, ierr))
      do ibcond = 1, nbconds
        PetscCallA(RDyGetNumBoundaryConditionEdges(rdy_, ibcond, num_edges, ierr))
        PetscCallA(RDyGetBoundaryConditionFlowType(rdy_, ibcond, bcond_type, ierr))
      enddo

      ! run the simulation to completion using the time parameters in the
      ! config file
      PetscCallA(RDyGetTime(rdy_, prev_time, ierr))
      PetscCallA(RDyGetCouplingInterval(rdy_, coupling_interval, ierr))
      PetscCallA(RDySetCouplingInterval(rdy_, coupling_interval, ierr))

      do while (.not. RDyFinished(rdy_)) ! returns true based on stopping criteria

        ! apply a 1 mm/hr rain over the entire domain 
        rain(:) = 1.d0/3600.d0/1000.d0
        PetscCallA(RDySetWaterSource(rdy_, rain, ierr))

        ! advance the solution by the coupling interval specified in the config file
        PetscCallA(RDyAdvance(rdy_, ierr))

        ! the following just check that RDycore is doing the right thing
        PetscCallA(RDyGetTime(rdy_, time, ierr))
        if (time <= prev_time) then
          SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Non-increasing time!")
        end if
        PetscCallA(RDyGetTimeStep(rdy_, time_step, ierr))
        if (time_step <= 0.0) then
          SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Non-positive time step!")
        end if

        if (abs(time - prev_time - coupling_interval) >= 1e-12) then
          SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "RDyAdvance advanced time improperly!")
        end if
        prev_time = prev_time + coupling_interval

        PetscCallA(RDyGetStep(rdy_, step, ierr));
        if (step <= 0) then
          SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Non-positive step index!")
        end if

        PetscCallA(RDyGetHeight(rdy_, h, ierr))
        PetscCallA(RDyGetXVelocity(rdy_, vx, ierr))
        PetscCallA(RDyGetYVelocity(rdy_, vy, ierr))
      end do

      deallocate(h, vx, vy, rain)
      PetscCallA(RDyDestroy(rdy_, ierr))
    end if

    PetscCallA(RDyFinalize(ierr));
  end if

end program
