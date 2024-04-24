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
  PetscMPIInt          :: myrank
  integer(RDyTimeUnit) :: time_unit
  PetscReal            :: cur_time
  PetscInt, parameter  :: ndof = 3
  PetscReal, target    :: L1_norms(3), L2_norms(3), Linf_norms(3), global_area
  PetscInt             :: num_global_cells
  PetscErrorCode       :: ierr

  if (command_argument_count() < 1) then
    call usage()
  else
    call get_command_argument(1, config_file)

    ! initialize subsystems
    PetscCallA(RDyInit(ierr))

    PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, myrank, ierr))

    if (trim(config_file) /= trim('-help')) then
      ! create rdycore and set it up with the given file
      PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
      PetscCallA(RDyMMSSetup(rdy_, ierr))

      ! run the problem to completion
      do while (.not. RDyFinished(rdy_)) ! returns true based on stopping criteria
        PetscCallA(RDyAdvance(rdy_, ierr))
      enddo

      ! compute error norms
      PetscCallA(RDyGetTimeUnit(rdy_, time_unit, ierr))
      PetscCallA(RDyGetTime(rdy_, time_unit, cur_time, ierr))
      PetscCallA(RDyMMSComputeErrorNorms(rdy_, cur_time, l1_norms, l2_norms, linf_norms, num_global_cells, global_area, ierr))

      if (myrank == 0) then
        write(*,*)'Avg-cell-area    :', global_area/num_global_cells
        write(*,*)'Avg-length-scale :', (global_area/num_global_cells) ** 0.5d0
        write(*,*)'Error-Norm-1     :', l1_norms(:)
        write(*,*)'Error-Norm-2     :', l2_norms(:)
        write(*,*)'Error-Norm-Max   :', linf_norms(:)
      endif

      ! shut off
      PetscCallA(RDyFinalize(ierr))
    endif
  endif

end program
