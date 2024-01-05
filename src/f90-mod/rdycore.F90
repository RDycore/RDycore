! This file defines the Fortran interface for RDycore.
#include <petsc/finclude/petscsys.h>



module rdycore

  use iso_c_binding, only: c_ptr, c_int, c_int64_t, c_loc

  implicit none

  public :: RDyDouble, RDy, RDyInit, RDyFinalize, RDyInitialized, &
            RDyCreate, RDySetup, RDyAdvance, RDyDestroy, &
            RDyGetNumLocalCells, RDyGetNumBoundaryConditions, &
            RDyGetNumBoundaryEdges, RDyGetBoundaryConditionFlowType, &
            RDySetDirichletBoundaryValues, &
            RDyGetTime, RDyGetTimeStep, RDyGetStep, RDyGetCouplingInterval, &
            RDySetCouplingInterval, &
            RDyGetHeightOfLocalCell, RDyGetXMomentumOfLocalCell, RDyGetYMomentumOfLocalCell, &
            RDySetWaterSourceForLocalCell, RDySetXMomentumSourceForLocalCell, RDySetYMomentumSourceForLocalCell, &
            RDySetManningsNForLocalCell

  ! RDycore uses double-precision floating point numbers
  integer, parameter :: RDyDouble = selected_real_kind(12)

  type :: RDy
    ! C pointer to RDy type
    type(c_ptr)                  :: c_rdy
  end type RDy

  interface
    integer(c_int) function rdyinitfortran_() bind(c, name="RDyInitFortran")
      use iso_c_binding, only: c_int
    end function

    logical(c_bool) function rdyinitialized_() bind(c, name="RDyInitialized")
      use iso_c_binding, only: c_bool
    end function

    integer(c_int) function rdyfinalize_() bind(c, name="RDyFinalize")
      use iso_c_binding, only: c_int
    end function

    integer(c_int) function rdycreate_(comm, filename, rdy) bind(c, name="RDyCreateF90")
      use iso_c_binding, only: c_int, c_ptr
      integer,            intent(in)  :: comm
      type(c_ptr), value, intent(in)  :: filename
      type(c_ptr),        intent(out) :: rdy
    end function

    integer(c_int) function rdysetlogfile_(rdy, filename) bind(c, name="RDySetLogFile")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: filename
    end function

    integer(c_int) function rdysetup_(rdy) bind(c, name="RDySetup")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
    end function

    integer(c_int) function rdygetnumlocalcells_(rdy, num_cells) bind(c, name="RDyGetNumLocalCells")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in)  :: rdy
      PetscInt,           intent(out) :: num_cells
    end function

    integer(c_int) function rdygetnumboundaryconditions_(rdy, num_bnd_conds) bind(c, name="RDyGetNumBoundaryConditions")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in)  :: rdy
      PetscInt,           intent(out) :: num_bnd_conds
    end function

    integer(c_int) function rdygetnumboundaryedges_(rdy, boundary_id, num_edges) bind(c, name="RDyGetNumBoundaryEdges")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt,    value, intent(in) :: boundary_id
      PetscInt,    intent(out)       :: num_edges
    end function

    integer(c_int) function rdysetdirichletboundaryvalues_(rdy, boundary_id, num_edges, ndof, bc_values) bind(c, name="RDySetDirichletBoundaryValues")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in)  :: rdy
      PetscInt,    value, intent(in)  :: boundary_id
      PetscInt,    value, intent(in)  :: num_edges
      PetscInt,    value, intent(in)  :: ndof
      type(c_ptr), value, intent(in)  :: bc_values
    end function

    integer(c_int) function rdygetboundaryconditionflowtype_(rdy, boundary_id, bnd_cond_type) bind(c, name="RDyGetBoundaryConditionFlowType")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr),    value, intent(in)  :: rdy
      PetscInt, value, intent(in)  :: boundary_id
      PetscInt,        intent(out) :: bnd_cond_type
    end function

    integer(c_int) function rdygettime_(rdy, time) bind(c, name="RDyGetTime")
      use iso_c_binding, only: c_int, c_ptr, c_double
      type(c_ptr),    value, intent(in)  :: rdy
      real(c_double),        intent(out) :: time
    end function

    integer(c_int) function rdygettimestep_(rdy, dt) bind(c, name="RDyGetTimeStep")
      use iso_c_binding, only: c_int, c_ptr, c_double
      type(c_ptr),    value, intent(in)  :: rdy
      real(c_double),        intent(out) :: dt
    end function

    integer(c_int) function rdygetstep_(rdy, step) bind(c, name="RDyGetStep")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in)  :: rdy
      PetscInt,           intent(out) :: step
    end function

    integer(c_int) function rdygetcouplinginterval_(rdy, interval) bind(c, name="RDyGetCouplingInterval")
      use iso_c_binding, only: c_int, c_ptr, c_double
      type(c_ptr),    value, intent(in)  :: rdy
      real(c_double),        intent(out) :: interval
    end function

    integer(c_int) function rdysetcouplinginterval_(rdy, interval) bind(c, name="RDySetCouplingInterval")
      use iso_c_binding, only: c_int, c_ptr, c_double
      type(c_ptr),    value, intent(in) :: rdy
      real(c_double), value, intent(in) :: interval
    end function

    integer(c_int) function rdygetheightoflocalcell_(rdy, size, h) bind(c, name="RDyGetHeightOfLocalCell")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt,           intent(in) :: size
      type(c_ptr), value, intent(in) :: h
    end function

    integer(c_int) function rdygetxmomentumoflocalcell_(rdy, size, hu) bind(c, name="RDyGetXMomentumOfLocalCell")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt,           intent(in) :: size
      type(c_ptr), value, intent(in) :: hu
    end function

    integer(c_int) function rdygetymomentumoflocalcell_(rdy, size, hv) bind(c, name="RDyGetYMomentumOfLocalCell")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt,           intent(in) :: size
      type(c_ptr), value, intent(in) :: hv
    end function

    integer(c_int) function rdysetwatersourceforlocalcell_(rdy, size, watsrc) bind(c, name="RDySetWaterSourceForLocalCell")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt, value, intent(in)    :: size
      type(c_ptr), value, intent(in) :: watsrc
    end function

    integer(c_int) function rdysetxmomentumsourceforlocalcell_(rdy, size, xmomsrc) bind(c, name="RDySetXMomentumSourceForLocalCell")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt, value, intent(in)    :: size
      type(c_ptr), value, intent(in) :: xmomsrc
    end function

    integer(c_int) function rdysetymomentumsourceforlocalcell_(rdy, size, ymomsrc) bind(c, name="RDySetYMomentumSourceForLocalCell")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt, value, intent(in)    :: size
      type(c_ptr), value, intent(in) :: ymomsrc
    end function

    integer(c_int) function rdysetmanningsnforlocalcell_(rdy, size, n) bind(c, name="RDySetManningsNForLocalCell")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      PetscInt, value, intent(in)    :: size
      type(c_ptr), value, intent(in) :: n
    end function

    integer(c_int) function rdyadvance_(rdy) bind(c, name="RDyAdvance")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
    end function

    logical(c_bool) function rdyfinished_(rdy) bind(c, name="RDyFinished")
      use iso_c_binding, only: c_bool, c_ptr
      type(c_ptr), value, intent(in) :: rdy
    end function

    integer(c_int) function rdydestroy_(rdy) bind(c, name="RDyDestroy")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), intent(inout) :: rdy
    end function

  end interface

contains

  subroutine RDyInit(ierr)
    use petscsys
    implicit none
    integer, intent(out) :: ierr
    call PetscInitialize(ierr)
    if ((ierr == 0) .and. .not. RDyInitialized()) ierr = rdyinitfortran_()
  end subroutine

  subroutine RDyFinalize(ierr)
    integer, intent(out) :: ierr
    ierr = rdyfinalize_()
  end subroutine

  logical function RDyInitialized()
    RDyInitialized = rdyinitialized_()
  end function

  subroutine RDyCreate(comm, filename, rdy_, ierr)
    use iso_c_binding, only: c_null_char
    character(len=1024), intent(in) :: filename
    integer,   intent(in)  :: comm
    type(RDy), intent(out) :: rdy_
    integer,   intent(out) :: ierr

    integer                      :: n
    character(len=1024), pointer :: config_file

    n = len_trim(filename)
    allocate(config_file)
    config_file(1:n) = filename(1:n)
    config_file(n+1:n+1) = c_null_char
    ierr = rdycreate_(comm, c_loc(config_file), rdy_%c_rdy)
    deallocate(config_file)
  end subroutine

  subroutine RDySetLogFile(rdy_, filename, ierr)
    use iso_c_binding, only: c_null_char
    character(len=1024), intent(in)  :: filename
    type(RDy),           intent(out) :: rdy_
    integer,             intent(out) :: ierr

    integer                      :: n
    character(len=1024), pointer :: log_file

    n = min(len_trim(filename), 1024)
    allocate(log_file)
    log_file(1:n) = filename(1:n)
    log_file(n+1:n+1) = c_null_char
    ierr = rdysetlogfile_(rdy_%c_rdy, c_loc(log_file))
    deallocate(log_file)
  end subroutine

  subroutine RDySetup(rdy_, ierr)
    type(RDy), intent(inout) :: rdy_
    integer,   intent(out)   :: ierr
    ierr = rdysetup_(rdy_%c_rdy)
  end subroutine

  subroutine RDyGetNumLocalCells(rdy_, num_cells, ierr)
    type(RDy), intent(inout) :: rdy_
    PetscInt,  intent(out)   :: num_cells
    integer,   intent(out)   :: ierr
    ierr = rdygetnumlocalcells_(rdy_%c_rdy, num_cells)
  end subroutine

  subroutine RDyGetNumBoundaryConditions(rdy_, num_bnd_conds, ierr)
    type(RDy), intent(inout) :: rdy_
    PetscInt,  intent(out)   :: num_bnd_conds
    integer,   intent(out)   :: ierr
    ierr = rdygetnumboundaryconditions_(rdy_%c_rdy, num_bnd_conds)
  end subroutine

  subroutine RDyGetNumBoundaryEdges(rdy_, boundary_id, num_edges, ierr)
    type(RDy), intent(inout) :: rdy_
    PetscInt,  intent(in)    :: boundary_id
    PetscInt,  intent(out)   :: num_edges
    integer,   intent(out)   :: ierr
    ierr = rdygetnumboundaryedges_(rdy_%c_rdy, boundary_id-1, num_edges)
  end subroutine

  subroutine RDySetDirichletBoundaryValues(rdy_, boundary_id, num_edges, ndof, bc_values, ierr)
    type(RDy),       intent(inout)       :: rdy_
    PetscInt,        intent(in)          :: boundary_id
    PetscInt,        intent(in)          :: num_edges
    PetscInt,        intent(in)          :: ndof
    real(RDyDouble), pointer, intent(in) :: bc_values(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetdirichletboundaryvalues_(rdy_%c_rdy, boundary_id-1, num_edges, ndof, c_loc(bc_values))
  end subroutine

  subroutine RDyGetBoundaryConditionFlowType(rdy_, boundary_id, bnd_cond_type, ierr)
    type(RDy), intent(inout) :: rdy_
    PetscInt,  intent(in)    :: boundary_id
    PetscInt,  intent(out)   :: bnd_cond_type
    integer,   intent(out)   :: ierr
    ierr = rdygetboundaryconditionflowtype_(rdy_%c_rdy, boundary_id-1, bnd_cond_type)
  end subroutine

  subroutine RDyGetTime(rdy_, time, ierr)
    type(RDy),       intent(inout) :: rdy_
    real(RDyDouble), intent(out)   :: time
    integer,         intent(out)   :: ierr
    ierr = rdygettime_(rdy_%c_rdy, time)
  end subroutine

  subroutine RDyGetTimeStep(rdy_, timestep, ierr)
    type(RDy),       intent(inout) :: rdy_
    real(RDyDouble), intent(out)   :: timestep
    integer,         intent(out)   :: ierr
    ierr = rdygettimestep_(rdy_%c_rdy, timestep)
  end subroutine

  subroutine RDyGetCouplingInterval(rdy_, interval, ierr)
    type(RDy),       intent(inout) :: rdy_
    real(RDyDouble), intent(out)   :: interval
    integer,         intent(out)   :: ierr
    ierr = rdygetcouplinginterval_(rdy_%c_rdy, interval)
  end subroutine

  subroutine RDySetCouplingInterval(rdy_, interval, ierr)
    type(RDy),       intent(inout) :: rdy_
    real(RDyDouble), intent(in)    :: interval
    integer,         intent(out)   :: ierr
    ierr = rdysetcouplinginterval_(rdy_%c_rdy, interval)
  end subroutine

  subroutine RDyGetStep(rdy_, step, ierr)
    type(RDy), intent(inout) :: rdy_
    PetscInt,  intent(out)   :: step
    integer,   intent(out)   :: ierr
    ierr = rdygetstep_(rdy_%c_rdy, step)
  end subroutine

  subroutine RDyGetHeightOfLocalCell(rdy_, size, h, ierr)
    type(RDy),       intent(inout)          :: rdy_
    PetscInt,        intent(in)             :: size
    real(RDyDouble), pointer, intent(inout) :: h(:)
    integer,         intent(out)            :: ierr
    ierr = rdygetheightoflocalcell_(rdy_%c_rdy, size, c_loc(h))
  end subroutine

  subroutine RDyGetXMomentumOfLocalCell(rdy_, size, hu, ierr)
    type(RDy),       intent(inout)          :: rdy_
    PetscInt,        intent(in)             :: size
    real(RDyDouble), pointer, intent(inout) :: hu(:)
    integer,         intent(out)            :: ierr
    ierr = rdygetxmomentumoflocalcell_(rdy_%c_rdy, size, c_loc(hu))
  end subroutine

  subroutine RDyGetYMomentumOfLocalCell(rdy_, size, hv, ierr)
    type(RDy),       intent(inout)          :: rdy_
    PetscInt,        intent(in)             :: size
    real(RDyDouble), pointer, intent(inout) :: hv(:)
    integer,         intent(out)            :: ierr
    ierr = rdygetymomentumoflocalcell_(rdy_%c_rdy, size, c_loc(hv))
  end subroutine

  subroutine RDySetWaterSourceForLocalCell(rdy_, size, watsrc, ierr)
    type(RDy),       intent(inout)       :: rdy_
    PetscInt,        intent(in)          :: size
    real(RDyDouble), pointer, intent(in) :: watsrc(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetwatersourceforlocalcell_(rdy_%c_rdy, size, c_loc(watsrc))
  end subroutine

  subroutine RDySetXMomentumSourceForLocalCell(rdy_, size, xmomsrc, ierr)
    type(RDy),       intent(inout)       :: rdy_
    PetscInt,        intent(in)          :: size
    real(RDyDouble), pointer, intent(in) :: xmomsrc(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetxmomentumsourceforlocalcell_(rdy_%c_rdy, size, c_loc(xmomsrc))
  end subroutine

  subroutine RDySetYMomentumSourceForLocalCell(rdy_, size, ymomsrc, ierr)
    type(RDy),       intent(inout)       :: rdy_
    PetscInt,        intent(in)          :: size
    real(RDyDouble), pointer, intent(in) :: ymomsrc(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetymomentumsourceforlocalcell_(rdy_%c_rdy, size, c_loc(ymomsrc))
  end subroutine

  subroutine RDySetManningsNForLocalCell(rdy_, size, n, ierr)
    type(RDy),       intent(inout)       :: rdy_
    PetscInt,        intent(in)          :: size
    real(RDyDouble), pointer, intent(in) :: n(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetmanningsnforlocalcell_(rdy_%c_rdy, size, c_loc(n))
  end subroutine

  subroutine RDyAdvance(rdy_, ierr)
    type(RDy), intent(inout) :: rdy_
    integer,   intent(out)   :: ierr
    ierr = rdyadvance_(rdy_%c_rdy)
  end subroutine

  function RDyFinished(rdy_)
    type(RDy), intent(inout) :: rdy_
    logical :: RDyFinished
    RDyFinished = rdyfinished_(rdy_%c_rdy)
  end function

  subroutine RDyDestroy(rdy_, ierr)
    type(RDy), intent(inout) :: rdy_
    integer,   intent(out)   :: ierr
    ierr = rdydestroy_(rdy_%c_rdy)
  end subroutine

end module rdycore
