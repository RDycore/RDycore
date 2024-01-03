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
            RDyGetHeight, RDyGetXVelocity, RDyGetYVelocity

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

    integer(c_int) function rdygetheight_(rdy, h) bind(c, name="RDyGetHeight")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: h
    end function

    integer(c_int) function rdygetxvelocity_(rdy, vx) bind(c, name="RDyGetXVelocity")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: vx
    end function

    integer(c_int) function rdygetyvelocity_(rdy, vy) bind(c, name="RDyGetYVelocity")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: vy
    end function

    integer(c_int) function rdysetwatersource_(rdy, watsrc) bind(c, name="RDySetWaterSource")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: watsrc
    end function

    integer(c_int) function rdysetxmomentumsource_(rdy, xmomsrc) bind(c, name="RDySetXMomentumSource")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: xmomsrc
    end function

    integer(c_int) function rdysetymomentumsource_(rdy, ymomsrc) bind(c, name="RDySetYMomentumSource")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: ymomsrc
    end function

    integer(c_int) function rdysetmanningn_(rdy, watsrc) bind(c, name="RDySetManningN")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
      type(c_ptr), value, intent(in) :: watsrc
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

  subroutine RDyGetHeight(rdy_, h, ierr)
    type(RDy),       intent(inout)          :: rdy_
    real(RDyDouble), pointer, intent(inout) :: h(:)
    integer,         intent(out)            :: ierr
    ierr = rdygetheight_(rdy_%c_rdy, c_loc(h))
  end subroutine

  subroutine RDyGetXVelocity(rdy_, vx, ierr)
    type(RDy),       intent(inout)          :: rdy_
    real(RDyDouble), pointer, intent(inout) :: vx(:)
    integer,         intent(out)            :: ierr
    ierr = rdygetxvelocity_(rdy_%c_rdy, c_loc(vx))
  end subroutine

  subroutine RDyGetYVelocity(rdy_, vy, ierr)
    type(RDy),       intent(inout)          :: rdy_
    real(RDyDouble), pointer, intent(inout) :: vy(:)
    integer,         intent(out)            :: ierr
    ierr = rdygetyvelocity_(rdy_%c_rdy, c_loc(vy))
  end subroutine

  subroutine RDySetWaterSource(rdy_, watsrc, ierr)
    type(RDy),       intent(inout)       :: rdy_
    real(RDyDouble), pointer, intent(in) :: watsrc(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetwatersource_(rdy_%c_rdy, c_loc(watsrc))
  end subroutine

  subroutine RDySetXMomentumSource(rdy_, xmomsrc, ierr)
    type(RDy),       intent(inout)       :: rdy_
    real(RDyDouble), pointer, intent(in) :: xmomsrc(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetxmomentumsource_(rdy_%c_rdy, c_loc(xmomsrc))
  end subroutine

  subroutine RDySetYMomentumSource(rdy_, ymomsrc, ierr)
    type(RDy),       intent(inout)       :: rdy_
    real(RDyDouble), pointer, intent(in) :: ymomsrc(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetymomentumsource_(rdy_%c_rdy, c_loc(ymomsrc))
  end subroutine

  subroutine RDySetManningN(rdy_, watsrc, ierr)
    type(RDy),       intent(inout)       :: rdy_
    real(RDyDouble), pointer, intent(in) :: watsrc(:)
    integer,         intent(out)         :: ierr
    ierr = rdysetmanningn_(rdy_%c_rdy, c_loc(watsrc))
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
