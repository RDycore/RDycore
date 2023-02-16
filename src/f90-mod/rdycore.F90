! This file defines the Fortran interface for RDycore.
#include <petsc/finclude/petscsys.h>

module rdycore

  use iso_c_binding

  implicit none

  public :: RDy, RDyInit, RDyFinalize, RDyInitialized, &
            RDyCreate, RDySetup, RDyRun, RDyDestroy

  type :: RDy
    ! C pointer to RDy type
    type(c_ptr)                  :: c_rdy
    ! string containing config filename
    character(len=1024), pointer :: config_file
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

    integer(c_int) function rdysetup_(rdy) bind(c, name="RDySetup")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
    end function

    integer(c_int) function rdyrun_(rdy) bind(c, name="RDyRun")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), value, intent(in) :: rdy
    end function

    integer(c_int) function rdydestroy_(rdy) bind(c, name="RDyDestroy")
      use iso_c_binding, only: c_int, c_ptr
      type(c_ptr), intent(inout) :: rdy
    end function

  end interface

contains

  subroutine RDyInit(ierr)
    implicit none
    integer, intent(out) :: ierr
    if (.not. RDyInitialized()) ierr = rdyinitfortran_()
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

    integer :: n

    n = len_trim(filename)
    allocate(rdy_%config_file)
    rdy_%config_file(1:n) = filename(1:n)
    rdy_%config_file(n+1:n+1) = c_null_char
    ierr = rdycreate_(comm, c_loc(rdy_%config_file), rdy_%c_rdy)
  end subroutine

  subroutine RDySetup(rdy_, ierr)
    type(RDy), intent(inout) :: rdy_
    integer,   intent(out)   :: ierr
    ierr = rdysetup_(rdy_%c_rdy)
  end subroutine

  subroutine RDyRun(rdy_, ierr)
    type(RDy), intent(inout) :: rdy_
    integer,   intent(out)   :: ierr
    ierr = rdyrun_(rdy_%c_rdy)
  end subroutine

  subroutine RDyDestroy(rdy_, ierr)
    type(RDy), intent(inout) :: rdy_
    integer,   intent(out)   :: ierr
    ierr = rdydestroy_(rdy_%c_rdy)
    if (associated(rdy_%config_file)) deallocate(rdy_%config_file)
  end subroutine

end module rdycore
