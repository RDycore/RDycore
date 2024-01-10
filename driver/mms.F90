module mms_driver
#include <petsc/finclude/petsc.h>
  use petsc
  implicit none

  PetscInt, parameter :: H     = 1
  PetscInt, parameter :: DH_DX = 2
  PetscInt, parameter :: DH_DY = 3
  PetscInt, parameter :: DH_DT = 4
  PetscInt, parameter :: U     = 5
  PetscInt, parameter :: DU_DX = 6
  PetscInt, parameter :: DU_DY = 7
  PetscInt, parameter :: DU_DT = 8
  PetscInt, parameter :: V     = 9
  PetscInt, parameter :: DV_DX = 10
  PetscInt, parameter :: DV_DY = 11
  PetscInt, parameter :: DV_DT = 12
  PetscInt, parameter :: HU    = 13
  PetscInt, parameter :: HV    = 14
  PetscInt, parameter :: Z     = 15
  PetscInt, parameter :: DZ_DX = 16
  PetscInt, parameter :: DZ_DY = 17
  PetscInt, parameter :: N     = 18

contains
  subroutine usage()
    print *, "rdycore_mms_f90: usage:"
    print *, "rdycore_mms_f90 <input.yaml>"
    print *, ""
  end subroutine

end module mms_driver

module problem1

#include <petsc/finclude/petsc.h>
  use petsc
  use mms_driver
  implicit none

  PetscReal, parameter :: GRAVITY = 9.806d0
  PetscReal, parameter :: Lx = 5.d0
  PetscReal, parameter :: Ly = 5.d0
  PetscReal, parameter :: h0 = 0.005d0
  PetscReal, parameter :: u0 = 0.025d0
  PetscReal, parameter :: v0 = 0.025d0
  PetscReal, parameter :: t0 = 20.d0
  PetscReal, parameter :: n0 = 0.01d0
  PetscReal, parameter :: z0 = 0.0025d0

  contains
  subroutine problem1_getdata(t, xc, yc, datatype, data)
    use petsc
    implicit none
    PetscReal , intent(in)  :: t, xc, yc
    PetscInt  , intent(in)  :: datatype
    PetscReal , intent(out) :: data

    PetscReal               :: sin_xc , cos_xc, sin_yc, cos_yc, exp_t
    PetscErrorCode          :: ierr

    sin_xc = sin(PETSC_PI * xc / Lx)
    cos_xc = cos(PETSC_PI * xc / Ly)
    sin_yc = sin(PETSC_PI * yc / Lx)
    cos_yc = cos(PETSC_PI * yc / Ly)
    exp_t = exp(t/t0)

    select case (datatype)
    case (H)
      data = h0 * (1.d0 + sin_xc * sin_yc) * exp_t
    case (DH_DX)
      data = PETSC_PI * h0 / Lx * exp_t * sin_yc * cos_xc
    case (DH_DY)
      data = PETSC_PI * h0 / Ly * exp_t * sin_xc * cos_yc
    case (DH_DT)
      data = h0 / t0 * (1.d0 + sin_xc * sin_yc) * exp_t
    case (U)
      data = u0 * cos_xc * sin_yc * exp_t
    case (DU_DX)
      data = (-1) * PETSC_PI * u0 / Lx * sin_xc * sin_yc * exp_t
    case (DU_DY)
      data = PETSC_PI * u0 / Ly * cos_xc * cos_yc * exp_t
    case (DU_DT)
      data = u0 / t0 * cos_xc * sin_yc * exp_t
    case (V)
      data = v0 * sin_xc * cos_yc * exp_t
    case (DV_DX)
      data = PETSC_PI * v0 / Lx * cos_xc * cos_yc * exp_t
    case (DV_DY)
      data = (-1) * PETSC_PI * v0 / Ly * sin_xc * sin_yc * exp_t
    case (DV_DT)
      data = v0 / t0 * sin_xc * cos_yc * exp_t
    case (HU)
      data = (h0 * (1.d0 + sin_xc * sin_yc) * exp_t) * (u0 * cos_xc * sin_yc * exp_t)
    case (HV)
      data = (h0 * (1.d0 + sin_xc * sin_yc) * exp_t) * (v0 * sin_xc * cos_yc * exp_t)
    case (Z)
      data = z0 * sin_xc * sin_yc
    case (DZ_DX)
      data = z0 * PETSC_PI / Lx * cos_xc * sin_yc
    case (DZ_DY)
      data = z0 * PETSC_PI / Ly * sin_xc * cos_yc
    case (N)
      data = n0 * (1.d0 + sin_xc * sin_yc)
    case default
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unknown data type!")
    end select

    end subroutine problem1_getdata

    subroutine problem1_sourceterm(t, ncells, xc, yc, h_source, hu_source, hv_source)
      use petsc
      implicit none
      PetscReal , intent(in)           :: t
      PetscInt  , intent(in)           :: ncells
      PetscReal , pointer, intent(in)  :: xc(:), yc(:)
      PetscReal , pointer, intent(out) :: h_source(:), hu_source(:), hv_source(:)

      PetscInt :: icell
      PetscReal :: Cd

      ! data values
      PetscReal :: h_d, dhdx_d, dhdy_d, dhdt_d
      PetscReal :: u_d, dudx_d, dudy_d, dudt_d
      PetscReal :: v_d, dvdx_d, dvdy_d, dvdt_d
      PetscReal :: dzdx_d, dzdy_d
      PetscReal :: n_d

      do icell = 1, ncells

        call problem1_getdata(t, xc(icell), yc(icell), H, h_d)
        call problem1_getdata(t, xc(icell), yc(icell), DH_DX, dhdx_d)
        call problem1_getdata(t, xc(icell), yc(icell), DH_DY, dhdy_d)
        call problem1_getdata(t, xc(icell), yc(icell), DH_DT, dhdt_d)

        call problem1_getdata(t, xc(icell), yc(icell), U, u_d)
        call problem1_getdata(t, xc(icell), yc(icell), DU_DX, dudx_d)
        call problem1_getdata(t, xc(icell), yc(icell), DU_DY, dudy_d)
        call problem1_getdata(t, xc(icell), yc(icell), DU_DT, dudt_d)

        call problem1_getdata(t, xc(icell), yc(icell), V, v_d)
        call problem1_getdata(t, xc(icell), yc(icell), DV_DX, dvdx_d)
        call problem1_getdata(t, xc(icell), yc(icell), DV_DY, dvdy_d)
        call problem1_getdata(t, xc(icell), yc(icell), DV_DT, dvdt_d)

        call problem1_getdata(t, xc(icell), yc(icell), N, n_d)

        call problem1_getdata(t, xc(icell), yc(icell), DZ_DX, dzdx_d)
        call problem1_getdata(t, xc(icell), yc(icell), DZ_DY, dzdy_d)

        Cd = GRAVITY * n_d * n_d * ((h_d)**(-1.d0/3.d0))

        h_source(icell) = dhdt_d + u_d * dhdx_d + h_d * dudx_d + v_d * dhdy_d + h_d * dvdy_d

        hu_source(icell) = u_d * dhdt_d + h_d * dudt_d
        hu_source(icell) = hu_source(icell) + 2.d0 * u_d * h_d * dudx_d + u_d * u_d * dhdx_d + GRAVITY * h_d * dhdx_d
        hu_source(icell) = hu_source(icell) + u_d * h_d * dvdy_d + v_d * h_d * dudy_d + u_d * v_d * dhdy_d
        hu_source(icell) = hu_source(icell) + dzdx_d * GRAVITY * h_d
        hu_source(icell) = hu_source(icell) + Cd * u_d * (u_d * u_d + v_d * v_d)**0.5d0

        hv_source(icell) = v_d * dhdt_d + h_d * dvdt_d
        hv_source(icell) = hv_source(icell) + u_d * h_d * dvdx_d + v_d * h_d * dudx_d + u_d * v_d * dhdx_d
        hv_source(icell) = hv_source(icell) + v_d * v_d * dhdy_d + 2.d0 * v_d * h_d * dvdy_d + GRAVITY * h_d * dhdy_d
        hv_source(icell) = hv_source(icell) + dzdy_d * GRAVITY * h_d
        hv_source(icell) = hv_source(icell) + Cd * v_d * (u_d * u_d + v_d * v_d)**0.5d0
      enddo

    end subroutine problem1_sourceterm

    subroutine problem1_dirichletvalue(t, nedges, xc_bnd_cell, yc_bnd_cell, xc_edge, yc_edge, h_bnd, hu_bnd, hv_bnd, bc_values)
      use petsc
      implicit none
      PetscReal , intent(in)           :: t
      PetscInt  , intent(in)           :: nedges
      PetscReal , pointer, intent(in)  :: xc_bnd_cell(:), yc_bnd_cell(:), xc_edge(:), yc_edge(:)
      PetscReal , pointer, intent(out) :: h_bnd(:), hu_bnd(:), hv_bnd(:), bc_values(:)

      PetscInt                         :: iedge
      PetscReal                        :: dx, dy, xc, yc, h_bnd_d, u_bnd_d, v_bnd_d

      do iedge = 1, nedges
        dx = xc_bnd_cell(iedge) - xc_edge(iedge)
        dy = yc_bnd_cell(iedge) - yc_edge(iedge)
        xc = xc_edge(iedge) - dx
        yc = yc_edge(iedge) - dy

        call problem1_getdata(t, xc, yc, H, h_bnd_d)
        call problem1_getdata(t, xc, yc, U, u_bnd_d)
        call problem1_getdata(t, xc, yc, V, v_bnd_d)

        h_bnd(iedge) = h_bnd_d
        hu_bnd(iedge) = h_bnd_d * u_bnd_d
        hv_bnd(iedge) = h_bnd_d * v_bnd_d

        bc_values((iedge - 1)*3 + 1) = h_bnd_d
        bc_values((iedge - 1)*3 + 2) = h_bnd_d * u_bnd_d
        bc_values((iedge - 1)*3 + 3) = h_bnd_d * v_bnd_d

      enddo

    end subroutine problem1_dirichletvalue

end module problem1

program mms_f90

#include <petsc/finclude/petsc.h>
#include <finclude/rdycore.h>

  use rdycore
  use mms_driver
  use petsc
  use problem1

  implicit none

  character(len=1024) :: config_file
  type(RDy)           :: rdy_
  PetscInt            :: icell, ncells, nedges, nbcs, ndof, bc_type
  PetscInt, parameter :: bc_idx = 1
  PetscReal           :: cur_time
  PetscReal, pointer  :: xc_cell(:), yc_cell(:)
  PetscReal, pointer  :: h_source(:), hu_source(:), hv_source(:), mannings_n(:)
  PetscReal, pointer  :: xc_edge(:), yc_edge(:), xc_bnd_cell(:), yc_bnd_cell(:), h_bnd(:), hu_bnd(:), hv_bnd(:), bc_values(:)
  Vec                 :: ic_vec
  PetscScalar, pointer :: ic_ptr(:)
  PetscErrorCode      :: ierr

  if (command_argument_count() < 1) then
    call usage()
  else
    call get_command_argument(1, config_file)

    ! initialize subsystems
    PetscCallA(RDyInit(ierr))

    if (trim(config_file) /= trim('-help')) then
      ! create rdycore and set it up with the given file
      PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
      PetscCallA(RDySetup(rdy_, ierr))

      ! get information about cells
      PetscCallA(RDyGetNumLocalCells(rdy_, ncells, ierr))
      allocate(xc_cell(ncells), yc_cell(ncells))
      PetscCallA(RDyGetLocalCellXCentroids(rdy_, ncells, xc_cell, ierr))
      PetscCallA(RDyGetLocalCellYCentroids(rdy_, ncells, yc_cell, ierr))

      PetscCallA(RDyGetNumBoundaryConditions(rdy_, nbcs, ierr))
      if (nbcs /= 1) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Only expecting one boundary condition to be present in the yaml")
      endif

      PetscCallA(RDyGetBoundaryConditionFlowType(rdy_, bc_idx, bc_type, ierr))
      if (bc_type /= CONDITION_DIRICHLET) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Only expecting CONDITION_DIRICHLET to be present in the yaml")
      endif

      PetscCallA(RDyGetNumBoundaryEdges(rdy_, bc_idx, nedges, ierr))
      allocate(xc_edge(nedges), yc_edge(nedges), xc_bnd_cell(nedges), yc_bnd_cell(nedges))
      allocate(h_bnd(nedges), hu_bnd(nedges), hv_bnd(nedges), bc_values(3*nedges))

      ! get geometric attributes about edges
      PetscCallA(RDyGetBoundaryEdgeXCentroids(rdy_, bc_idx, nedges, xc_edge, ierr));
      PetscCallA(RDyGetBoundaryEdgeYCentroids(rdy_, bc_idx, nedges, yc_edge, ierr));
      PetscCallA(RDyGetBoundaryCellXCentroids(rdy_, bc_idx, nedges, xc_bnd_cell, ierr));
      PetscCallA(RDyGetBoundaryCellYCentroids(rdy_, bc_idx, nedges, yc_bnd_cell, ierr));

      ! allocate memory for manning's N and source sink terms
      allocate(h_source(ncells), hu_source(ncells), hv_source(ncells), mannings_n(ncells))
      !PetscCallA(RDyCreatePrognosticVector(rdy_, ic_vec, ierr))
      !PetscCallA(VecGetArrayF90(ic_vec, ic_ptr, ierr))

      cur_time = 0.d0
      do icell = 1, ncells
        call problem1_getdata(cur_time, xc_cell(icell), yc_cell(icell), H, h_source(icell))
        call problem1_getdata(cur_time, xc_cell(icell), yc_cell(icell), HU, hu_source(icell))
        call problem1_getdata(cur_time, xc_cell(icell), yc_cell(icell), HV, hv_source(icell))
        call problem1_getdata(cur_time, xc_cell(icell), yc_cell(icell), N, mannings_n(icell))

        !ic_ptr((icell - 1)*3 + 1) = h_source(icell)
        !ic_ptr((icell - 1)*3 + 2) = hu_source(icell)
        !ic_ptr((icell - 1)*3 + 3) = hv_source(icell)
      enddo

      !PetscCallA(VecRestoreArrayF90(ic_vec, ic_ptr, ierr))

      PetscCallA(RDySetManningsNForLocalCell(rdy_, ncells, mannings_n, ierr));
      !PetscCallA(RDySetInitialConditions(rdy_, ic_vec, ierr));

      ndof = 3
      do while (.not. RDyFinished(rdy_)) ! returns true based on stopping criteria
        PetscCallA(RDyGetTime(rdy_, cur_time, ierr))

        call problem1_sourceterm(cur_time, ncells, xc_cell, yc_cell, h_source, hu_source, hv_source)
        call problem1_dirichletvalue(cur_time, nedges, xc_bnd_cell, yc_bnd_cell, xc_edge, yc_edge, h_bnd, hu_bnd, hv_bnd, bc_values)

        ! set the MMS source terms
        PetscCallA(RDySetWaterSourceForLocalCell(rdy_, ncells, h_source, ierr))
        PetscCallA(RDySetXMomentumSourceForLocalCell(rdy_, ncells, hu_source, ierr))
        PetscCallA(RDySetYMomentumSourceForLocalCell(rdy_, ncells, hv_source, ierr))

        ! set dirchlet BC
        PetscCallA(RDySetDirichletBoundaryValues(rdy_, bc_idx, nedges, ndof, bc_values, ierr))

        ! advance the solution by the coupling interval
        PetscCallA(RDyAdvance(rdy_, ierr))

      enddo

      deallocate(xc_cell, yc_cell, h_source, hu_source, hv_source)
      deallocate(xc_edge, yc_edge, xc_bnd_cell, yc_bnd_cell, h_bnd, hu_bnd, hv_bnd, bc_values)

      ! shut off
      PetscCallA(RDyFinalize(ierr))
    endif
  endif


end program
