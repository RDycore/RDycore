module rdy_driver
#include <petsc/finclude/petsc.h>
  use petsc
  implicit none
contains
  subroutine usage()
    print *, "rdycore_f90: usage:"
    print *, "rdycore_f90 <input.yaml>"
    print *, ""
  end subroutine


  subroutine opendata(filename, data_vec, ndata)
    implicit none
    character(*)   :: filename
    Vec            :: data_vec
    PetscInt       :: ndata

    PetscInt       :: size
    PetscViewer    :: viewer
    PetscErrorCode :: ierr

    PetscCallA(VecCreate(PETSC_COMM_SELF, data_vec, ierr))
    PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF, filename, FILE_MODE_READ, viewer, ierr))
    PetscCallA(VecLoad(data_vec, viewer, ierr));
    PetscCallA(PetscViewerDestroy(viewer, ierr));

    PetscCallA(VecGetSize(data_vec, size, ierr))
    ndata = size / 2

  end subroutine

  subroutine getcurrentdata(data_ptr, ndata, cur_time, temporally_interpolate, cur_data_idx, cur_data)
    implicit none
    PetscScalar, pointer   :: data_ptr(:)
    PetscInt , intent(in)  :: ndata
    PetscReal, intent(in)  :: cur_time
    PetscBool, intent(in)  :: temporally_interpolate
    PetscInt , intent(out) :: cur_data_idx
    PetscReal, intent(out) :: cur_data

    PetscBool            :: found
    PetscInt, parameter  :: stride = 2
    PetscInt             :: itime
    PetscReal            :: time_dn, time_up
    PetscReal            :: data_dn, data_up

    found = PETSC_FALSE
    do itime = 1, ndata-1

      time_dn = data_ptr((itime-1)*stride + 1)
      data_dn = data_ptr((itime-1)*stride + 2)

      time_up = data_ptr((itime-1)*stride + 3)
      data_up = data_ptr((itime-1)*stride + 4)

      if (cur_time >= time_dn .and. cur_time < time_up) then
        found = PETSC_TRUE
        cur_data_idx = itime
        exit
      endif
    enddo

    if (.not.found) then
      cur_data_idx = ndata
      cur_data = data_ptr((cur_data_idx-1)*stride + 2)
    else
      if (temporally_interpolate) then
        cur_data = (cur_time - time_dn)/(time_up - time_dn)*(data_up - data_dn) + data_dn
      else
        cur_data = data_dn
      endif
    endif

  end subroutine

end module rdy_driver

program rdycore_f90
#include <petsc/finclude/petsc.h>
#include <finclude/rdycore.h>

  use rdycore
  use rdy_driver
  use petsc

  implicit none

  character(len=1024)  :: config_file
  type(RDy)            :: rdy_
  PetscErrorCode       :: ierr
  PetscInt             :: n, step, iedge
  PetscInt             :: nbconds, ibcond, num_edges, bcond_type
  PetscReal, pointer   :: h(:), hu(:), hv(:), rain(:), bc_values(:), values(:), values_bnd(:)
  PetscInt,  pointer   :: nat_id(:), nat_id_bnd_cell(:)
  integer(RDyTimeUnit) :: time_unit
  PetscReal            :: time, time_step, prev_time, coupling_interval, cur_time

  PetscBool            :: rain_specified, bc_specified
  character(len=1024)  :: rainfile, bcfile
  Vec                  :: rain_vec, bc_vec
  PetscScalar, pointer :: rain_ptr(:), bc_ptr(:)
  PetscInt             :: nrain, nbc, num_edges_dirc_bc
  PetscInt             :: dirc_bc_idx, global_dirc_bc_idx
  PetscInt             :: cur_rain_idx, prev_rain_idx
  PetscInt             :: cur_bc_idx, prev_bc_idx
  PetscReal            :: cur_rain, cur_bc
  PetscBool            :: interpolate_rain, interpolate_bc, flg
  PetscInt, parameter  :: ndof = 3

  if (command_argument_count() < 1) then
    call usage()
  else
    ! fetch config file name
    call get_command_argument(1, config_file)

    ! initialize subsystems
    PetscCallA(RDyInit(ierr))

    if (trim(config_file) /= trim('-help')) then

      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-rain', rainfile, rain_specified, ierr))
      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-homogeneous_bc_file', bcfile, bc_specified, ierr))

      ! Flags to temporally interpolate rain and bc forcing
      interpolate_rain = PETSC_FALSE
      interpolate_bc   = PETSC_FALSE
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-temporally_interpolate_rain', interpolate_rain, flg, ierr))
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-temporally_interpolate_bc', interpolate_bc, flg, ierr))

      if (rain_specified) then
        call opendata(rainfile, rain_vec, nrain)
        PetscCallA(VecGetArrayF90(rain_vec, rain_ptr, ierr))
      endif

      if (bc_specified) then
        call opendata(bcfile, bc_vec, nbc)
        PetscCallA(VecGetArrayF90(bc_vec, bc_ptr, ierr))
      endif

      ! create rdycore and set it up with the given file
      PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
      PetscCallA(RDySetup(rdy_, ierr))

      ! allocate arrays for inspecting simulation data
      PetscCallA(RDyGetNumLocalCells(rdy_, n, ierr))
      allocate(h(n), hu(n), hv(n), rain(n), values(n), nat_id(n))

      ! get some mesh attributes
      PetscCallA(RDyGetLocalCellXCentroids(rdy_, n, values, ierr))
      PetscCallA(RDyGetLocalCellYCentroids(rdy_, n, values, ierr))
      PetscCallA(RDyGetLocalCellZCentroids(rdy_, n, values, ierr))
      PetscCallA(RDyGetLocalCellNaturalIDs(rdy_, n, nat_id, ierr))

      values(:) = 0.d0
      PetscCallA(RDySetDomainXMomentumSource(rdy_, n, values, ierr))
      PetscCallA(RDySetDomainYMomentumSource(rdy_, n, values, ierr))

      ! get information about boundary conditions
      dirc_bc_idx = 0
      num_edges_dirc_bc = 0
      PetscCallA(RDyGetNumBoundaryConditions(rdy_, nbconds, ierr))
      do ibcond = 1, nbconds
        PetscCallA(RDyGetNumBoundaryEdges(rdy_, ibcond, num_edges, ierr))
        PetscCallA(RDyGetBoundaryConditionFlowType(rdy_, ibcond, bcond_type, ierr))

        if (bcond_type == CONDITION_DIRICHLET) then
          if (bc_specified .and. dirc_bc_idx > 0) then
            SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "When BC file specified via -homogeneous_bc_file argument, only one CONDITION_DIRICHLET can be present in the yaml")
          endif
          dirc_bc_idx = ibcond
          num_edges_dirc_bc = num_edges
        endif

        allocate(nat_id_bnd_cell(num_edges), values_bnd(num_edges))
        PetscCallA(RDyGetBoundaryCellNaturalIDs(rdy_, ibcond, num_edges, nat_id_bnd_cell, ierr))
        PetscCallA(RDyGetBoundaryEdgeXCentroids(rdy_, ibcond, num_edges, values_bnd, ierr))
        PetscCallA(RDyGetBoundaryEdgeYCentroids(rdy_, ibcond, num_edges, values_bnd, ierr))
        PetscCallA(RDyGetBoundaryEdgeZCentroids(rdy_, ibcond, num_edges, values_bnd, ierr))
        PetscCallA(RDyGetBoundaryCellXCentroids(rdy_, ibcond, num_edges, values_bnd, ierr))
        PetscCallA(RDyGetBoundaryCellYCentroids(rdy_, ibcond, num_edges, values_bnd, ierr))
        PetscCallA(RDyGetBoundaryCellZCentroids(rdy_, ibcond, num_edges, values_bnd, ierr))

        deallocate(nat_id_bnd_cell)
      enddo
      allocate(bc_values(num_edges_dirc_bc * ndof))

      ! run the simulation to completion using the time parameters in the
      ! config file
      PetscCallA(RDyGetTimeUnit(rdy_, time_unit, ierr))
      PetscCallA(RDyGetTime(rdy_, time_unit, prev_time, ierr))
      PetscCallA(RDyGetCouplingInterval(rdy_, time_unit, coupling_interval, ierr))
      PetscCallA(RDySetCouplingInterval(rdy_, time_unit, coupling_interval, ierr))

      prev_rain_idx = 0
      prev_bc_idx = 0

      do while (.not. RDyFinished(rdy_)) ! returns true based on stopping criteria

        ! apply a 1 mm/hr rain over the entire domain (region 0)
        if (.not. rain_specified) then
          rain(:) = 1.d0/3600.d0/1000.d0
          PetscCallA(RDySetDomainWaterSource(rdy_, n, rain, ierr))
        else
          PetscCallA(RDyGetTime(rdy_, time_unit, cur_time, ierr))
          call getcurrentdata(rain_ptr, nrain, cur_time, interpolate_rain, cur_rain_idx, cur_rain)
          if (interpolate_rain .or. cur_rain_idx /= prev_rain_idx) then
            prev_rain_idx = cur_rain_idx
            rain(:) = cur_rain
            PetscCallA(RDySetDomainWaterSource(rdy_, n, rain, ierr))
          endif
        endif

        if (bc_specified) then
          call getcurrentdata(bc_ptr, nbc, cur_time, interpolate_bc, cur_bc_idx, cur_bc)
          if (interpolate_bc .or. cur_bc_idx /= prev_bc_idx) then
            prev_bc_idx = cur_bc_idx
            do iedge = 1, num_edges_dirc_bc
              bc_values((iedge-1)*ndof + 1) = cur_bc
              bc_values((iedge-1)*ndof + 2) = 0.d0
              bc_values((iedge-1)*ndof + 3) = 0.d0
            enddo
          endif
          if (num_edges_dirc_bc > 0) then
            PetscCallA(RDySetDirichletBoundaryValues(rdy_, dirc_bc_idx, num_edges_dirc_bc, ndof, bc_values, ierr))
          endif
        endif

        ! advance the solution by the coupling interval specified in the config file
        PetscCallA(RDyAdvance(rdy_, ierr))

        ! the following just check that RDycore is doing the right thing
        PetscCallA(RDyGetTime(rdy_, time_unit, time, ierr))
        if (time <= prev_time) then
          SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Non-increasing time!")
        end if
        PetscCallA(RDyGetTimeStep(rdy_, time_unit, time_step, ierr))
        if (time_step <= 0.0) then
          SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Non-positive time step!")
        end if

        if (.not. RDyRestarted(rdy_)) then
          if (abs(time - prev_time - coupling_interval) >= 1e-12) then
            SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "RDyAdvance advanced time improperly!")
          end if
          prev_time = prev_time + coupling_interval
        else
          prev_time = time
        end if

        PetscCallA(RDyGetStep(rdy_, step, ierr));
        if (step <= 0) then
          SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Non-positive step index!")
        end if

        PetscCallA(RDyGetLocalCellHeights(rdy_, n, h, ierr))
        PetscCallA(RDyGetLocalCellXMomenta(rdy_, n, hu, ierr))
        PetscCallA(RDyGetLocalCellYMomenta(rdy_, n, hv, ierr))
      end do

      deallocate(h, hu, hv, rain, bc_values, values, nat_id)
      if (rain_specified) then
        PetscCallA(VecRestoreArrayF90(rain_vec, rain_ptr, ierr))
        PetscCallA(VecDestroy(rain_vec, ierr))
      endif
      if (bc_specified) then
        PetscCallA(VecRestoreArrayF90(bc_vec, bc_ptr, ierr))
        PetscCallA(VecDestroy(bc_vec, ierr))
      endif
      PetscCallA(RDyDestroy(rdy_, ierr))
    end if

    PetscCallA(RDyFinalize(ierr));
  end if

end program
