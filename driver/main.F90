module rdy_driver
#include <petsc/finclude/petsc.h>
  use petsc
  implicit none

#define DATASET_UNSET 0
#define DATASET_CONSTANT 1
#define DATASET_HOMOGENEOUS 2
#define DATASET_RASTER 3
#define DATASET_UNSTRUCTURED 4
#define DATASET_MULTI_HOMOGENEOUS 5

  type :: time_struct
    PetscInt :: year
    PetscInt :: month
    PetscInt :: day
    PetscInt :: hour
    PetscInt :: minute
  end type time_struct

  type, public :: HomogeneousDataset
    character(len=1024) :: filename
    Vec :: data_vec
    PetscInt :: ndata
    PetscScalar, pointer :: data_ptr(:)
    PetscBool    :: temporally_interpolate
    PetscInt     :: cur_idx, prev_idx
  end type HomogeneousDataset

  type, public :: UnstructuredDataset
    character(len=1024) :: dir
    character(len=1024) :: file
    character(len=1024) :: mesh_file
    character(len=1024) :: map_file

    PetscReal :: dtime_in_hour
    PetscInt  :: ndata_file

    Vec :: data_vec
    PetscScalar, pointer :: data_ptr(:)

    PetscInt :: ndata
    PetscInt :: stride

    type(time_struct)  :: start_date, current_date

    PetscInt           :: mesh_nelements          ! number of cells or boundary edges in RDycore mesh
    PetscInt, pointer  :: data2mesh_idx(:)        ! for each RDycore element (cells or boundary edges), the index of the data in the unstructured dataset
    PetscReal, pointer :: data_xc(:), data_yc(:)  ! x and y coordinates of data
    PetscReal, pointer :: mesh_xc(:), mesh_yc(:)  ! x and y coordinates of RDycore elments

    PetscBool :: write_map_for_debugging  ! if true, write the mapping between the RDycore cells and the dataset for debugging
    PetscBool :: write_map                ! if true, write the map between the RDycore cells and the dataset
    PetscBool :: read_map                 ! if true, read the map between the RDycore cells and the dataset

  end type UnstructuredDataset

  type, public :: BoundaryCondition
    PetscInt :: datatype

    type(HomogeneousDataset)  :: homogeneous
    type(UnstructuredDataset) :: unstructured

    PetscInt :: ndata
    PetscInt :: dirichlet_bc_idx
    PetscScalar, pointer :: data_for_rdycore(:)
  end type BoundaryCondition

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

  ! Parse the command line options for the boundary condition dataset
  subroutine ParseBoundaryDataOptions(bc)
    !
    implicit none
    !
    type(BoundaryCondition) :: bc
    !
    PetscInt          :: dataset_type_count
    PetscBool         :: flg
    PetscBool         :: homogenous_bc_flag
    PetscBool         :: unstructured_bc_dir_flag
    PetscBool         :: unstructured_start_date_flag
    PetscInt, pointer :: date(:)
    PetscInt          :: ndate = 5
    PetscErrorCode    :: ierr

    bc%datatype                           = DATASET_UNSET
    bc%ndata                              = 0
    bc%dirichlet_bc_idx                   = -1
    bc%homogeneous%temporally_interpolate = PETSC_FALSE

    dataset_type_count = 0

    ! parse information about spatially-homogeneous, temporally-varying boundary condition dataset
    PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-temporally_interpolate_bc', bc%homogeneous%temporally_interpolate, flg, ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-homogeneous_bc_file', bc%homogeneous%filename, homogenous_bc_flag, ierr))

    if (homogenous_bc_flag) then
      dataset_type_count = dataset_type_count + 1
      bc%datatype        = DATASET_HOMOGENEOUS
    endif

    ! parse information about unstructured boundary condition dataset
    PetscCall(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-unstructured_bc_dir", bc%unstructured%dir, unstructured_bc_dir_flag, ierr))
    PetscCall(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-unstructured_bc_write_map_for_debugging", bc%unstructured%write_map_for_debugging, flg, ierr))
    PetscCall(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-unstructured_bc_write_map_file", bc%unstructured%map_file, bc%unstructured%write_map, ierr))
    PetscCall(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-unstructured_bc_read_map_file", bc%unstructured%map_file, bc%unstructured%read_map, ierr))
    PetscCall(PetscOptionsGetIntArray(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-unstructured_bc_start_date", date, ndate, unstructured_start_date_flag, ierr))

    if (unstructured_start_date_flag) then
      dataset_type_count = dataset_type_count + 1
      if (ndate /= 5) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "-unstructured_bc_start_date should be in YY,MO,DD,HH,MM format")
      endif

      if (unstructured_bc_dir_flag .eqv. PETSC_FALSE) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Need to specify path to unstructured BC data via -unstructured_bc_dir <dir>")
      endif

      PetscCall(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-unstructured_bc_mesh_file", bc%unstructured%mesh_file, flg, ierr))
      if (flg .eqv. PETSC_FALSE) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Need to specify the mesh file -unstructured_bc_mesh_file <file>")
      endif

      bc%datatype = DATASET_UNSTRUCTURED
      bc%unstructured%start_date%year   = date(1)
      bc%unstructured%start_date%month  = date(2)
      bc%unstructured%start_date%day    = date(3)
      bc%unstructured%start_date%hour   = date(4)
      bc%unstructured%start_date%minute = date(5)

      bc%unstructured%current_date%year   = date(1)
      bc%unstructured%current_date%month  = date(2)
      bc%unstructured%current_date%day    = date(3)
      bc%unstructured%current_date%hour   = date(4)
      bc%unstructured%current_date%minute = date(5)

    endif

    if (dataset_type_count > 1) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "More than one boundary condition type cannot be specified")
    endif

  end subroutine ParseBoundaryDataOptions

  ! Loads up a spatially homogeneous, temporally varying dataset, which is
  ! a PETSc Vec in binary format.
  subroutine OpenHomogeneousDataset(data)
    !
    use rdycore
    use petsc
    !
      implicit none
    !
    type(HomogeneousDataset) :: data
    PetscErrorCode           :: ierr

    ! open the dataset file and read the data into a vector
    call opendata(data%filename, data%data_vec, data%ndata)

    ! get the data pointer to the data in the vector
    PetscCallA(VecGetArray(data%data_vec, data%data_ptr, ierr))

    ! set initial settings
    data%cur_idx = 0
    data%prev_idx = 0

  end subroutine OpenHomogeneousDataset

  ! Resotres the data pointer to the data in the vector and destroys the
  ! vector
  subroutine DestroyHomogeneousDataset(data)
    !
    use rdycore
    use petsc
    !
      implicit none
    !
    type(HomogeneousDataset) :: data
    PetscErrorCode           :: ierr

    ! get the data pointer to the data in the vector
    PetscCallA(VecRestoreArray(data%data_vec, data%data_ptr, ierr))
    PetscCallA(VecDestroy(data%data_vec, ierr))

  end subroutine DestroyHomogeneousDataset

  ! Finds information about dirichlet BC
  subroutine FindDirichletBCID(rdy_, dirc_bc_idx, num_edges_dirc_bc, global_dirc_bc_idx, multiple_dirc_bcs_present)
    !
#include <finclude/rdycore.h>
    !
    use rdycore
    use petsc
    !
      implicit none
    !
    type(RDy)               :: rdy_
    PetscInt               :: dirc_bc_idx, num_edges_dirc_bc
    PetscMPIInt            :: global_dirc_bc_idx
    PetscBool              :: multiple_dirc_bcs_present
    !
    PetscInt :: ibcond, nbconds, num_edges, bcond_type
    PetscErrorCode           :: ierr

    dirc_bc_idx       = -1
    global_dirc_bc_idx = -1
    num_edges_dirc_bc = 0
    multiple_dirc_bcs_present = PETSC_FALSE

    PetscCallA(RDyGetNumBoundaryConditions(rdy_, nbconds, ierr))
    do ibcond = 1, nbconds
      PetscCallA(RDyGetNumBoundaryEdges(rdy_, ibcond, num_edges, ierr))
      PetscCallA(RDyGetBoundaryConditionFlowType(rdy_, ibcond, bcond_type, ierr))

      if (bcond_type == CONDITION_DIRICHLET) then
        if (dirc_bc_idx > -1) then
          multiple_dirc_bcs_present = PETSC_TRUE
        endif
        dirc_bc_idx       = ibcond
        num_edges_dirc_bc = num_edges
      endif
    enddo

    call MPI_Allreduce(dirc_bc_idx, global_dirc_bc_idx, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD, ierr)

  end subroutine FindDirichletBCID

  ! Postprocess the boundary condition dataset for the homogeneous case
  subroutine DoPostprocessForBoundaryHomogeneousDataset(rdy_, bc_dataset)
    !
    use rdycore
    use petsc
    !
      implicit none
    !
    type(RDy)               :: rdy_
    type(BoundaryCondition) :: bc_dataset
    !
    PetscInt    :: dirc_bc_idx, num_edges_dirc_bc
    PetscMPIInt :: global_dirc_bc_idx
    PetscBool   :: multiple_dirc_bcs_present

    call FindDirichletBCID(rdy_, dirc_bc_idx, num_edges_dirc_bc, global_dirc_bc_idx, multiple_dirc_bcs_present)

    ! do some sanity checking
    if (multiple_dirc_bcs_present) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "When BC file specified via -homogeneous_bc_file argument, only one CONDITION_DIRICHLET can be present in the yaml")
    endif
    if (global_dirc_bc_idx == -1) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "No Dirichlet BC specified in the yaml file")
    endif

    bc_dataset%ndata            = num_edges_dirc_bc * 3
    bc_dataset%dirichlet_bc_idx = global_dirc_bc_idx

    allocate(bc_dataset%data_for_rdycore(bc_dataset%ndata))

  end subroutine DoPostprocessForBoundaryHomogeneousDataset

  ! Set up boundary condition based on command line options
  subroutine CreateBoundaryConditionDataset(rdy_, bc_dataset)
    !
    use rdycore
    use petsc
    !
      implicit none
    !
    type(RDy)               :: rdy_
    type(BoundaryCondition) :: bc_dataset
    !
    PetscInt  :: expected_data_stride

    select case (bc_dataset%datatype)
    case (DATASET_UNSET)
      ! do nothing
    case (DATASET_HOMOGENEOUS)
      call OpenHomogeneousDataset(bc_dataset%homogeneous)
      call DoPostprocessForBoundaryHomogeneousDataset(rdy_, bc_dataset)
    case (DATASET_UNSTRUCTURED)
      expected_data_stride = 3;
      !call OpenUnstructuredDataset(bc_dataset%unstructured, expected_data_stride));
      !call DoPostprocessForBoundaryUnstructuredDataset(rdy_, bc_dataset));
    case default
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "More than one boundary condition type cannot be specified")
    end select

  end subroutine CreateBoundaryConditionDataset

  ! Destroy boundary condition
  subroutine DestroyBoundaryConditionDataset(rdy_, bc_dataset)
    !
    use rdycore
    use petsc
    !
      implicit none
    !
    type(RDy)               :: rdy_
    type(BoundaryCondition) :: bc_dataset

    select case (bc_dataset%datatype)
    case (DATASET_UNSET)
      ! do nothing
    case (DATASET_HOMOGENEOUS)
      call DestroyHomogeneousDataset(bc_dataset%homogeneous)
    case (DATASET_UNSTRUCTURED)
    case default
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "More than one boundary condition type cannot be specified")
    end select

  end subroutine DestroyBoundaryConditionDataset

  ! Set saptially homogeneous water height boundary condition for all boundary cells
  subroutine SetHomogeneousBoundary(bc_data, cur_time, num_values, data_for_rdycore)
    !
    use rdycore
    use petsc
    !
      implicit none
    !
    type(HomogeneousDataset) :: bc_data
    PetscReal                :: cur_time
    PetscInt                 :: num_values
    PetscScalar, pointer     :: data_for_rdycore(:)
    !
    PetscInt                 :: i, ndata
    PetscInt                 :: cur_bc_idx, prev_bc_idx
    PetscReal                :: cur_bc
    PetscBool                :: temporally_interpolate

    cur_bc_idx             = bc_data%cur_idx
    prev_bc_idx            = bc_data%prev_idx
    temporally_interpolate = bc_data%temporally_interpolate
    ndata                  = bc_data%ndata

    call GetCurrentData(bc_data%data_ptr, ndata, cur_time, temporally_interpolate, cur_bc_idx, cur_bc);

    do i = 1, num_values
      data_for_rdycore((i - 1)* 3 + 1) = cur_bc
      data_for_rdycore((i - 1)* 3 + 2) = 0.d0
      data_for_rdycore((i - 1)* 3 + 3) = 0.d0
    enddo

  end subroutine SetHomogeneousBoundary

  ! Apply boundary condition to the RDycore object
  subroutine ApplyBoundaryCondition(rdy_, cur_time, bc_dataset)
#include <petsc/finclude/petsc.h>
#include <finclude/rdycore.h>
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)               :: rdy_
    type(BoundaryCondition) :: bc_dataset
    PetscReal               :: cur_time
    !
    PetscErrorCode          :: ierr

    select case (bc_dataset%datatype)
    case (DATASET_UNSET)
      ! do nothing
    case (DATASET_HOMOGENEOUS)
      if (bc_dataset%ndata > 0) then
        call SetHomogeneousBoundary(bc_dataset%homogeneous, cur_time, bc_dataset%ndata / 3, bc_dataset%data_for_rdycore)
        PetscCallA(RDySetFlowDirichletBoundaryValues(rdy_, bc_dataset%dirichlet_bc_idx, bc_dataset%ndata / 3, 3, bc_dataset%data_for_rdycore, ierr))
      endif
    case (DATASET_UNSTRUCTURED)
    case default
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "More than one boundary condition type cannot be specified")
    end select

  end subroutine ApplyBoundaryCondition

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
  PetscReal, pointer   :: rain(:), values(:)
  PetscInt,  pointer   :: nat_id(:)
  integer(RDyTimeUnit) :: time_unit
  PetscReal            :: time, time_step, prev_time, coupling_interval, cur_time

  PetscBool            :: rain_specified, bc_specified
  character(len=1024)  :: rainfile, bcfile
  Vec                  :: rain_vec, bc_vec
  PetscScalar, pointer :: rain_ptr(:)
  PetscInt             :: nrain, nbc, num_edges_dirc_bc
  PetscInt             :: dirc_bc_idx, global_dirc_bc_idx
  PetscInt             :: cur_rain_idx, prev_rain_idx
  PetscInt             :: cur_bc_idx, prev_bc_idx
  PetscReal            :: cur_rain, cur_bc
  PetscBool            :: interpolate_rain, flg
  type(BoundaryCondition) :: bc_dataset
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

      ! Flags to temporally interpolate rain and bc forcing
      interpolate_rain = PETSC_FALSE
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-temporally_interpolate_rain', interpolate_rain, flg, ierr))

      call ParseBoundaryDataOptions(bc_dataset)

      if (rain_specified) then
        call opendata(rainfile, rain_vec, nrain)
        PetscCallA(VecGetArray(rain_vec, rain_ptr, ierr))
      endif

      ! create rdycore and set it up with the given file
      PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
      PetscCallA(RDySetup(rdy_, ierr))

      call CreateBoundaryConditionDataset(rdy_, bc_dataset)

      ! allocate arrays for inspecting simulation data
      PetscCallA(RDyGetNumLocalCells(rdy_, n, ierr))
      allocate(rain(n), values(n))

      ! run the simulation to completion using the time parameters in the
      ! config file
      PetscCallA(RDyGetTimeUnit(rdy_, time_unit, ierr))
      PetscCallA(RDyGetTime(rdy_, time_unit, prev_time, ierr))
      PetscCallA(RDyGetCouplingInterval(rdy_, time_unit, coupling_interval, ierr))
      PetscCallA(RDySetCouplingInterval(rdy_, time_unit, coupling_interval, ierr))

      prev_rain_idx = 0
      prev_bc_idx = 0

      do while (.not. RDyFinished(rdy_)) ! returns true based on stopping criteria

        PetscCallA(RDyGetTime(rdy_, time_unit, cur_time, ierr))

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

        call ApplyBoundaryCondition(rdy_, cur_time, bc_dataset);

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

      end do

      deallocate(rain, values)
      if (rain_specified) then
        PetscCallA(VecRestoreArray(rain_vec, rain_ptr, ierr))
        PetscCallA(VecDestroy(rain_vec, ierr))
      endif

      call DestroyBoundaryConditionDataset(rdy_, bc_dataset)

      PetscCallA(RDyDestroy(rdy_, ierr))
    end if

    PetscCallA(RDyFinalize(ierr));
  end if

end program
