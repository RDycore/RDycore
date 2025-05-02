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

#define PETSC_ID_TYPE "int32"

  type :: time_struct
    PetscInt :: year
    PetscInt :: month
    PetscInt :: day
    PetscInt :: hour
    PetscInt :: minute
  end type time_struct

  type, public :: ConstantDataset
    PetscReal :: rate
  end type ConstantDataset

  type, public :: HomogeneousDataset
    character(len=1024) :: filename
    Vec :: data_vec
    PetscInt :: ndata
    PetscScalar, pointer :: data_ptr(:)
    PetscBool    :: temporally_interpolate
    PetscInt     :: cur_idx, prev_idx
  end type HomogeneousDataset

  type, public :: RasterDataset
    character(len=1024) :: dir
    character(len=1024) :: file
    character(len=1024) :: map_file

    type(time_struct)  :: start_date, current_date

    PetscInt :: header_offset

    Vec :: data_vec
    PetscScalar, pointer :: data_ptr(:)

    ! temporal duration of the rainfall dataset
    PetscReal :: dtime_in_hour
    PetscInt  :: ndata_file

    ! header of dataset
    PetscInt :: ncols, nrows
    PetscReal :: xlc, ylc
    PetscReal :: cellsize

    PetscInt           :: mesh_ncells_local      ! number of cells locally owned
    PetscInt, pointer  :: data2mesh_idx(:)       ! for each RDycore element (cells or boundary edges), the index of the data in the raster dataset
    PetscReal, pointer :: data_xc(:), data_yc(:) ! x and y coordinates of data
    PetscReal, pointer :: mesh_xc(:), mesh_yc(:) ! x and y coordinates of RDycore elments

    PetscBool :: write_map_for_debugging         ! if true, write the mapping between the RDycore cells and the dataset for debugging
    PetscBool :: write_map                       ! if true, write the map between the RDycore cells and the dataset
    PetscBool :: read_map                        ! if true, read the map between the RDycore cells and the dataset
  end type RasterDataset

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

    PetscInt :: header_offset

    PetscInt           :: mesh_nelements         ! number of cells or boundary edges in RDycore mesh
    PetscInt, pointer  :: data2mesh_idx(:)       ! for each RDycore element (cells or boundary edges), the index of the data in the unstructured dataset
    PetscReal, pointer :: data_xc(:), data_yc(:) ! x and y coordinates of data
    PetscReal, pointer :: mesh_xc(:), mesh_yc(:) ! x and y coordinates of RDycore elments

    PetscBool :: write_map_for_debugging         ! if true, write the mapping between the RDycore cells and the dataset for debugging
    PetscBool :: write_map                       ! if true, write the map between the RDycore cells and the dataset
    PetscBool :: read_map                        ! if true, read the map between the RDycore cells and the dataset

  end type UnstructuredDataset

  type, public :: SourceSink
    PetscInt :: datatype

    type(ConstantDataset)     :: constant
    type(HomogeneousDataset)  :: homogeneous
    type(RasterDataset)       :: raster
    type(UnstructuredDataset) :: unstructured

    PetscInt             :: ndata
    PetscScalar, pointer :: data_for_rdycore(:)
  end type SourceSink

  type, public :: BoundaryCondition
    PetscInt :: datatype

    type(HomogeneousDataset)  :: homogeneous
    type(UnstructuredDataset) :: unstructured

    PetscInt             :: ndata
    PetscInt             :: dirichlet_bc_idx
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

  subroutine ParseRainfallDataOptions(rain)
    !
    implicit none
    !
    type(SourceSink) :: rain
    !
    PetscInt                      :: dataset_type_count
    PetscBool                     :: flg
    PetscBool                     :: constant_rain_flag
    PetscBool                     :: homogeneous_rain_flag
    PetscBool                     :: raster_rain_flag
    PetscBool                     :: raster_start_date_flag
    PetscBool                     :: unstructured_rain_flag
    PetscBool                     :: unstructured_start_date_flag
    PetscInt,pointer,dimension(:) :: date
    PetscInt                      :: ndate = 5
    PetscErrorCode                :: ierr

    dataset_type_count = 0

    rain%datatype                           = DATASET_UNSET
    rain%ndata                              = 0
    rain%constant%rate                      = 0.d0
    rain%homogeneous%temporally_interpolate = PETSC_FALSE

    ! Constant rainfall dataset
    PetscCallA(PetscOptionsGetReal(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-constant_rain_rate', rain%constant%rate, constant_rain_flag, ierr))
    if (constant_rain_flag) then
      rain%datatype = DATASET_CONSTANT
      dataset_type_count = dataset_type_count + 1
    endif

    ! Homogeneous rainfall dataset
    PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-temporally_interpolate_spatially_homogeneous_rain', rain%homogeneous%temporally_interpolate, flg, ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-homogeneous_rain_file', rain%homogeneous%filename, homogeneous_rain_flag, ierr))
    if (homogeneous_rain_flag) then
      rain%datatype = DATASET_HOMOGENEOUS
      dataset_type_count = dataset_type_count + 1
    endif

    ! Raster rainfall dataset
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-raster_rain_dir', rain%raster%dir, raster_rain_flag, ierr))
    PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-raster_rain_write_map_for_debugging', rain%raster%write_map_for_debugging, flg, ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-raster_rain_write_map_file', rain%raster%map_file, rain%raster%write_map, ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-raster_rain_read_map_file', rain%raster%map_file, rain%raster%read_map, ierr))
    if (raster_rain_flag) then
      rain%datatype = DATASET_RASTER
      dataset_type_count = dataset_type_count + 1

      allocate(date(ndate))
      PetscCall(PetscOptionsGetIntArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER, "-raster_rain_start_date", date, ndate, raster_start_date_flag, ierr));
      if (ndate /= 5) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "-raster_rain_start_date should be in YY,MO,DD,HH,MM format")
      endif
      rain%raster%start_date%year   = date(1)
      rain%raster%start_date%month  = date(2)
      rain%raster%start_date%day    = date(3)
      rain%raster%start_date%hour   = date(4)
      rain%raster%start_date%minute = date(5)

      rain%raster%current_date%year   = date(1)
      rain%raster%current_date%month  = date(2)
      rain%raster%current_date%day    = date(3)
      rain%raster%current_date%hour   = date(4)
      rain%raster%current_date%minute = date(5)

      deallocate(date)
    endif

    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-unstructured_rain_dir', rain%unstructured%dir, unstructured_rain_flag, ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-unstructured_rain_write_map_file', rain%unstructured%map_file, rain%unstructured%write_map, ierr))
    PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-unstructured_rain_read_map_file', rain%unstructured%map_file, rain%unstructured%read_map, ierr))
    if (unstructured_rain_flag) then
      rain%datatype = DATASET_UNSTRUCTURED
      dataset_type_count = dataset_type_count + 1

      allocate(date(ndate))
      PetscCall(PetscOptionsGetIntArray(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER, "-unstructured_rain_start_date", date, ndate, unstructured_start_date_flag, ierr));
      if (ndate /= 5) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "-unstructured_rain_start_date should be in YY,MO,DD,HH,MM format")
      endif

      PetscCall(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, "-unstructured_rain_mesh_file", rain%unstructured%mesh_file, flg, ierr))
      if (flg .eqv. PETSC_FALSE) then
        SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Need to specify the mesh file -unstructured_rain_mesh_file <file>")
      endif

      rain%unstructured%start_date%year   = date(1)
      rain%unstructured%start_date%month  = date(2)
      rain%unstructured%start_date%day    = date(3)
      rain%unstructured%start_date%hour   = date(4)
      rain%unstructured%start_date%minute = date(5)

      rain%unstructured%current_date%year   = date(1)
      rain%unstructured%current_date%month  = date(2)
      rain%unstructured%current_date%day    = date(3)
      rain%unstructured%current_date%hour   = date(4)
      rain%unstructured%current_date%minute = date(5)

      deallocate(date)
    endif

    if (dataset_type_count > 1) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "More than one rainfall dataset type cannot be specified")
    endif

  end subroutine ParseRainfallDataOptions


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

    allocate(date(ndate))
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
    type(RDy)      :: rdy_
    PetscInt       :: dirc_bc_idx, num_edges_dirc_bc
    PetscInt       :: global_dirc_bc_idx
    PetscBool      :: multiple_dirc_bcs_present
    !
    PetscInt       :: ibcond, nbconds, num_edges, bcond_type
    PetscErrorCode :: ierr

    dirc_bc_idx               = -1
    global_dirc_bc_idx        = -1
    num_edges_dirc_bc         = 0
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

    call MPI_Allreduce(dirc_bc_idx, global_dirc_bc_idx, 1, MPIU_INTEGER, MPI_MAX, PETSC_COMM_WORLD, ierr)

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
    PetscInt    :: global_dirc_bc_idx
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

  subroutine DetermineDatasetFilename(dir, current_date, file)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    character(len=1024) :: dir
    type(time_struct)   :: current_date
    character(len=1024) :: file
    !
    PetscErrorCode      :: ierr

    write(file, '(A,"/",I4,"-",I2.2,"-",I2.2,":",I2.2,"-",I2.2,".",A,".bin")') trim(dir), current_date%year, current_date%month,  current_date%day, current_date%hour, current_date%minute, PETSC_ID_TYPE

  end subroutine DetermineDatasetFilename

  subroutine OpenRasterDataset(data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RasterDataset) :: data
    !
    character(len=PETSC_MAX_PATH_LEN) :: outputString
    PetscInt                          :: tmpInt
    PetscErrorCode                    :: ierr

    call DetermineDatasetFilename(data%dir, data%current_date, data%file)
    write(outputString,'(a,a,a)') 'Opening  ',trim(data%file),'\n'
    PetscCallA(PetscPrintf(PETSC_COMM_WORLD, outputString, ierr))

    data%dtime_in_hour = 1.0
    data%ndata_file    = 1

    call opendata(data%file, data%data_vec, tmpInt)
    PetscCallA(VecGetArray(data%data_vec, data%data_ptr, ierr))

    data%header_offset = 5

    data%ncols    = int(data%data_ptr(1))
    data%nrows    = int(data%data_ptr(2))
    data%xlc      = data%data_ptr(3)
    data%ylc      = data%data_ptr(4)
    data%cellsize = data%data_ptr(5)

  end subroutine OpenRasterDataset

  subroutine OpenNextRasterDataset(data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RasterDataset) :: data
    !
    character(len=PETSC_MAX_PATH_LEN) :: outputString
    PetscInt                          :: tmpInt
    PetscErrorCode                    :: ierr

    PetscCallA(VecRestoreArray(data%data_vec, data%data_ptr, ierr))
    PetscCallA(VecDestroy(data%data_vec, ierr))

    data%current_date%hour = data%current_date%hour + 1

    call OpenRasterDataset(data)

  end subroutine OpenNextRasterDataset

  subroutine OpenUnstructuredDataset(data, expected_data_stride)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(UnstructuredDataset) :: data
    PetscInt                  :: expected_data_stride
    !
    PetscInt                  :: size
    PetscErrorCode            :: ierr

    call DetermineDatasetFilename(data%dir, data%current_date, data%file)

    data%dtime_in_hour = 1.0
    data%ndata_file    = 1

    ! open the dataset file and read the data into a vector
    call opendata(data%file, data%data_vec, size)

    ! get the data pointer to the data in the vector
    PetscCallA(VecGetArray(data%data_vec, data%data_ptr, ierr))
    PetscCallA(VecGetSize(data%data_vec, size, ierr))

    data%header_offset = 2
    data%ndata         = data%data_ptr(1)
    data%stride        = data%data_ptr(2)

    if ((size - 2) / data%stride /= data%ndata) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "The number of data points in the unstructured dataset is not equal to the expected number of data points")
    endif

    if (data%stride /= expected_data_stride) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "The stride of the unstructured dataset is not equal to the expected stride")
    endif

  end subroutine OpenUnstructuredDataset

  subroutine OpenNextUnstructuredDataset(data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(UnstructuredDataset) :: data
    !
    character(len=PETSC_MAX_PATH_LEN) :: outputString
    PetscInt                          :: tmpInt
    PetscErrorCode                    :: ierr

    PetscCallA(VecRestoreArray(data%data_vec, data%data_ptr, ierr))
    PetscCallA(VecDestroy(data%data_vec, ierr))

    data%current_date%hour = data%current_date%hour + 1

    call OpenUnstructuredDataset(data, data%stride)

  end subroutine OpenNextUnstructuredDataset

  subroutine WriteMap(rdy_, filename, ncells, d2m)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)           :: rdy_
    character(len=1024) :: filename
    PetscInt            :: ncells
    PetscInt, pointer   :: d2m(:)
    !
    Vec                  :: global_vec
    PetscScalar, pointer :: global_ptr(:)
    PetscInt             :: i
    PetscErrorCode       :: ierr

    PetscCallA(RDyCreateOneDOFGlobalVec(rdy_, global_vec, ierr))
    PetscCallA(VecGetArray(global_vec, global_ptr, ierr))
    do i = 1, ncells
      global_ptr(i) = d2m(i)
    enddo
    PetscCallA(VecRestoreArray(global_vec, global_ptr, ierr))

    PetscCallA(RDyWriteOneDOFGlobalVecToBinaryFile(rdy_, filename, global_vec, ierr))
    PetscCallA(VecDestroy(global_vec, ierr))

  end subroutine WriteMap

  subroutine DoPostprocessForBoundaryUnstructuredDataset(rdy_, bc_dataset)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)               :: rdy_
    type(BoundaryCondition) :: bc_dataset
    !
    PetscInt               :: dirc_bc_idx, num_edges_dirc_bc
    PetscInt               :: global_dirc_bc_idx
    PetscBool              :: multiple_dirc_bcs_present

    call FindDirichletBCID(rdy_, dirc_bc_idx, num_edges_dirc_bc, global_dirc_bc_idx, multiple_dirc_bcs_present)

    ! do some sanity checking
    if (multiple_dirc_bcs_present) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "When BC file specified via -unstructured_bc_file argument, only one CONDITION_DIRICHLET can be present in the yaml")
    endif
    if (global_dirc_bc_idx == -1) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "No Dirichlet BC specified in the yaml file")
    endif

    bc_dataset%ndata            = num_edges_dirc_bc * 3
    bc_dataset%dirichlet_bc_idx = global_dirc_bc_idx

    allocate(bc_dataset%data_for_rdycore(bc_dataset%ndata))
    allocate(bc_dataset%unstructured%mesh_xc(num_edges_dirc_bc))
    allocate(bc_dataset%unstructured%mesh_yc(num_edges_dirc_bc))
    allocate(bc_dataset%unstructured%data2mesh_idx(bc_dataset%ndata))

    if (bc_dataset%ndata > 0) then
      bc_dataset%unstructured%mesh_nelements = num_edges_dirc_bc

      call GetBoundaryEdgeCentroidsFromRDycoreMesh(rdy_, num_edges_dirc_bc, global_dirc_bc_idx, bc_dataset%unstructured%mesh_xc, bc_dataset%unstructured%mesh_yc)

      call ReadUnstructuredDatasetCoordinates(bc_dataset%unstructured)

      ! set up the mapping between the dataset and boundary edges
      call CreateUnstructuredDatasetMap(bc_dataset%unstructured)

    endif

  end subroutine DoPostprocessForBoundaryUnstructuredDataset

  subroutine GetCellCentroidsFromRDycoreMesh(rdy_, n, xc, yc)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)               :: rdy_
    PetscInt                :: n
    PetscReal, pointer      :: xc(:), yc(:)
    PetscErrorCode          :: ierr

    PetscCallA(RDyGetLocalCellXCentroids(rdy_, n, xc, ierr))
    PetscCallA(RDyGetLocalCellYCentroids(rdy_, n, yc, ierr))

  end subroutine GetCellCentroidsFromRDycoreMesh

  subroutine GetBoundaryEdgeCentroidsFromRDycoreMesh(rdy_, n, idx, xc, yc)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)               :: rdy_
    PetscInt                :: n, idx
    PetscReal, pointer      :: xc(:), yc(:)
    PetscErrorCode          :: ierr

    PetscCallA(RDyGetBoundaryEdgeXCentroids(rdy_, idx, n, xc, ierr))
    PetscCallA(RDyGetBoundaryEdgeYCentroids(rdy_, idx, n, yc, ierr))

  end subroutine GetBoundaryEdgeCentroidsFromRDycoreMesh

  subroutine ReadUnstructuredDatasetCoordinates(data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(UnstructuredDataset) :: data
    !
    PetscInt                 :: ndata, stride, offset, i
    PetscScalar, pointer     :: vec_ptr(:)
    Vec                      :: vec
    PetscViewer              :: viewer
    PetscErrorCode           :: ierr

    PetscCallA(VecCreate(PETSC_COMM_SELF, vec, ierr))
    PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF, data%mesh_file, FILE_MODE_READ, viewer, ierr))
    PetscCallA(VecLoad(vec, viewer, ierr))
    PetscCallA(PetscViewerDestroy(viewer, ierr))

    PetscCallA(VecGetArray(vec, vec_ptr, ierr))

    data%ndata = vec_ptr(1)
    allocate(data%data_xc(data%ndata))
    allocate(data%data_yc(data%ndata))

    stride = vec_ptr(2)
    if (stride /= 2) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "The stride of the unstructured dataset is not equal to 2")
    endif

    offset = 2;
    do i = 1, data%ndata
      data%data_xc(i) = vec_ptr(offset + (i - 1) * stride + 1)
      data%data_yc(i) = vec_ptr(offset + (i - 1) * stride + 2)
    enddo

    PetscCallA(VecRestoreArray(vec, vec_ptr, ierr))
    PetscCallA(VecDestroy(vec, ierr))

  end subroutine ReadUnstructuredDatasetCoordinates

  subroutine CreateUnstructuredDatasetMap(data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(UnstructuredDataset) :: data
    !
    PetscInt                 :: ndata, icell, i, count
    PetscReal                :: xc, yc, dx, dy, dist, min_dist

    ndata = data%ndata

    do icell = 1, data%mesh_nelements
      xc = data%mesh_xc(icell)
      yc = data%mesh_yc(icell)

      count = 0
      do i = 1, data%ndata
        count = count + 1
        dx = xc - data%data_xc(count)
        dy = yc - data%data_yc(count)
        dist = sqrt(dx * dx + dy * dy)
        if (i == 1) then
          min_dist = dist
          data%data2mesh_idx(icell) = count
        else
          if (dist < min_dist) then
            min_dist = dist
            data%data2mesh_idx(icell) = count
          endif
        endif
      enddo
    enddo

  end subroutine CreateUnstructuredDatasetMap

  subroutine ReadRainfallDatasetMap(rdy_, filename, ncells, d2m)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)           :: rdy_
    character(len=1024) :: filename
    PetscInt            :: ncells
    PetscInt, pointer   :: d2m(:)
    !
    Vec                  :: global_vec
    PetscScalar, pointer :: global_ptr(:)
    PetscInt             :: size, i
    PetscErrorCode       :: ierr

    PetscCallA(RDyReadOneDOFGlobalVecFromBinaryFile(rdy_, filename, global_vec, ncells))

    ! get the data pointer to the data in the vector
    PetscCallA(VecGetArray(global_vec, global_ptr, ierr))
    PetscCallA(VecGetSize(global_vec, size, ierr))
    if (size /= ncells) then
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "The number of cells in the rainfall dataset map is not equal to the expected number of cells")
    endif

    do i = 1, ncells
      d2m(i) = global_ptr(i)
    enddo

    PetscCallA(VecRestoreArray(global_vec, global_ptr, ierr))
    PetscCallA(VecDestroy(global_vec, ierr))

  end subroutine ReadRainfallDatasetMap

  subroutine CreateRasterDatasetMapping(rdy_, data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)               :: rdy_
    type(RasterDataset)     :: data
    !
    PetscInt               :: ndata, icell, irow, icol, count
    PetscReal              :: xc, yc, dx, dy, dist, min_dist

    ndata = data%ncols * data%nrows

    do icell = 1, data%mesh_ncells_local
      xc = data%mesh_xc(icell)
      yc = data%mesh_yc(icell)
      min_dist = (max(data%ncols, data%nrows) + 1.d0) * data%cellsize

      count = 0
      do irow = 1, data%nrows
        do icol = 1, data%ncols
          count = count + 1
          dx = xc - data%data_xc(count)
          dy = yc - data%data_yc(count)
          dist = sqrt(dx * dx + dy * dy)
          if (dist < min_dist) then
            min_dist = dist
            data%data2mesh_idx(icell) = count
          endif
        enddo
      enddo
    enddo

  end subroutine CreateRasterDatasetMapping

  subroutine DoPostprocessForSourceRasterDataset(rdy_, data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)           :: rdy_
    type(RasterDataset) :: data
    !
    PetscInt            :: ndata, irow, icol, count
    PetscErrorCode      :: ierr

    ndata = data%ncols * data%nrows

    allocate(data%data_xc(ndata))
    allocate(data%data_yc(ndata))

    count = 0
    do irow = 1, data%nrows
      do icol = 1, data%ncols
        count = count + 1
        data%data_xc(count) = data%xlc + (icol - 1) * data%cellsize + data%cellsize / 2.d0
        data%data_yc(count) = data%ylc + (data%nrows - irow) * data%cellsize + data%cellsize / 2.d0
      enddo
    enddo

    PetscCallA(RDyGetNumLocalCells(rdy_, data%mesh_ncells_local, ierr))
    allocate(data%mesh_xc(data%mesh_ncells_local))
    allocate(data%mesh_yc(data%mesh_ncells_local))
    allocate(data%data2mesh_idx(data%mesh_ncells_local))

    call GetCellCentroidsFromRDycoreMesh(rdy_, data%mesh_ncells_local, data%mesh_xc, data%mesh_yc)

    call CreateRasterDatasetMapping(rdy_, data)

    if (data%write_map) then
      call WriteMap(rdy_, data%map_file, data%mesh_ncells_local, data%data2mesh_idx)
    endif
    if (data%read_map) then
      call ReadRainfallDatasetMap(rdy_, data%map_file, data%mesh_ncells_local, data%data2mesh_idx)
    endif

  end subroutine DoPostprocessForSourceRasterDataset

  subroutine DoPostprocessForSourceUnstructuredDataset(rdy_, data)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)                 :: rdy_
    type(UnstructuredDataset) :: data
    !
    PetscErrorCode            :: ierr

    PetscCallA(RDyGetNumLocalCells(rdy_, data%mesh_nelements, ierr))
    allocate(data%mesh_xc(data%mesh_nelements))
    allocate(data%mesh_yc(data%mesh_nelements))
    allocate(data%data2mesh_idx(data%mesh_nelements))

    call GetCellCentroidsFromRDycoreMesh(rdy_, data%mesh_nelements, data%mesh_xc, data%mesh_yc)
    call ReadUnstructuredDatasetCoordinates(data)
    call CreateUnstructuredDatasetMap(data)

    if (data%write_map) then
      call WriteMap(rdy_, data%map_file, data%mesh_nelements, data%data2mesh_idx)
    endif

    if (data%read_map) then
      call ReadRainfallDatasetMap(rdy_, data%map_file, data%mesh_nelements, data%data2mesh_idx)
    endif

  end subroutine DoPostprocessForSourceUnstructuredDataset

  subroutine CreateRainfallConditionDataset(rdy_, n, rain_dataset)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)               :: rdy_
    PetscInt                :: n
    type(SourceSink)        :: rain_dataset
    !
    PetscInt  :: expected_data_stride

    select case (rain_dataset%datatype)
    case (DATASET_UNSET)
      ! do nothing
    case (DATASET_CONSTANT)
      rain_dataset%ndata = n
      allocate(rain_dataset%data_for_rdycore(n))
    case (DATASET_HOMOGENEOUS)
      rain_dataset%ndata = n
      allocate(rain_dataset%data_for_rdycore(n))
      call OpenHomogeneousDataset(rain_dataset%homogeneous)
    case (DATASET_RASTER)
      rain_dataset%ndata = n
      allocate(rain_dataset%data_for_rdycore(n))
      call OpenRasterDataset(rain_dataset%raster)
      call DoPostprocessForSourceRasterDataset(rdy_, rain_dataset%raster)
    case (DATASET_UNSTRUCTURED)
      expected_data_stride = 1
      rain_dataset%ndata = n
      allocate(rain_dataset%data_for_rdycore(n))
      call OpenUnstructuredDataset(rain_dataset%unstructured, expected_data_stride)
      call DoPostprocessForSourceUnstructuredDataset(rdy_, rain_dataset%unstructured)
    case default
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "More than one rainfall condition type cannot be specified")
    end select

  end subroutine CreateRainfallConditionDataset

  subroutine SetConstantRainfall(rate, num_values, data_for_rdycore)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    PetscReal                :: rate
    PetscInt                 :: num_values
    PetscScalar, pointer     :: data_for_rdycore(:)
    !
    PetscInt                 :: i

    do i = 1, num_values
      data_for_rdycore(i) = rate
    enddo

  end subroutine SetConstantRainfall

  subroutine SetHomogeneousData(rain_data, cur_time, num_values, data_for_rdycore)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(HomogeneousDataset) :: rain_data
    PetscReal                :: cur_time
    PetscInt                 :: num_values
    PetscScalar, pointer     :: data_for_rdycore(:)
    !
    PetscInt                 :: i, ndata
    PetscInt                 :: cur_rain_idx, prev_rain_idx
    PetscReal                :: cur_rain
    PetscBool                :: temporally_interpolate

    cur_rain_idx             = rain_data%cur_idx
    prev_rain_idx            = rain_data%prev_idx
    temporally_interpolate   = rain_data%temporally_interpolate
    ndata                    = rain_data%ndata

    call GetCurrentData(rain_data%data_ptr, ndata, cur_time, temporally_interpolate, cur_rain_idx, cur_rain);

    do i = 1, num_values
      data_for_rdycore(i) = cur_rain
    enddo

  end subroutine SetHomogeneousData

  subroutine SetRasterData(data, cur_time, num_values, data_for_rdycore)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RasterDataset) :: data
    PetscReal           :: cur_time
    PetscInt            :: num_values
    PetscScalar, pointer :: data_for_rdycore(:)
    !
    PetscInt            :: icell, idx, ndata, ndata_file
    PetscInt            :: offset
    PetscReal, parameter :: mm_per_hr_2_m_per_sec = 1.0 / (1000.d0 * 3600.d0)

    if (cur_time / 3600.d0 >= (data%ndata_file) * data%dtime_in_hour) then
      ndata_file = data%ndata_file
      call OpenNextRasterDataset(data)
      data%ndata_file = ndata_file + 1
    endif

    offset = data%header_offset;
    do icell = 1, num_values
      idx = data%data2mesh_idx(icell)
      data_for_rdycore(icell) = data%data_ptr(idx + offset) * mm_per_hr_2_m_per_sec
    enddo

  end subroutine SetRasterData

  ! Apply rainfall source to the RDycore object
  subroutine ApplyRainfallDataset(rdy_, cur_time, rain_dataset)
#include <petsc/finclude/petsc.h>
#include <finclude/rdycore.h>
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RDy)               :: rdy_
    type(SourceSink)        :: rain_dataset
    PetscReal               :: cur_time
    !
    PetscInt, parameter     :: region_idx = 1
    PetscErrorCode          :: ierr

    select case (rain_dataset%datatype)
    case (DATASET_UNSET)
      ! do nothing
    case (DATASET_CONSTANT)
      if (rain_dataset%ndata > 0) then
        call SetConstantRainfall(rain_dataset%constant%rate, rain_dataset%ndata, rain_dataset%data_for_rdycore)
        call RDySetRegionalWaterSource(rdy_, region_idx, rain_dataset%ndata, rain_dataset%data_for_rdycore, ierr)
      endif
    case (DATASET_HOMOGENEOUS)
      if (rain_dataset%ndata > 0) then
        call SetHomogeneousData(rain_dataset%homogeneous, cur_time, rain_dataset%ndata, rain_dataset%data_for_rdycore)
        call RDySetRegionalWaterSource(rdy_, region_idx, rain_dataset%ndata, rain_dataset%data_for_rdycore, ierr)
      endif
    case (DATASET_RASTER)
      if (rain_dataset%ndata > 0) then
        call SetRasterData(rain_dataset%raster, cur_time, rain_dataset%ndata, rain_dataset%data_for_rdycore)
        call RDySetRegionalWaterSource(rdy_, region_idx, rain_dataset%ndata, rain_dataset%data_for_rdycore, ierr)
      endif
    case (DATASET_UNSTRUCTURED)
      if (rain_dataset%ndata > 0) then
        call SetUnstructuredData(rain_dataset%unstructured, cur_time, rain_dataset%ndata, rain_dataset%data_for_rdycore)
        call RDySetRegionalWaterSource(rdy_, region_idx, rain_dataset%ndata, rain_dataset%data_for_rdycore, ierr)
      endif
    case default
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend this to other types of rainfall datasets")
    end select

  end subroutine ApplyRainfallDataset

  subroutine DestroyRasterDataset(rain_dataset)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(RasterDataset) :: rain_dataset
    !
    PetscErrorCode      :: ierr

    ! get the data pointer to the data in the vector
    PetscCallA(VecRestoreArray(rain_dataset%data_vec, rain_dataset%data_ptr, ierr))
    PetscCallA(VecDestroy(rain_dataset%data_vec, ierr))

  end subroutine DestroyRasterDataset

  subroutine DestroyUnstructuredDataset(rain_dataset)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(UnstructuredDataset) :: rain_dataset
    !
    PetscErrorCode      :: ierr

    ! get the data pointer to the data in the vector
    PetscCallA(VecRestoreArray(rain_dataset%data_vec, rain_dataset%data_ptr, ierr))
    PetscCallA(VecDestroy(rain_dataset%data_vec, ierr))

  end subroutine DestroyUnstructuredDataset

  subroutine DestroyRainfallDataset(rain_dataset)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(SourceSink) :: rain_dataset

    select case (rain_dataset%datatype)
    case (DATASET_UNSET)
      ! do nothing
    case (DATASET_CONSTANT)
      ! do nothing
    case (DATASET_HOMOGENEOUS)
      call DestroyHomogeneousDataset(rain_dataset%homogeneous)
      deallocate(rain_dataset%data_for_rdycore)
    case (DATASET_RASTER)
      call DestroyRasterDataset(rain_dataset%raster)
      deallocate(rain_dataset%data_for_rdycore)
    case (DATASET_UNSTRUCTURED)
      call DestroyUnstructuredDataset(rain_dataset%unstructured)
      deallocate(rain_dataset%data_for_rdycore)
    case default
      SETERRA(PETSC_COMM_WORLD, PETSC_ERR_USER, "More than one rainfall condition type cannot be specified")
    end select

  end subroutine DestroyRainfallDataset

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
      call OpenUnstructuredDataset(bc_dataset%unstructured, expected_data_stride)
      call DoPostprocessForBoundaryUnstructuredDataset(rdy_, bc_dataset)
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

  subroutine SetUnstructuredData(data, cur_time, num_values, data_for_rdycore)
    !
    use rdycore
    use petsc
    !
    implicit none
    !
    type(UnstructuredDataset) :: data
    PetscReal                 :: cur_time
    PetscInt                  :: num_values
    PetscScalar, pointer      :: data_for_rdycore(:)
    !
    PetscInt                 :: ii, icell, idx, ndata_file, stride, offset

    if (cur_time / 3600.d0 >= (data%ndata_file) * data%dtime_in_hour) then
      ndata_file = data%ndata_file
      call OpenNextUnstructuredDataset(data)
      data%ndata_file = ndata_file + 1
    endif

    offset = data%header_offset;
    stride = data%stride

    do icell = 1, num_values
      idx = (data%data2mesh_idx(icell) - 1) * stride
      do ii = 1, stride
        data_for_rdycore((icell - 1) * stride + ii) = data%data_ptr(idx + ii + offset)
      enddo
    enddo

  end subroutine SetUnstructuredData

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
    PetscInt, parameter     :: ndof = 3
    PetscErrorCode          :: ierr

    select case (bc_dataset%datatype)
    case (DATASET_UNSET)
      ! do nothing
    case (DATASET_HOMOGENEOUS)
      if (bc_dataset%ndata > 0) then
        call SetHomogeneousBoundary(bc_dataset%homogeneous, cur_time, bc_dataset%ndata / ndof, bc_dataset%data_for_rdycore)
        PetscCallA(RDySetFlowDirichletBoundaryValues(rdy_, bc_dataset%dirichlet_bc_idx, bc_dataset%ndata / ndof, ndof, bc_dataset%data_for_rdycore, ierr))
      endif
    case (DATASET_UNSTRUCTURED)
      if (bc_dataset%ndata > 0) then
        call SetUnstructuredData(bc_dataset%unstructured, cur_time, bc_dataset%ndata / ndof, bc_dataset%data_for_rdycore)
        PetscCallA(RDySetFlowDirichletBoundaryValues(rdy_, bc_dataset%dirichlet_bc_idx, bc_dataset%ndata / ndof, ndof, bc_dataset%data_for_rdycore, ierr))
      endif
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
  type(SourceSink)        :: rain_dataset
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

      call ParseRainfallDataOptions(rain_dataset)
      call ParseBoundaryDataOptions(bc_dataset)

      if (rain_specified) then
        call opendata(rainfile, rain_vec, nrain)
        PetscCallA(VecGetArray(rain_vec, rain_ptr, ierr))
      endif

      ! create rdycore and set it up with the given file
      PetscCallA(RDyCreate(PETSC_COMM_WORLD, config_file, rdy_, ierr))
      PetscCallA(RDySetup(rdy_, ierr))

      ! allocate arrays for inspecting simulation data
      PetscCallA(RDyGetNumLocalCells(rdy_, n, ierr))
      allocate(rain(n), values(n))

      call CreateRainfallConditionDataset(rdy_, n, rain_dataset)
      call CreateBoundaryConditionDataset(rdy_, bc_dataset)

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

        call ApplyRainfallDataset(rdy_, cur_time, rain_dataset)
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

      call DestroyRainfallDataset(rain_dataset)
      call DestroyBoundaryConditionDataset(rdy_, bc_dataset)

      PetscCallA(RDyDestroy(rdy_, ierr))
    end if

    PetscCallA(RDyFinalize(ierr));
  end if

end program
