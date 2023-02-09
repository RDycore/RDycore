#include <private/rdycoreimpl.h>
#include <private/rdymemory.h>
#include <rdycore.h>

static PetscBool initialized_ = PETSC_FALSE;

/// Initializes a process for use by RDycore. Call this at the beginning of
/// your program
PetscErrorCode RDyInit(int argc, char *argv[], const char *help) {
  PetscFunctionBegin;
  if (!initialized_) {
    PetscCall(PetscInitialize(&argc, &argv, (char *)0, (char *)help));
    initialized_ = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/// Initializes the RDycore library without arguments. It's used by the Fortran
/// interface, which calls PetscInitialize itself and then this function.
PetscErrorCode RDyInitNoArguments(void) {
  PetscFunctionBegin;
  if (!initialized_) {
    PetscCall(PetscInitializeNoArguments());
    initialized_ = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

// Functions called at shutdown. This can be used by all subsystems via
// RDyOnFinalize().
typedef void (*ShutdownFunc)(void);
static ShutdownFunc *shutdown_funcs_     = NULL;
static int           num_shutdown_funcs_ = 0;
static int           shutdown_funcs_cap_ = 0;

/// Call this to register a shutdown function that is called during TDyFinalize.
PetscErrorCode RDyOnFinalize(void (*shutdown_func)(void)) {
  PetscFunctionBegin;
  if (shutdown_funcs_ == NULL) {
    shutdown_funcs_cap_ = 32;
    PetscCall(RDyAlloc(ShutdownFunc, shutdown_funcs_cap_, &shutdown_funcs_));
  } else if (num_shutdown_funcs_ == shutdown_funcs_cap_) {  // need more space!
    shutdown_funcs_cap_ *= 2;
    PetscCall(RDyRealloc(ShutdownFunc, shutdown_funcs_cap_, &shutdown_funcs_));
  }
  shutdown_funcs_[num_shutdown_funcs_] = shutdown_func;
  ++num_shutdown_funcs_;
  PetscFunctionReturn(0);
}

/// Shuts down a process in which RDyInit or RDyInitNotArguments was called.
/// (Has no effect otherwise.)
PetscErrorCode RDyFinalize(void) {
  PetscFunctionBegin;

  // Call shutdown functions in reverse order, and destroy the list.
  if (shutdown_funcs_ != NULL) {
    for (int i = num_shutdown_funcs_ - 1; i >= 0; --i) {
      shutdown_funcs_[i]();
    }
    RDyFree(shutdown_funcs_);
  }

  PetscFinalize();

  initialized_ = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/// Returns PETSC_TRUE if the RDyCore library has been initialized, PETSC_FALSE
/// otherwise.
PetscBool RDyInitialized(void) { return initialized_; }

/// Creates a new RDy object representing an RDycore simulation context.
/// @param comm        [in] the MPI communicator used by the simulation
/// @param config_file [in] a path to a configuration (.yaml) file
/// @param rdy         [out] a pointer that stores the newly created RDy.
PetscErrorCode RDyCreate(MPI_Comm comm, const char *config_file, RDy *rdy) {
  PetscFunctionBegin;

  PetscCall(PetscNew(rdy));

  // MPI comm stuff
  (*rdy)->comm = comm;
  MPI_Comm_rank(comm, &((*rdy)->rank));
  MPI_Comm_size(comm, &((*rdy)->nproc));

  // set the config file
  strncpy((*rdy)->filename, config_file, PETSC_MAX_PATH_LEN - 1);

  PetscFunctionReturn(0);
}

/// Destroys the given RDy object, freeing any allocated resources.
/// @param rdy [out] a pointer to the RDy object to be destroyed.
PetscErrorCode RDyDestroy(RDy *rdy) {
  PetscFunctionBegin;

  // Destroy tables of named flow/sediment/salinity conditions.
  // NOTE: we can make destructors for these things if they get more complex
  for (PetscInt i = 0; i < (*rdy)->num_flow_conditions; ++i) {
    RDyFree((*rdy)->flow_conditions[i].name);
  }
  for (PetscInt i = 0; i < (*rdy)->num_sediment_conditions; ++i) {
    RDyFree((*rdy)->sediment_conditions[i].name);
  }
  for (PetscInt i = 0; i < (*rdy)->num_salinity_conditions; ++i) {
    RDyFree((*rdy)->salinity_conditions[i].name);
  }

  // Destroy regions and surfaces.
  for (PetscInt i = 0; i < (*rdy)->num_regions; ++i) {
    RDyRegionDestroy(&((*rdy)->regions[i]));
  }
  for (PetscInt i = 0; i < (*rdy)->num_surfaces; ++i) {
    RDySurfaceDestroy(&((*rdy)->surfaces[i]));
  }

  if ((*rdy)->dm) {
    DMDestroy(&((*rdy)->dm));
  }

  PetscCall(RDyFree(*rdy));
  *rdy = NULL;
  PetscFunctionReturn(0);
}

// Fills the given region with data. Region storage itself is not allocated and
// must exist.
PetscErrorCode RDyRegionCreate(const char* name,
                               PetscInt num_cells,
                               const PetscInt cell_ids[num_cells],
                               RDyRegion* region) {
  PetscFunctionBegin;

  PetscCall(RDyAlloc(PetscInt, num_cells, &region->cell_ids));
  memcpy(region->cell_ids, cell_ids, sizeof(PetscInt) * num_cells);
  RDyAlloc(char, strlen(name)+1, &region->name);
  strcpy((char*)region->name, name);
  region->num_cells = num_cells;

  PetscFunctionReturn(0);
}

// deallocates region data -- does not deallocate the region itself
PetscErrorCode RDyRegionDestroy(RDyRegion* region) {
  PetscFunctionBegin;

  RDyFree(region->cell_ids);
  RDyFree(region->name);

  PetscFunctionReturn(0);
}

// Fills the given surface with data. Surface storage itself is not allocated
// and must exist.
PetscErrorCode RDySurfaceCreate(const char* name,
                                PetscInt num_edges,
                                const PetscInt edge_ids[num_edges],
                                RDySurface* surface) {
  PetscFunctionBegin;

  PetscCall(RDyAlloc(PetscInt, num_edges, &surface->edge_ids));
  memcpy(surface->edge_ids, edge_ids, sizeof(PetscInt) * num_edges);
  RDyAlloc(char, strlen(name)+1, &surface->name);
  strcpy((char*)surface->name, name);
  surface->num_edges = num_edges;

  PetscFunctionReturn(0);
}

PetscErrorCode RDySurfaceDestroy(RDySurface* surface) {
  PetscFunctionBegin;

  RDyFree(surface->edge_ids);
  RDyFree(surface->name);

  PetscFunctionReturn(0);
}

PetscErrorCode RDyFindRegion(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;
  *index = -1;
  PetscCheck(rdy->num_regions > 0, rdy->comm, PETSC_ERR_USER, "No regions found!");

  // Currently, we do a linear search on the name of the region, which is O(N)
  // for N regions. If this is too slow, we can sort the regions by name and
  // use binary search, which is O(log2 N).
  for (PetscInt i = 0; i < rdy->num_regions; ++i) {
    if (!strcmp(rdy->regions[i].name, name)) {
      *index = i;
      break;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RDyFindSurface(RDy rdy, const char *name, PetscInt *index) {
  PetscFunctionBegin;
  *index = -1;
  PetscCheck(rdy->num_surfaces > 0, rdy->comm, PETSC_ERR_USER, "No surfaces found!");

  // Currently, we do a linear search on the name of the surface, which is O(N)
  // for N regions. If this is too slow, we can sort the surfaces by name and
  // use binary search, which is O(log2 N).
  for (PetscInt i = 0; i < rdy->num_surfaces; ++i) {
    if (!strcmp(rdy->surfaces[i].name, name)) {
      *index = i;
      break;
    }
  }
  PetscFunctionReturn(0);
}
