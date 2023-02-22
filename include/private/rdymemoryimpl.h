#ifndef RDYMEMORYIMPL_H
#define RDYMEMORYIMPL_H

#include <petscsys.h>

// Allocates a block of memory of the given type, consisting of count
// contiguous elements and placing the allocated memory in the given result
// pointer. Memory is zero-initialized. Returns a PetscErrorCode.
// @param [in] type The data type for which storage is allocated
// @param [in] count The number of elements in the requested allocation
// @param [out] result A pointer to the requested memory block
#define RDyAlloc(type, count, result) PetscCalloc1(sizeof(type) * (count), result)

// Resizes the given block of memory to the new requested size.
// @param [in] type The data type for which storage is resized
// @param [in] count The number of elements in the requested re-allocation
// @param [inout] memory A pointer to a previously allocated memory block.
#define RDyRealloc(type, count, memory) PetscRealloc(sizeof(type)*count, memory)

// Frees a block of memory allocated by RDyAlloc. Returns a PetscErrorCode.
// @param [inout] memory A pointer to a previously allocated memory block.
#define RDyFree(memory) PetscFree(memory)

// Fills an array of the given type and given element count with the given
// value, performing an explicit cast for each value. Returns a 0 error code,
// as it cannot fail under detectable conditions.
// NOTE: Note the leading "0", which provides the return code. This trick may
// NOTE: produce an "unused value" warning with certain compiler settings if
// NOTE: the error code is not captured by the caller. Since we use the
// NOTE: PetscCall(func(args)) convention, this shouldn't be an issue.
#define RDyFill(type, count, memory, value) \
  0;                                        \
  for (size_t i = 0; i < (count); ++i) {    \
    memory[i] = (type)value;                \
  }

#endif
