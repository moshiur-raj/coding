#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H 1

extern void mem_alloc(void **lattice_0, void **row, void **col, void **drand);

extern void mem_dealloc(int nargs, ...);

#endif
