#include <string.h>

#include "common.h"
#include "memory-management.h"
#include "initialization.h"
#include "metropolis.h"

parameters_t _param;

int start_simulation(parameters_t param, pointers_t ptr)
{
	// initialize the global variable _param.
	memcpy(&_param, &param, sizeof(parameters_t));

	// for storing initial and final states. final states are separate for each thread.
	LATTICE2D(lattice_0) = NULL;

	// for storing the index of randomly chosen spin.
	index_t *row = NULL, *col = NULL;
	// for storing random double precision float.
	double *drand = NULL;

	// allocate memory for the arrays shared among threads. This is needed to avoid regeneration
	// in each thread and to avoid extra function calls. It saves a lot of time at the expense of
	// memory allocation.
	mem_alloc((void **)&lattice_0, (void **)&row, (void **)&col, (void **)&drand);

	// initialize the arrays containing random variables.
	// lattice_0 stores the initial lattice.
	// row, col stores the indices of randomly chosen lattice point.
	// drand stores the random variables used to flip a spin randomly.
	gen_random_data(lattice_0, row, col, drand);

	// start metropolis algorithm and store the output.
	met_arg_t m_arg = {.lattice_0 = (void *)lattice_0, .row = row, .col = col, .drand = drand,
	                   .ptr = ptr};
	metropolis(m_arg);

	// deallocate memory
	mem_dealloc(4, &lattice_0, &row, &col, &drand);

	return 0;
}
