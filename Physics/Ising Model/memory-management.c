#include <stdlib.h>
#include <stdarg.h>

#include "common.h"

void mem_alloc(void **lattice_0, void **row, void **col, void **drand)
{
	*lattice_0 = malloc(_param.n_spins * sizeof(spin_t));

	*row   = malloc(_param.n_iter_0 * sizeof(index_t));
	*col   = malloc(_param.n_iter_0 * sizeof(index_t));
	*drand = malloc(_param.n_iter_0 * sizeof(double));
}

void mem_dealloc(int nargs, ...)
{
	va_list arg;
	va_start(arg, nargs);

	void **ptr = NULL;
	for (int i = 0; i < nargs; ++i)
	{
		ptr = va_arg(arg, void **);
		free(*ptr);
		*ptr = NULL;
	}

	va_end(arg);
}
