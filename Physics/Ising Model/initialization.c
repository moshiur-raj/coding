#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <sys/random.h>
#include <unistd.h>

#include "common.h"

#define N_THREADS_RNG 4

typedef enum
{
	spin,
	row_index,
	col_index,
	random_float
} array_type_t;

typedef struct
{
	void *arrptr;
	array_type_t arrtype;
} thrd_arg_t;

static void *start_thread(void *argptr)
{
	thrd_arg_t *arg = argptr;

	// exit if memory is not allocated
	if (arg->arrptr == NULL)
	{
		printf("\nMemory Was Not Allocated.\n");
		exit(EXIT_FAILURE);
	}

	const gsl_rng_type *T = gsl_rng_ranlxd2;
	gsl_rng *r = gsl_rng_alloc(T);

	int64_t seed;
	getrandom(&seed, sizeof(int64_t), 0);
	gsl_rng_set(r, seed);

	// for initializing s_0
	if (arg->arrtype == spin)
	{
		spin_t *array = arg->arrptr;
		double lim = 0.5 + _param.s_0 / 2;
		for (int32_t i = 0; i < _param.n_spins; ++i)
		{
			array[i] = 2 * (gsl_rng_uniform(r) < lim) - 1;
		}
	}
	// for initializing row
	else if (arg->arrtype == row_index)
	{
		index_t *array = arg->arrptr;
		for (int64_t i = 0; i < _param.n_iter_0; ++i)
		{
			array[i] = gsl_rng_uniform_int(r, _param.n_rows);
		}
	}
	// for initializing col
	else if (arg->arrtype == col_index)
	{
		index_t *array = arg->arrptr;
		for (int64_t i = 0; i < _param.n_iter_0; ++i)
		{
			array[i] = gsl_rng_uniform_int(r, _param.n_cols);
		}
	}
	// for initializing drand
	else if (arg->arrtype == random_float)
	{
		double *array = arg->arrptr;
		for (int64_t i = 0; i < _param.n_iter_0; ++i)
		{
			array[i] = gsl_rng_uniform(r);
		}
	}

	gsl_rng_free(r);

	return NULL;
}

void gen_random_data(LATTICE2D(lattice_0), index_t *row, index_t *col, double *drand)
{
	void *arrptr[N_THREADS_RNG] = {lattice_0, row, col, drand};
	array_type_t arrtype[N_THREADS_RNG] = {spin, row_index, col_index, random_float};
	thrd_arg_t arg[N_THREADS_RNG];

	pthread_t thread[N_THREADS_RNG];
	for (int i = 0; i < N_THREADS_RNG; ++i)
	{
		arg[i].arrptr = arrptr[i];
		arg[i].arrtype = arrtype[i];
		pthread_create(&thread[i], NULL, &start_thread, &arg[i]);
	}
	for (int i = 0; i < N_THREADS_RNG; ++i)
	{
		pthread_join(thread[i], NULL);
	}
}
