#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <sys/random.h>

#include "common.h"
#include "metropolis.h"

#define DISABLED (-1)

typedef struct
{
	spin_t *lattice_0, *domain;
	index_t *row, *col;
	double *drand, *t_domain, *t, *s_f, *u_f, *s_avg;
	double u_total_0, h_norm;
	// range[2] is for distributing the tasks among threads
	int32_t m_0, range[2];
} thrd_arg_t;

// sum the elements of an int array
static int32_t isum(spin_t *array, int32_t size)
{
	int32_t sum = 0;
	for (int32_t i = 0; i < size; ++i)
	{
		sum += array[i];
	}

	return sum;
}

// calculate the exchange energy of the lattice. this obviously excludes the magnetic interaction
// energy.
static double exchange_energy(LATTICE2D(lattice))
{
	double u_total = 0;
	for (int i = 0; i < _param.n_rows; ++i)
	{
		for (int j = 0; j < _param.n_cols; ++j)
		{
			u_total += -lattice[i][j]*(lattice[(i+1)%_param.n_rows][j]
			           + lattice[i][(j+1)%_param.n_cols]);
		}
	}

	return u_total;
}

static void initialize_thrd_args(thrd_arg_t thrd_arg[], met_arg_t m_arg, int32_t h_norm_index,
                                 int32_t s_f_i_index, int32_t s_f_l_index)
{
	DOUBLE2D_ARRAY(s_f) = (void *)m_arg.ptr.s_f;
	DOUBLE2D_ARRAY(s_avg) = (void *)m_arg.ptr.s_avg;
	DOUBLE2D_ARRAY(u_f) = (void *)m_arg.ptr.u_f;
	DOUBLE2D_ARRAY(s_f_i) = (void *)m_arg.ptr.s_f_i;
	DOUBLE2D_ARRAY(s_f_l) = (void *)m_arg.ptr.s_f_l;

	// calculate the magnetization (sum of all spins).
	int32_t m_0 = isum(m_arg.lattice_0, _param.n_spins), n_tasks;
	// calculate the total energy. including magnetic interaction energy.
	double u_total_0 = - m_arg.ptr.h_norm[h_norm_index] * m_0
	                   + exchange_energy((void *)m_arg.lattice_0);

	thrd_arg_t prototype = {.lattice_0 = m_arg.lattice_0, .domain = m_arg.ptr.domain,
	                        .row = m_arg.row, .col = m_arg.col, .drand = m_arg.drand,
	                        .t_domain = m_arg.ptr.t_domain, .m_0 = m_0, .u_total_0 = u_total_0,
	                        .h_norm = m_arg.ptr.h_norm[h_norm_index]};

	if (s_f_i_index != DISABLED)
	{
		prototype.s_f = s_f_i[s_f_i_index];
		prototype.s_avg = NULL;
		prototype.u_f = NULL;
	}
	else if (s_f_l_index != DISABLED)
	{
		prototype.s_f = s_f_l[s_f_l_index];
		prototype.s_avg = NULL;
		prototype.u_f = NULL;
	}
	else
	{
		prototype.s_f = s_f[h_norm_index];
		prototype.s_avg = s_avg[h_norm_index];
		prototype.u_f = u_f[h_norm_index];
	}

	if (_param.flag.plt_thermodynamic_functions
	    || _param.flag.plt_thermodynamic_functions_for_different_h
	    || _param.flag.plt_conv_inf_iter || _param.flag.plt_conv_inf_lattice)
	{
		prototype.t = m_arg.ptr.t;
		n_tasks = _param.n_t;
	}
	else if (_param.flag.plt_domains)
	{
		prototype.t = m_arg.ptr.t_domain;
		n_tasks = _param.n_t_domain;
	}

	for (int i = 0; i < _param.n_threads; ++i)
	{
		memcpy(&thrd_arg[i], &prototype, sizeof(thrd_arg_t));
	}

	// divide the tasks
	int32_t step, remainder;
	step = n_tasks / _param.n_threads;
	remainder = n_tasks % _param.n_threads;

	thrd_arg[0].range[0] = 0;
	thrd_arg[0].range[1] = step + (remainder-- > 0);
	for (int i = 1; i < _param.n_threads; ++i)
	{
		thrd_arg[i].range[0] = thrd_arg[i-1].range[1];
		thrd_arg[i].range[1] = thrd_arg[i].range[0] + step + (remainder-- > 0);
	}
}

// calculate the energy difference caused to flipping the spin at (i, j) position.
static inline double energy_diff(LATTICE2D(lattice), index_t i, index_t j, index_t n_rows,
                                 index_t n_cols, double h_norm)
{
	return 2 * lattice[i][j] * (h_norm + lattice[i-1+(i==0)*n_rows][j]
	         + lattice[(i+1)%n_rows][j] + lattice[i][j-1+(j==0)*n_cols]
	         + lattice[i][(j+1)%n_cols]);
}

static void *start_thread(void *argptr)
{
	thrd_arg_t *arg = argptr;

	LATTICE2D(lattice) = malloc(_param.n_spins * sizeof(spin_t));

	LATTICE2D(lattice_0) = (void *)arg->lattice_0;
	LATTICE2D_ARRAY(domain) = (void *)arg->domain;

	index_t *row = arg->row, *col = arg->col;

	int count = 0;
	for (int i = 0; i < _param.n_t_domain; ++i)
	{
		if (arg->t[arg->range[0]] <= arg->t_domain[count])
		{
			break;
		}
		count++;
	}

	// local variables for faster dereferencing.
	double h_norm = arg->h_norm;
	index_t n_rows = _param.n_rows, n_cols = _param.n_cols;
	double *drand = arg->drand, *t = arg->t;
	for (int32_t i = arg->range[0]; i < arg->range[1]; ++i)
	{
		// inilialize the lattice state
		// the algorithm will converge faster for the next temp (t[i+1]) if we keep don't reseed
		// lattice with lattice_0. however this process cannot be multithreaded in the way er are
		// multithreading here. it will only work for each thread. also the later t's will converge
		// much faster leading to inconsistent results. we might be able to adjust this by changing
		// n_iter for each t[i].
		memcpy(lattice, lattice_0, _param.n_spins*sizeof(spin_t));

		// metropolis algorithm
		int64_t s_total = 0;
		int32_t ds, m = arg->m_0;
		double dE, u_total = arg->u_total_0;
		for (int32_t pass = 0; pass < _param.n_pass; ++pass)
		{
			for (int64_t j = 0, k = 0; j < _param.n_iter_0; ++j)
			{
				dE = energy_diff(lattice, row[j], col[j], n_rows, n_cols, h_norm);
				ds = - 2*lattice[row[j]][col[j]]
				     * (dE <= 0 || drand[k++] <  exp(-1/t[i] * dE));
				lattice[row[j]][col[j]] += ds;

				// add the changes in spin, energy and ensemble average
				m += ds;
				u_total += -h_norm * ds + dE * (ds != 0);
				s_total += m;
			}
		}

		arg->s_f[i] = (double) m / _param.n_spins;
		// block if NULL. needed when not wanting to write to arrays. such as for convergence test.
		if (arg->s_avg != NULL)
		{
			arg->s_avg[i] = (double) s_total / _param.n_spins / _param.n_iter_0 / _param.n_pass;
		}
		if (arg->u_f != NULL)
		{
			arg->u_f[i] = u_total / _param.n_spins;
		}
		// If t is in t_domain then store the lattice state
		if (_param.flag.plt_domains && arg->t[i] == arg->t_domain[count])
		{
			// domain are offset by one count. first one stores the initial random lattice.
			memcpy(domain[++count], lattice, _param.n_spins*sizeof(spin_t));
		}
	}

	free(lattice);

	return NULL;
}

static void start_thread_animation(met_arg_t arg)
{
	LATTICE2D(lattice) = malloc(_param.n_spins * sizeof(spin_t));

	LATTICE2D(lattice_0) = (void *)arg.lattice_0;
	LATTICE2D_ARRAY(lattice_anim) = (void *)arg.ptr.lattice_anim;

	memcpy(lattice, lattice_0, _param.n_spins*sizeof(spin_t));

	double h_norm = arg.ptr.h_norm[0], t_anim = _param.t_anim;
	index_t n_rows = _param.n_rows, n_cols = _param.n_cols;

	index_t *row = arg.row, *col = arg.col;
	double *drand = arg.drand;
	double dE;

	int32_t count = 0;
	// to divide the total iteration in equal number of steps.
	int64_t step = _param.n_pass * _param.n_iter_0 / (_param.n_frames - 1) + 1, iter = 0,
	               target = 0;

	// metropolis algorithm
	for (int32_t pass = 1; pass <= _param.n_pass; ++pass)
	{
		for (int64_t j = 0, k = 0; j < _param.n_iter_0; ++j)
		{
			dE = energy_diff(lattice, row[j], col[j], n_rows, n_cols, h_norm);
			lattice[row[j]][col[j]] += - 2 * lattice[row[j]][col[j]]
			                     * (dE <= 0 || drand[k++] <  exp(-1/t_anim * dE));

			// store lattice state at equal intervals.
			if (iter++ == target)
			{
				memcpy(lattice_anim[count++], lattice, _param.n_spins*sizeof(spin_t));
				target += step;
			}
		}
	}

	// if the last state was missed.
	if (count == _param.n_frames - 1)
	{
		memcpy(lattice_anim[count], lattice, _param.n_spins*sizeof(spin_t));
	}

	free(lattice);
}

static void start_and_join_threads(pthread_t thrd[], void *(*function)(void *),
                                   thrd_arg_t thrd_arg[])
{
	for (int i = 0; i < _param.n_threads; ++i)
	{
		pthread_create(&thrd[i], NULL, function, &thrd_arg[i]);
	}
	for (int i = 0; i < _param.n_threads; ++i)
	{
		pthread_join(thrd[i], NULL);
	}
}

void metropolis(met_arg_t arg)
{
	pthread_t thrd[_param.n_threads];
	thrd_arg_t thrd_arg[_param.n_threads];

	// Start the threads
	if (_param.flag.plt_thermodynamic_functions
	    || _param.flag.plt_thermodynamic_functions_for_different_h
	    || _param.flag.plt_domains || _param.flag.plt_conv_inf_iter
	    || _param.flag.plt_conv_inf_lattice)
	{
		initialize_thrd_args(thrd_arg, arg, 0, DISABLED, DISABLED);
		start_and_join_threads(thrd, &start_thread, thrd_arg);

		// the first domain is the generated random initial lattice
		if (_param.flag.plt_domains)
		{
			memcpy(arg.ptr.domain, arg.lattice_0, _param.n_spins * sizeof(spin_t));
		}
		printf("\nCompleted Generating Average Spin / Domains for h_norm[0]\n");
		// no need to store domains anymore.
		_param.flag.plt_domains = false;
	}

	// Plot thermodynamic functions for different magnetic field strengths
	if (_param.flag.plt_thermodynamic_functions_for_different_h)
	{
		for (int i = 1; i < _param.n_h_norm; ++i)
		{
			initialize_thrd_args(thrd_arg, arg, i, DISABLED, DISABLED);
			start_and_join_threads(thrd, &start_thread, thrd_arg);
		}
		printf("\nCompleted Generating Average Spin for h_norm\n");
	}

	// convergence for infinite iterations
	if (_param.flag.plt_conv_inf_iter)
	{
		// backup the variable
		int64_t n_iter_0 = _param.n_iter_0;

		for (int i = 0; i < _param.n_n_iter_mult; ++i)
		{
			_param.n_iter_0 = arg.ptr.n_iter_mult[i] * n_iter_0;
			initialize_thrd_args(thrd_arg, arg, 0, i, DISABLED);
			start_and_join_threads(thrd, &start_thread, thrd_arg);
		}
		printf("\nCompleted Convergence Test For Infinite Iteration\n");

		// restore the variable
		_param.n_iter_0 = n_iter_0;
	}

	// convergence for infinite lattice
	if (_param.flag.plt_conv_inf_lattice)
	{
		// backup the variables
		int32_t n_rows_0 = _param.n_rows, n_cols_0 = _param.n_cols;
		int64_t n_iter_0 = _param.n_iter_0;

		for (int i = 0; i < _param.n_lattice_size_mult; ++i)
		{
			_param.n_rows = arg.ptr.lattice_size_mult[i] * n_rows_0;
			_param.n_cols = arg.ptr.lattice_size_mult[i] * n_cols_0;
			_param.n_spins = _param.n_rows * _param.n_cols;
			_param.n_iter_0 = arg.ptr.lattice_size_mult[i] * arg.ptr.lattice_size_mult[i] * n_iter_0;
			for (int64_t i = 0; i < _param.n_iter_0; ++i)
			{
				arg.row[i] %= _param.n_rows;
				arg.col[i] %= _param.n_cols;
			}
			initialize_thrd_args(thrd_arg, arg, 0, DISABLED, i);
			start_and_join_threads(thrd, &start_thread, thrd_arg);
		}

		// restore the variables
		_param.n_rows = n_rows_0;
		_param.n_cols = n_cols_0;
		_param.n_spins = _param.n_rows * _param.n_cols;
		_param.n_iter_0 = n_iter_0;
		printf("\nCompleted Convergence Test For Infinite Lattice\n");
	}

	// start storing lattice states for generating animation frames
	if (_param.flag.anim_state_transitions)
	{
		start_thread_animation(arg);
		printf("\nCompleted Generating Frames For Animation\n");
	}
}
