#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "pcg_basic.h"

// number of threads to use for calculations.
// set this to the no of threads in your cpu for good performance
#define NUM_THREAD	4

// structure for passing necessary data to each thread. each thread will generate
// data points for a certain layer number and a range of transparency.
struct thread_data_s
{
	int layer_num,
		transparency_index_min,
		transparency_index_max;
};

static int num_layers_min = 0, num_layers_max = 0, step_num_layers = 0, num_output_points = 0;

long int _num_rays = 0;	// number of rays to simulate for each unique (layer_num, transparency).
double *restrict _transparency = NULL, *restrict _trap_capacity = NULL;	// for holding data until all threads finish.

static void init_global_vars(char **argv)
{
	num_layers_min = strtol(argv[1], NULL, 10);
	num_layers_max = strtol(argv[2], NULL, 10);
	step_num_layers = strtol(argv[3], NULL, 10);
	_num_rays = strtol(argv[6], NULL, 10);
	num_output_points = strtol(argv[7], NULL, 10);
	double transparency_min = strtod(argv[4], NULL),
		   transparency_max = strtod(argv[5], NULL),
		   transparency_step = (transparency_max - transparency_min)/(num_output_points - 1);

	_transparency = malloc(num_output_points*sizeof(double));
	// generating the possible values of transparency
	for(int i = 0; i < num_output_points - 1; ++i)
	{
		// multiplication chosen instead of incrementing by transparency_step
		// due to rounding errors caused in addition.
		_transparency[i] = transparency_min + i*transparency_step;
	}
	// the last element initialized explicitly so it does not exceed 1.0 due to rounding errors.
	_transparency[num_output_points - 1] = transparency_max;
	_trap_capacity = malloc(num_output_points*sizeof(double));
}

// dividing work between threads (which thread will work on which range of transparency)
static inline int gen_transparency_index_min(int thread_no) { return (thread_no*num_output_points)/NUM_THREAD; }
static inline int gen_transparency_index_max(int thread_no) { return (++thread_no*num_output_points)/NUM_THREAD; }

static void gen_thread_data(char **argv,  struct thread_data_s *thread_data)
{
	for(int i = 0; i < NUM_THREAD; ++i)
	{
		thread_data[i].transparency_index_min = gen_transparency_index_min(i);
		thread_data[i].transparency_index_max = gen_transparency_index_max(i);
	}

	return;
}

// functions for calculating "trap capacity"
extern void *simulation(void *thread_data_vptr);
extern void *prediction(void *thread_data_vptr);

// printing data stored in either transparency or trap_capacity to stdout
static inline void print_data(double *restrict doubleptr)
{
	for(int i = 0; i < num_output_points; ++i)
	{
		printf("%.6le\t", doubleptr[i]);
	}
	putchar('\n');
}

// generating (transparency, trap_capacity) [multithreaded]
static void gen_data(void *(*thread_func)(void *), struct thread_data_s *thread_data)
{
	pthread_t thread[NUM_THREAD];
	for(int layer_num = num_layers_min; layer_num <= num_layers_max; layer_num += step_num_layers)
	{
		for(int i = 0; i < NUM_THREAD; ++i)
		{
			thread_data[i].layer_num = layer_num;
			pthread_create(&thread[i], NULL, thread_func, thread_data + i);
		}
		for(int i = 0; i < NUM_THREAD; ++i)
		{
			pthread_join(thread[i], NULL);
		}
		print_data(_trap_capacity);
	}

	return;
}

int main(int argc, char **argv)
{
	init_global_vars(argv);

	struct thread_data_s thread_data[NUM_THREAD];
	gen_thread_data(argv,  thread_data);

	print_data(_transparency);
	gen_data(simulation, thread_data);
	// trap_capacity for layer_num = 0. needed for recursion.
	memset(_trap_capacity, 0, num_output_points*sizeof(double));
	gen_data(prediction, thread_data);

	return 0;
}
