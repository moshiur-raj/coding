#ifndef DEFS_H
#define DEFS_H

#include <stdlib.h>

extern long int _num_rays;
extern double *restrict _transparency, *restrict _trap_capacity;

// number of threads to use for calculations.
extern int _num_threads;

// structure for passing necessary data to each thread. each thread will generate
// data points for a certain layer number and a range of transparency.
struct thread_data_s
{
	int layer_num,
		transparency_index_min,
		transparency_index_max;
	uint *seedp;
};

#endif
