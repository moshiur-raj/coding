#include <stdlib.h>

#include "defs.h"

// collect how many rays hit the ground.
static inline long int collect_rays(int last_layer, double transparency, int seed)
{
	struct drand48_data randbuffer;
	srand48_r(seed, &randbuffer);
	double rand;

	enum {ground_layer = 0, first_layer = 1};
	int layer_num;
	long int trapped_rays = 0;

	for(long int i = 0; i < _num_rays; ++i)
	{
		layer_num = first_layer;
		int direction = 1;
		while((layer_num != ground_layer) && (layer_num != last_layer + 1))
		{
			drand48_r(&randbuffer, &rand);
			direction *= (2 * ( rand < transparency ) - 1);
			layer_num +=  direction;
		}
		trapped_rays += (layer_num == ground_layer);
	}

	return trapped_rays;
}

void *simulation(void *thread_data_vptr)
{
	struct thread_data_s *thread_data_ptr = thread_data_vptr;
	for(int i = thread_data_ptr->transparency_index_min; i < thread_data_ptr->transparency_index_max; ++i)
	{
		_trap_capacity[i] = (double)collect_rays(thread_data_ptr->layer_num, _transparency[i],
		                     rand_r(thread_data_ptr->seedp)) / _num_rays;
	}

	return NULL;
}

static inline double predict_trap_capacity(double fn, double t)
{
	return 1 - (1 - fn) * t / (1 - (1 - t) * fn);
}

void *prediction(void *thread_data_vptr)
{
	struct thread_data_s *thread_data_ptr = thread_data_vptr;
	for(int i = thread_data_ptr->transparency_index_min; i < thread_data_ptr->transparency_index_max; ++i)
	{
		_trap_capacity[i] = predict_trap_capacity(_trap_capacity[i], _transparency[i]);
	}

	return NULL;
}
