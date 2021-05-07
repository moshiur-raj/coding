#include <stdlib.h>
#include <sys/random.h>
#include "pcg_basic.h"

extern long int _num_rays;
extern double *restrict _transparency, *restrict _trap_capacity;

struct thread_data_s
{
	int layer_num,
		transparency_index_min,
		transparency_index_max;
};

// typedef-ing re-entrant random number generator data types
typedef pcg32_random_t rand_buffer_t;
typedef uint32_t rand_t;

// initiate random number generator state buffer
static inline void init_rand_buff(rand_buffer_t *rand_buff_ptr)
{
	rand_t randvar[4];
	getentropy(randvar, 4*sizeof(rand_t));
	pcg32_srandom_r(rand_buff_ptr, randvar[0], randvar[1]);
	pcg32_srandom_r(rand_buff_ptr + 1, randvar[2], randvar[3]);

	return;
}

// return -1 or +1 based on transparency.
// -1 indicated the ray going to the lower layer.
// +1 indicates the ray going to the upper layer.
static inline int update_layer(rand_buffer_t *rand_buffer_ptr, rand_t threshold)
{
	return (pcg32_random_r(rand_buffer_ptr) < threshold) ? 1 : (2*(pcg32_random_r(rand_buffer_ptr + 1) & 1) - 1);
}

// collect how many rays hit the ground.
static inline long int collect_rays(int last_layer, double transparency)
{
	long int trapped_rays = 0;
	rand_buffer_t rand_buff[2];
	init_rand_buff(rand_buff);
	rand_t threshold = transparency*UINT32_MAX;
	for(long int i = 0; i < _num_rays; ++i)
	{
		enum {ground_layer = 0, first_layer = 1};
		int layer_num = first_layer;
		while((layer_num != ground_layer) && (layer_num != last_layer + 1))
		{
			layer_num += update_layer(rand_buff, threshold);
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
		_trap_capacity[i] = (double)((long double)collect_rays(thread_data_ptr->layer_num, _transparency[i])/_num_rays);
	}

	return NULL;
}

static inline double predict_trap_capacity(double fn, double T)
{
	return (fn + (1 - T)*(1 - 2*fn))/(1 - (1 - T)*fn); // recursive formula
}

void *prediction(void *thread_data_vptr)
{
	struct thread_data_s *thread_data_ptr = thread_data_vptr;
	for(int i = thread_data_ptr->transparency_index_min; i < thread_data_ptr->transparency_index_max; ++i)
	{
		_trap_capacity[i] = predict_trap_capacity(_trap_capacity[i], (1 + _transparency[i])/2);
	}

	return NULL;
}
