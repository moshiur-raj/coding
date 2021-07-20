#include <stdio.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

#include <variable-types.h>

#define KILL_THREAD (-1)

struct PthreadFunctionArg
{
	int start, stop;
	const struct SimulationParameters *restrict param;
};

static inline void wall_collision(int i, const struct SimulationParameters *restrict param)
{
	if (param->position[i].x < param->radius)
	{
		param->acceleration[i].x += param->wca_factor;
	}
	else if (param->position[i].y < param->radius)
	{
		param->acceleration[i].y += param->wca_factor;
	}
	else if (param->boxsize.x - param->position[i].x < param->radius)
	{
		param->acceleration[i].x -= param->wca_factor;
	}
	else if (param->boxsize.y - param->position[i].y < param->radius)
	{
		param->acceleration[i].y -= param->wca_factor;
	}
	#ifdef VEC3D
		else if (param->position[i].z < param->radius)
		{
			param->acceleration[i].z += param->wca_factor;
		}
		else if (param->boxsize.z - param->position[i].z < param->radius)
		{
			param->acceleration[i].z -= param->wca_factor;
		}
	#endif
}

static inline double magnitude_squared(vector_t *vec)
{
	#ifdef VEC2D
		return vec->x*vec->x + vec->y*vec->y;
	#endif
	
	#ifdef VEC3D
		return vec->x*vec->x + vec->y*vec->y + vec->z*vec->z;
	#endif
}

static inline void molecular_collision(int i, const struct SimulationParameters *restrict param)
{
	vector_t dr;
	for (int j = i + 1; j < param->nmolecules; ++j)
	{
		dr.x = param->position[i].x - param->position[j].x;
		dr.y = param->position[i].y - param->position[j].y;
		#ifdef VEC3D
			dr.z = param->position[i].z - param->position[j].z;
		#endif

		if (magnitude_squared(&dr) < param->radius_squared)
		{
			param->acceleration[i].x += param->mca_factor * dr.x;
			param->acceleration[i].y += param->mca_factor * dr.y;
			#ifdef VEC3D
				param->acceleration[i].z += param->mca_factor * dr.z;
			#endif

			param->acceleration[j].x -= param->acceleration[i].x;
			param->acceleration[j].y -= param->acceleration[i].y;
			#ifdef VEC3D
				param->acceleration[j].z -= param->acceleration[i].z;
			#endif
		}
	}
}

static inline void handle_collisions_unit(int start, int stop,
                                          const struct SimulationParameters *restrict param)
{
	for(int i = start; i < stop; ++i)
	{
		wall_collision(i, param);
		molecular_collision(i, param);
	}
}

void *pthread_function(void *ptr)
{
	const struct PthreadFunctionArg *restrict arg = ptr;
	if (arg->start == KILL_THREAD) {return NULL;}

	handle_collisions_unit(arg->start, arg->stop, arg->param);

	return NULL;
}

static inline void initialize_args(struct PthreadFunctionArg *restrict arg,
                     const struct SimulationParameters *restrict param)
{
	int increment = param->nmolecules / param->nthread;
	for (int i = 0; i < param->nthread; ++i)
	{
		arg[i].param = param;
		arg[i].start = i*increment;
		arg[i].stop  = arg[i].start + increment;
	}
	arg[param->nthread - 1].stop = param->nmolecules;
}

void handle_collisions(const struct SimulationParameters *restrict param)
{
	memset(param->acceleration, 0, (size_t)param->nmolecules*sizeof(param->acceleration[0]));
	
	struct PthreadFunctionArg arg[param->nthread];
	initialize_args(arg, param);

	pthread_t thread[param->nthread];
	for (int i = 0; i < param->nthread; ++i)
	{
		pthread_create(&thread[i], NULL, pthread_function, &arg[i]);
	}
	for (int i = 0; i < param->nthread; ++i)
	{
		pthread_join(thread[i], NULL);
	}
}

void st_handle_collisions(const struct SimulationParameters *restrict param)
{
	memset(param->acceleration, 0, (size_t)param->nmolecules*sizeof(param->acceleration[0]));

	handle_collisions_unit(0, param->nmolecules, param);
}
