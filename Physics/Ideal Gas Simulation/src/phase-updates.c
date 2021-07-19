#include <pthread.h>

#include <stdio.h>
#include <variable-types.h>

#define KILL_THREAD (-1)

struct PthreadFunctionArg
{
	int start, stop;
	double dt;
	vector_t *restrict position, *restrict velocity, *restrict acceleration;
};

static void *pthread_function(void *ptr)
{
	const struct PthreadFunctionArg *arg = ptr;
	if (arg->start == KILL_THREAD) {return NULL;}

	for (int i = arg->start; i < arg->stop; ++i)
	{
		arg->position[i].x += arg->velocity[i].x * arg->dt;
		arg->position[i].y += arg->velocity[i].y * arg->dt;
		#ifdef VEC3D
			arg->position[i].z += arg->velocity[i].z * arg->dt;
		#endif

		arg->velocity[i].x += arg->acceleration[i].x * arg->dt;
		arg->velocity[i].y += arg->acceleration[i].y * arg->dt;
		#ifdef VEC3D
			arg->velocity[i].z += arg->acceleration[i].z * dt;
		#endif
	}

	return NULL;
}

static inline void initialize_args(struct PthreadFunctionArg *restrict arg,
                                   const struct SimulationParameters *restrict param)
{
	struct PthreadFunctionArg prototype = {.dt = param->dt,
	                                           .position= param->position,
	                                           .velocity=param->velocity,
	                                           .acceleration = param->acceleration};
	if (param->num_molecules >= param->nthread)
	{
		int increment = param->num_molecules / param->nthread;
		for (int i = 0; i < param->nthread; ++i)
		{
			arg[i] = prototype;
			arg[i].start = i*increment;
			arg[i].stop  = arg[i].start + increment;
		}
		arg[param->nthread - 1].stop = param->num_molecules;
	}
	else
	{
		for(int i = 0; i < param->num_molecules; ++i)
		{
			arg[i] = prototype;
			arg[i].start = i;
			arg[i].stop  = i + 1;
		}
		for (int i = param->num_molecules; i < param->nthread; ++i)
		{
			arg[i].start = KILL_THREAD;
		}
	}
}

void update_phase(const struct SimulationParameters *restrict param)
{
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
