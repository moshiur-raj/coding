#include <pthread.h>

#include <stdio.h>
#include <variable-types.h>

#define KILL_THREAD (-1)

struct PthreadFunctionArg
{
	int start, stop;
	const struct SimulationParameters *restrict param;
};

static inline void phase_update_unit(int start, int stop,
                                     const struct SimulationParameters *restrict param)
{
	for (int i = start; i < stop; ++i)
	{
		param->position[i].x += param->velocity[i].x * param->dt;
		param->position[i].y += param->velocity[i].y * param->dt;
		#ifdef VEC3D
			param->position[i].z += param->velocity[i].z * param->dt;
		#endif

		param->velocity[i].x += param->acceleration[i].x * param->dt;
		param->velocity[i].y += param->acceleration[i].y * param->dt;
		#ifdef VEC3D
			param->velocity[i].z += param->acceleration[i].z * dt;
		#endif
	}
}

static void *pthread_function(void *ptr)
{
	const struct PthreadFunctionArg *arg = ptr;
	if (arg->start == KILL_THREAD) {return NULL;}

	phase_update_unit(arg->start, arg->stop, arg->param);

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

void st_update_phase(const struct SimulationParameters *restrict param)
{
	phase_update_unit(0, param->nmolecules, param);
}
