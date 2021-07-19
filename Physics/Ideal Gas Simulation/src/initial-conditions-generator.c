#include <stdlib.h>
#include <math.h>
#include <sys/random.h>

#include <variable-types.h>

#define SPAWN_OFFSET_FACTOR (1.5)
#define UNIVERSAL_GAS_CONST (8.314)

static inline void vrand48_r(struct drand48_data *restrict buffer, double magnitude,
                              vector_t *restrict vector)
{
	drand48_r(buffer, &vector->x);
	vector->x *= magnitude;
	#ifdef VEC2D
		vector->y = sqrt(magnitude*magnitude - vector->x*vector->x);
	#endif
	#ifdef VEC3D
		const double yz_component_squared = magnitude*magnitude - vector->x*vector->x;
		drand48_r(buffer, &unit_vector->y);
		unit_vector->y *= sqrt(yz_component_squared);
		unit_vector->z  = sqrt(yz_component_squared - vector->y*vector->y);
	#endif
}

void gen_initial_conditions(const struct SimulationParameters *restrict param)
{
	double drandom;
	struct drand48_data buffer;
	long int seed;
	getentropy(&seed, sizeof(seed));
	srand48_r(seed, &buffer);

	for(int i = 0; i < param->num_molecules; ++i)
	{
		drand48_r(&buffer, &drandom);
		param->position[i].x = drandom * param->boxsize.x;
		param->position[i].y = drandom * param->boxsize.y;
		#ifdef VEC3D
			param->position[i].z = drandom * param->boxsize.z;
		#endif
	}

	double rms_speed = sqrt(3*UNIVERSAL_GAS_CONST*param->temperature / param->relative_mass);
	for (int i = 0; i < param->num_molecules - 1; i += 2)
	{
		vrand48_r(&buffer, rms_speed, &param->velocity[i]);
		param->velocity[i + 1].x = -param->velocity[i].x;
		param->velocity[i + 1].y = -param->velocity[i].y;
		#ifdef VEC3D
			param->velocity[i + 1].z = -param->velocity[i].z;
		#endif
	}
}
