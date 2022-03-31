#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/random.h>

#include <variable-types.h>

#define SPAWN_OFFSET_FACTOR (1.5)
#define UNIVERSAL_GAS_CONST (8.314)

struct BoxSize {double x, y, z;};

static inline void posrand_r(struct BoxSize *restrict reduced_boxsize,
                             double offset, vector_t *position,
                             struct drand48_data *restrict buffer)
{
		double drandom;
		drand48_r(buffer, &drandom);
		position->x = offset + drandom * reduced_boxsize->x;
		drand48_r(buffer, &drandom);
		position->y = offset + drandom * reduced_boxsize->y;
		#ifdef VEC3D
			drand48_r(&buffer, &drandom);
			position->z = offset + drandom * reduced_boxsize->z;
		#endif
}

static inline void gen_position(int i, struct BoxSize *restrict reduced_boxsize, double offset,
                                vector_t *restrict position, struct drand48_data *restrict buffer,
                                double radius_squared)
{
	int isduplicate;
	do
	{
		posrand_r(reduced_boxsize, offset, &position[i], buffer);
		vector_t dr;
		isduplicate = 0;
		for(int j = 0; j < i; ++j)
		{
			dr.x = position[i].x - position[j].x;
			dr.y = position[i].y - position[j].y;
			isduplicate += dr.x*dr.x + dr.y*dr.y < radius_squared;
		}
	} while (isduplicate);
}


static inline void velrand_r(struct drand48_data *restrict buffer, double magnitude,
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
	struct drand48_data buffer;
	long int seed;
	getentropy(&seed, sizeof(seed));
	srand48_r(seed, &buffer);

	double offset = param->radius * SPAWN_OFFSET_FACTOR;
	struct BoxSize reduced_boxsize = {
		.x = param->boxsize.x - 2*offset,
		.y = param->boxsize.y - 2*offset,
		.z = param->boxsize.z - 2*offset
	};

	for(int i = 0; i < param->nmolecules; ++i)
	{
		gen_position(i, &reduced_boxsize, offset, param->position, &buffer, param->radius_squared);
	}

	double rms_speed = sqrt(3*UNIVERSAL_GAS_CONST*param->temperature / param->relative_mass);
	for (int i = 0; i < param->nmolecules - 1; i += 2)
	{
		velrand_r(&buffer, rms_speed, &param->velocity[i]);
		param->velocity[i + 1].x = -param->velocity[i].x;
		param->velocity[i + 1].y = -param->velocity[i].y;
		#ifdef VEC3D
			param->velocity[i + 1].z = -param->velocity[i].z;
		#endif
	}
}
