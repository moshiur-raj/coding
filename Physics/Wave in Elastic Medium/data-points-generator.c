#include "types.h"

static inline
void update_for_boundary(struct Parameters *param)
{
	Float_t (*restrict position)[param->ndim] = param->position;
	Float_t (*restrict velocity)[param->ndim] = param->velocity;
	Int_t *restrict index = param->index.boundary;
	Int_t (*restrict neighbor)[param->ndim_ng][2] = param->neighbor;
	Int_t nparticles = param->particle_count.boundary;
	const Float_t dt = param->dt;
	const Float_t dt_scaled = param->dt_scaled;
	// boundary particles are fixed.
}

static inline
void update_for_inside(struct Parameters *param)
{
	Float_t (*restrict position)[param->ndim] = param->position;
	Float_t (*restrict velocity)[param->ndim] = param->velocity;
	Int_t *restrict index = param->index.inside;
	Int_t (*restrict neighbor)[param->ndim_ng][2] = param->neighbor;

	for (Int_t dim = 0; dim < param->ndim; ++dim)
	{
		for (Int_t i = 0; i < param->particle_count.inside; ++i)
		{
			position[index[i]][dim] += velocity[index[i]][dim] * param->dt;
		}

		for (Int_t dim_ng = 0; dim_ng < param->ndim_ng; ++dim_ng)
		{
			for (Int_t i = 0; i < param->particle_count.inside; ++i)
			{
				velocity[index[i]][dim] += (position[neighbor[index[i]][dim_ng][0]][dim]
										  + position[neighbor[index[i]][dim_ng][1]][dim]
										  - 2 * position[index[i]][dim]) * param->dt_scaled;
			}
		}
	}
}

static inline
void dump_data(struct Parameters *param)
{
	Float_t (*restrict shm)[param->ndim][param->particle_count.plot] = param->shm;
	Float_t (*restrict position)[param->ndim] = param->position;

	static Int_t offset = 0;
	for (Int_t dim =  0; dim < param->ndim; ++dim)
	{
		for (Int_t i = 0, j = 0; i < param->particle_count.total; i += param->di_data_dump)
		{
			shm[offset][dim][j++] = position[i][dim];
		}
	}
	++offset;
}

void generate_data_points(struct Parameters param)
{
	Float_t t = 0, dt_data_dump = 1/param.fps, threshold;
	threshold = dt_data_dump;
	dump_data(&param);
	while (t < param.time_limit)
	{
		update_for_boundary(&param);
		update_for_inside(&param);
		t += param.dt;

		if (t > threshold)
		{
			dump_data(&param);
			threshold += dt_data_dump;
		}
	}
}
