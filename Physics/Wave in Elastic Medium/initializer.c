#include "types.h"
#include "config.h"

#include <stdlib.h>
#include <stdio.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

static
void allocate_memory(struct Parameters *param)
{
	param->position = malloc(param->particle_count.total * param->ndim * sizeof(Float_t));
	param->velocity = malloc(param->particle_count.total * param->ndim * sizeof(Float_t));
	param->min      = malloc(param->ndim * sizeof(Float_t));
	param->max      = malloc(param->ndim * sizeof(Float_t));

	param->neighbor = malloc(param->particle_count.total * param->ndim_ng * 2 * sizeof(Int_t));
	param->index.boundary = malloc(param->particle_count.boundary * sizeof(Int_t));
	param->index.inside = malloc(param->particle_count.inside * sizeof(Int_t));

	int shm_fd = shm_open(SHM_BLOCK_NAME, O_CREAT | O_EXCL | O_RDWR, 0600);
	if (shm_fd == -1) {perror("Error opening shared block\n"); exit(EXIT_FAILURE);}
	size_t size = param->particle_count.plot * param->ndim * param->nframes * sizeof(Float_t);
	ftruncate(shm_fd, size);
	param->shm = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
}

void initialize(struct Parameters *param)
{
	allocate_memory(param);

	Float_t (*restrict position)[param->ndim] = param->position;
	Int_t (*restrict neighbor)[param->ndim_ng][2] = param->neighbor;
	Float_t (*restrict velocity)[param->ndim] = param->velocity;

	// This is only for a 1D string in 2D
	for (Int_t i = 0; i < param->particle_count.total; ++i)
	{
		position[i][0] = (Float_t) i / param->particle_count.total;
		position[i][1] = 10 * position[i][0] * (position[i][0] - 1) * exp(-30 * position[i][0]);
	}
	param->min[0] = 0;
	param->max[0] = 1;
	param->min[1] = -0.15;
	param->max[1] = 0.15;
	memset(velocity, 0, param->particle_count.total * param->ndim * sizeof(Float_t));

	param->index.boundary[0] = 0;
	param->index.boundary[1] = param->particle_count.total - 1;
	for (Int_t i = 0; i < param->particle_count.inside; ++i)
	{
		param->index.inside[i] = i + 1;
	}

	for (Int_t i = 0; i < param->particle_count.total; ++i)
	{
		neighbor[i][0][0] = i - 1;
		neighbor[i][0][1] = i + 1;
	}
}

void cleanup(struct Parameters param)
{
	free(param.position);
	free(param.velocity);
	free(param.min);
	free(param.max);
	free(param.neighbor);
	free(param.index.boundary);
	free(param.index.inside);
	shm_unlink(SHM_BLOCK_NAME);
}
