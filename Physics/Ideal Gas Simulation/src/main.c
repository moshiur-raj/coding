#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <variable-types.h>
#include "argument-parser.h"
#include "initial-conditions-generator.h"
#include "collision-handler.h"
#include "phase-updates.h"
#include "data-dumper.h"

static void allocate_memory(struct SimulationParameters *param)
{
	param->position     = malloc((size_t)param->num_molecules*sizeof(vector_t));
	param->velocity     = malloc((size_t)param->num_molecules*sizeof(vector_t));
	param->acceleration = malloc((size_t)param->num_molecules*sizeof(vector_t));

	if (param->position == NULL || param->velocity == NULL || param ->acceleration == NULL)
	{
		fprintf(stderr, "ERROR: Could Not Allocate Memory\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
}

void delete_previous_data_dumps(void)
{
	if (access(X_DATA, F_OK) || access(Y_DATA, F_OK) || access(Z_DATA, F_OK))
	{
		fprintf(stderr, "WARNING: data already exists. deleting the old data in 5 seconds...\n");
		// sleep(5);
		remove(X_DATA);
		remove(Y_DATA);
		#ifdef VEC3D
			remove(Z_DATA);
		#endif
		fprintf(stderr, "WARNING: old data deleted.\n");
		fflush(stderr);
	}
}

void start_simulation(const struct SimulationParameters *restrict param)
{
	// static int iteration = 0;
	delete_previous_data_dumps();

	double t = 0, time_to_dump_data = 0;
	while (t < param->time_limit)
	{
		handle_collisions(param);
		update_phase(param);

		t += param->dt;
		if (t > time_to_dump_data)
		{
			time_to_dump_data += param->frametime;
			dump_data(param->position, param->num_molecules);
		}
		// printf(">>> %ith iteration completed.\n", iteration++);
		// fflush(stdout);
	}
}
static void free_memory(const struct SimulationParameters *param)
{
	free(param->position);
	free(param->velocity);
	free(param->acceleration);
}

int main(int argc, char *argv[])
{
	const struct SimulationParameters param;
	parse_args(argc, argv, (struct SimulationParameters *)&param);
	allocate_memory((struct SimulationParameters *)&param);

	gen_initial_conditions(&param);
	start_simulation(&param);

	free_memory(&param);
	return 0;
}
