#include <stdlib.h>
#include <sys/mman.h>

#include "config.h"
#include "initialization.h"
#include "parameter-optimization.h"
#include "graph-rendering.h"
#include "memory-management.h"
#include "point-generation.h"

int main(int argc, char *argv[])
{
	double (*x)[OBSERVED_POINT_COUNT], (*y)[OBSERVED_POINT_COUNT],
	       *param_min, *param_max, *param_opt,
	       (*x_out)[POINT_COUNT], (*y_out)[POINT_COUNT];
	yfunctionptr_t *yfunction;

	initialize(&x, &y, &yfunction, &param_min, &param_max, &param_opt, &x_out, &y_out);
	optimize_parameters(x, y, yfunction, param_min, param_max, param_opt);
	gen_points(x, yfunction, param_opt, x_out, y_out);
	render_graph(x_out, y_out);

	free_memory(x, y, yfunction, param_min, param_max, param_opt, NULL);
	shm_unlink(SHM_BLOCK1_NAME);
	shm_unlink(SHM_BLOCK2_NAME);
	return 0;
}
