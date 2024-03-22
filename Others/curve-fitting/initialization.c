#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/fcntl.h>

#include "config.h"
#include "memory-management.h"

static void gen_shm_block_for_observed_data(double (*x)[OBSERVED_POINT_COUNT],
                                            double (*y)[OBSERVED_POINT_COUNT])
{
	int fd = shm_open(SHM_BLOCK2_NAME, O_CREAT | O_EXCL | O_RDWR, 0600);
	if (fd == -1) {perror("Error opening shared block2\n"); exit(EXIT_FAILURE);}
	size_t size = 2*OBSERVED_POINT_COUNT*sizeof(double);
	ftruncate(fd, size);
	double (*ptr)[OBSERVED_POINT_COUNT] = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
            fd, 0);
	memcpy(ptr, x, size/2);
	memcpy(++ptr, y, size/2);
}

void initialize(double (**x)[OBSERVED_POINT_COUNT], double (**y)[OBSERVED_POINT_COUNT],
                yfunctionptr_t **yfunction,
                double **param_min, double **param_max, double **param_opt,
                double (**x_out)[POINT_COUNT], double (**y_out)[POINT_COUNT])
{
	allocate_memory(x, y, yfunction, param_min, param_max, param_opt, x_out, y_out);

	double temp_x[X_DIM][OBSERVED_POINT_COUNT] = {
		XDATA
	};
	double temp_y[Y_DIM][OBSERVED_POINT_COUNT] = {
		YDATA
	};
	yfunctionptr_t temp_yfunction[Y_DIM] = {
		YFUNC
	};
	double temp_param_min[PARAM_COUNT] = {
		PARAM_MIN_DATA
	};
	double temp_param_max[PARAM_COUNT] = {
		PARAM_MAX_DATA
	};
	memcpy(*x, temp_x, sizeof(temp_x));
	memcpy(*y, temp_y, sizeof(temp_y));
	memcpy(*yfunction, temp_yfunction, sizeof(temp_yfunction));
	memcpy(*param_min, temp_param_min, sizeof(temp_param_min));
	memcpy(*param_max, temp_param_max, sizeof(temp_param_max));

	gen_shm_block_for_observed_data(*x, *y);
}

