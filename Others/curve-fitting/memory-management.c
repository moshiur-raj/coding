#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/fcntl.h>
#include <sys/mman.h>

#include "config.h"

void allocate_memory(double (**x)[OBSERVED_POINT_COUNT], double (**y)[OBSERVED_POINT_COUNT],
                     yfunctionptr_t **yfunction,
                     double **param_min, double **param_max, double **param_opt,
                     double (**x_out)[POINT_COUNT], double (**y_out)[POINT_COUNT])
{
	*x = malloc(X_DIM*OBSERVED_POINT_COUNT*sizeof(double));
	*y = malloc(Y_DIM*OBSERVED_POINT_COUNT*sizeof(double));
	*yfunction = malloc(Y_DIM*sizeof(yfunctionptr_t));
	*param_min = malloc(PARAM_COUNT*sizeof(double));
	*param_max = malloc(PARAM_COUNT*sizeof(double));
	*param_opt = malloc(PARAM_COUNT*sizeof(double));

	int fd = shm_open(SHM_BLOCK1_NAME, O_CREAT | O_EXCL | O_RDWR, 0600);
	if (fd == -1) {perror("Error opening shared block1\n"); exit(EXIT_FAILURE);}
	size_t size = POINT_COUNT*sizeof(double)*(X_DIM + Y_DIM);
	ftruncate(fd, size);
	*x_out = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	*y_out = *x_out + X_DIM;
}

void free_memory(void *ptr, ...)
{
	va_list ap;
	va_start(ap, ptr);
	while(ptr != NULL)
	{
		ptr = va_arg(ap, void *);
		free(ptr);
	}
	va_end(ap);
}
