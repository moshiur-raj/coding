#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include "config.h"

void allocate_memory(double (**x)[OBSERVED_POINT_COUNT], double (**y)[OBSERVED_POINT_COUNT],
                     yfunctionptr_t **yfunction,
                     double **param_min, double **param_max, double **param_opt,
                     double (**x_out)[POINT_COUNT], double (**y_out)[POINT_COUNT]);
void free_memory(void *ptr, ...);

#endif
