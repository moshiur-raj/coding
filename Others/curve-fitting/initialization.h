#ifndef VARIABLE_INITIALIZATION_H
#define VARIABLE_INITIALIZATION_H

#include "config.h"

void initialize(double (**x)[OBSERVED_POINT_COUNT], double (**y)[OBSERVED_POINT_COUNT],
                yfunctionptr_t **yfunction,
                double **param_min, double **param_max, double **param_opt,
                double (**x_out)[POINT_COUNT], double (**y_out)[POINT_COUNT]);

#endif
