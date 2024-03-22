#ifndef PARAMETER_OPTIMIZATION_H
#define PARAMETER_OPTIMIZATION_H

#include "config.h"

void optimize_parameters(double (*x)[OBSERVED_POINT_COUNT], double (*y)[OBSERVED_POINT_COUNT],
                            yfunctionptr_t *yfunction,
                            double *param_min, double *param_max, double *param_opt);

#endif
