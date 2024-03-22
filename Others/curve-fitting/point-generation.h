#ifndef POINT_GENERATION_H
#define POINT_GENERATION_H

#include "config.h"

void gen_points(double (*x)[OBSERVED_POINT_COUNT], yfunctionptr_t *yfunction,
                double *param_opt, double (*x_out)[POINT_COUNT], double (*y_out)[POINT_COUNT]);

#endif
