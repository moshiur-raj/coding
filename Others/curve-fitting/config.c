#include <math.h>

#define PARAM_COUNT	2
#define X_DIM	1
#define Y_DIM	1
#define OBSERVED_POINT_COUNT	5
#define POINT_COUNT	1000

#define XDATA			{1, 2, 3, 4, 5}
#define YDATA			{2, 5, 10, 17, 26}
#define PARAM_MIN_DATA	 -1, -1
#define PARAM_MAX_DATA	 1, 1
#define PARAM_STEP_DATA	 1e-3, 1e-3

#define YFUNC	yfunction0

#ifndef CONFIG_H

double yfunction0(double *param, double *x)
{
	return param[0]*x[0]*x[0] + param[1];
}

#endif

double yfunction0(double *param, double *x);
typedef double (*yfunctionptr_t)(double *param, double *x);

#define SHM_BLOCK1_NAME	"/curve-fitting-data-simulated.bin"
#define SHM_BLOCK2_NAME	"/curve-fitting-data-observed.bin"
#define X_LABEL			"x"
#define Y_LABEL			"y"
#define LEGEND			""
