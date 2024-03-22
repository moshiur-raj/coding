#include <stdlib.h>
#include "config.h"

struct ArrayBounds
{
	double min, max;
};

static inline struct ArrayBounds get_interval(double *x, int count)
{
	struct ArrayBounds interval;
	interval.min = interval.max = x[0];
	for (int i = 1; i < count; ++i)
	{
		if (x[i] < interval.min)
		{
			interval.min = x[i];
		}
		else if (x[i] > interval.max)
		{
			interval.max = x[i];
		}
	}

	return interval;
}

static inline void linspace(struct ArrayBounds interval, int count, double *x)
{
	double step = (interval.max - interval.min)/(POINT_COUNT - 1);
	x[0] = interval.min;
	for (int i = 1; i < POINT_COUNT - 1; ++i)
	{
		x[i] = interval.min + i*step;
	}
	x[POINT_COUNT - 1] = interval.max;
}

static inline void get_nth_xvector(int n, double (*input)[POINT_COUNT], double *output)
{
	for (int i = 0; i < X_DIM; ++i)
	{
		output[i] = input[i][n];
	}
}

void gen_points(double (*x)[OBSERVED_POINT_COUNT], yfunctionptr_t *yfunction,
                double *param_opt, double (*x_out)[POINT_COUNT], double (*y_out)[POINT_COUNT])
{
	for (int i = 0; i < X_DIM; ++i)
	{
		linspace(get_interval(x[i], OBSERVED_POINT_COUNT), POINT_COUNT, x_out[i]);
	}

	double temp_x[X_DIM];
	for (int i = 0; i < Y_DIM; ++i)
	{
		for (int j = 0; j < POINT_COUNT; ++j)
		{
			get_nth_xvector(j, x_out, temp_x);
			y_out[i][j] = yfunction[i](param_opt, temp_x);
		}
	}
}
