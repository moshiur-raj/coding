#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "config.h"

static void gen_param_sample(double *param_min, double *param_max,
                             double **param_sample, int *sample_count)
{
	double param_step[PARAM_COUNT] = {
		PARAM_STEP_DATA
	};

	for (int i = 0; i < PARAM_COUNT; ++i)
	{
		sample_count[i] = floor((param_max[i] - param_min[i])/param_step[i]) + 1;
		param_sample[i] = malloc(sample_count[i]*sizeof(double));
		param_sample[i][0] = param_min[i];
		for (int j = 1; j < sample_count[i] - 1; ++j)
		{
			param_sample[i][j] = param_min[i] + j*param_step[i];
		}
		param_sample[i][sample_count[i] - 1] = param_max[i];
	}
}

static inline void set_param(int *sample_index, double **param_sample, double *param)
{
	for (int i = 0; i < PARAM_COUNT; ++i)
	{
		param[i] = param_sample[i][sample_index[i]];
	}
}

static inline void get_nth_xvector(int n, double (*input)[OBSERVED_POINT_COUNT], double *output)
{
	for (int i = 0; i < X_DIM; ++i)
	{
		output[i] = input[i][n];
	}
}

static inline double get_error(double (*x)[OBSERVED_POINT_COUNT], double (*y)[OBSERVED_POINT_COUNT],
                               yfunctionptr_t *yfunction, double *param)
{
	double temp_x[X_DIM];
	double error = 0, temp;
	for (int i = 0; i < Y_DIM; ++i)
	{
		for (int j = 0; j < OBSERVED_POINT_COUNT; ++j)
		{
			get_nth_xvector(j, x, temp_x);
			temp = y[i][j] - yfunction[i](param, temp_x);
			error += 0.9*fabs(temp/y[i][j]) + 0.1*fabs(temp);
		}
	}

	return error;
}

static inline void increment_index(int *sample_index, int *sample_count)
{
	++sample_index[PARAM_COUNT - 1];
	for (int i = PARAM_COUNT - 1; i > 0; --i)
	{
		if (sample_index[i] < sample_count[i])
		{
			break;
		}
		else
		{
			sample_index[i] = 0;
			++sample_index[i - 1];
		}
	}
}

void optimize_parameters(double (*x)[OBSERVED_POINT_COUNT], double (*y)[OBSERVED_POINT_COUNT],
                            yfunctionptr_t *yfunction,
                            double *param_min, double *param_max, double *param_opt)
{
	double *param_sample[PARAM_COUNT], param[PARAM_COUNT];
	int sample_count[PARAM_COUNT];
	gen_param_sample(param_min, param_max, param_sample, sample_count);

	int sample_index[PARAM_COUNT];
	memset(sample_index, 0, PARAM_COUNT*sizeof(int));
	set_param(sample_index, param_sample, param);

	double error_min = get_error(x, y, yfunction, param), error;
	memcpy(param_opt, param, PARAM_COUNT*sizeof(double));
	++sample_index[PARAM_COUNT - 1];

	while (sample_index[0] < sample_count[0])
	{
		set_param(sample_index, param_sample, param);
		error = get_error(x, y, yfunction, param);
		if (error < error_min)
		{
			error_min = error;
			memcpy(param_opt, param, PARAM_COUNT*sizeof(double));
		}
		increment_index(sample_index, sample_count);
	}

	for (int i = 0; i < PARAM_COUNT; ++i)
	{
		printf("Optimized value for %ith parameter = %lf\n", i, param_opt[i]);
		free(param_sample[i]);
	}
}
