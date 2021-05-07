#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

int count_numbers(const char* pointer)
{
	int n = 0;
	while(*pointer != '\0')
	{
		if(*pointer == 'j')
		{
			++n;
		}
		++pointer;
	}
	return n;
}

void take_input(complex double* input_points, int n, char* str)
{
	for(int i = 0; i < n; ++i)
	{
		input_points[i] = CMPLX(strtod(str, &str), strtod(str, &str));
		str = str + 2;
	}
}

void print_cmplx(complex double z)
{
	if(cimag(z) >= 0)
	{
		printf("%lf+%lfj\n", creal(z), cimag(z));
	}
	else
	{
		printf("%lf%lfj\n", creal(z), cimag(z));
	}
}

void print_output_points(const complex double* output_points, int num_output_points)
{
	for(int i = 0; i < num_output_points; ++i)
	{
		print_cmplx(output_points[i]);
	}
}

void print_min_max(const complex double* output_points, int num_output_points, const complex double* input_points, int num_input_points)
{
	double x_min = creal(output_points[0]), x_max = x_min, y_min = cimag(output_points[0]), y_max = y_min;
	for(int i = 1; i < num_output_points; ++i)
	{
		if(creal(output_points[i]) < x_min)
		{
			x_min = creal(output_points[i]);
		}
		if(creal(output_points[i]) > x_max)
		{
			x_max = creal(output_points[i]);
		}
		if(cimag(output_points[i]) < y_min)
		{
			y_min = cimag(output_points[i]);
		}
		if(cimag(output_points[i]) > y_max)
		{
			y_max = cimag(output_points[i]);
		}
	}
	for(int i = 0; i < num_input_points; ++i)
	{
		if(creal(input_points[i]) < x_min)
		{
			x_min = creal(input_points[i]);
		}
		if(creal(input_points[i]) > x_max)
		{
			x_max = creal(input_points[i]);
		}
		if(cimag(input_points[i]) < y_min)
		{
			y_min = cimag(input_points[i]);
		}
		if(cimag(input_points[i]) > y_max)
		{
			y_max = cimag(input_points[i]);
		}
	}
	printf("%lf,%lf,%lf,%lf", x_min, x_max, y_min, y_max);
}
