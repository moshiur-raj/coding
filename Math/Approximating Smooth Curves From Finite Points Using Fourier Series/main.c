#include <stdlib.h>
#include "IO.c"
#include "calculations.c"

#define num_arg 3

int main(int argc, char** argv)
{
	int num_input_points = count_numbers(argv[1]);
	int n_max = (int)strtol(argv[2], NULL, 10);
	int num_output_points = (int)strtol(argv[3], NULL, 10);
	if(argc != num_arg + 1 || num_input_points < 1 || n_max < 0 || num_output_points < 1){return -1;}

	complex double input_points[num_input_points], coefficients_of_exp[2*n_max + 1], coefficients_of_power[2*n_max + 1], output_points[num_output_points];
	double  start_time[num_input_points];
	take_input(input_points, num_input_points, argv[1]);
	gen_start_time(input_points, start_time, num_input_points);
	gen_coefficients_of_exp(coefficients_of_exp, input_points, start_time, num_input_points, n_max);
	gen_coefficients_of_power(coefficients_of_power, n_max);
	gen_output_points(output_points, num_output_points, coefficients_of_exp, coefficients_of_power, n_max);

	print_output_points(output_points, num_output_points);
	printf(";");
	print_min_max(output_points, num_output_points, input_points, num_input_points);

	return 0;
}
