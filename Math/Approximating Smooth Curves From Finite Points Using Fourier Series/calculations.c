#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>

//#define M_PI		3.14159
#define period		(2*M_PI)
#define num_threads	4

struct integral_thread_data
{
	int n_max;
	int min_i;
	int max_i;
	int num_input_points;
	const complex double* input_points;
	const double* start_time;
	complex double* coefficients_of_exp;
};

struct fourier_thread_data
{
	int min_i;
	int max_i;
	int n_max;
	double dt;
	const complex double* coefficients_of_exp;
	const complex double* coefficients_of_power;
	complex double* output_points;
};

void gen_start_time(const complex double* input_points, double* start_time, int num_input_points)
{
	double length_till[num_input_points];
	length_till[0] = 0;
	for(int i = 1; i < num_input_points; ++i)
	{
		length_till[i] = length_till[i - 1] + cabs(input_points[i] - input_points[i - 1]);
	}
	for(int i = 0; i < num_input_points - 1; ++i)
	{
		start_time[i] = period*length_till[i]/length_till[num_input_points - 1];
	}
	start_time[num_input_points - 1] = period;
	//for(int i = 0; i < num_input_points; ++i)
	//{
		//printf("start time for %i : %lf\n", i, start_time[i]);
	//}
}

complex double integral(int n1, const complex double* input_points, const double* start_time, int num_input_points)
{
	//printf("Integral, n = %i\n", n1);
	double n = -n1*2*M_PI/period;
	complex double z = 0;
	if(n1 == 0)
	{
		for(int i = 0; i < num_input_points - 1; ++i)
		{
			double delta = start_time[i + 1] - start_time[i], t1 = start_time[i], t2 = start_time[i + 1];
			complex double a = input_points[i], b = input_points[i + 1];
			z += (a - t1*(b - a)/delta)*delta + (b - a)/2*(t1 + t2);
		}
		//printf("coeff for n=0 : %lf+%lfi\n", creal(z/period), cimag(z/period));
		return z/period;
	}
	for(int i = 0; i < num_input_points - 1; ++i)
	{
		double delta = start_time[i + 1] - start_time[i], t1 = start_time[i], t2 = start_time[i + 1];
		complex double a = input_points[i], b = input_points[i + 1];
		z += -1i*(a - t1*(b - a)/delta)/n*(cexp(1i*n*t2) - cexp(1i*n*t1)) + (b - a)/(delta*n)*((-t2*1i + 1/n)*cexp(1i*n*t2) - (-t1*1i + 1/n)*cexp(1i*n*t1));
	}
	//printf("coeff for n = %i : %lf+%lfi\n", n1, creal(z/period), cimag(z/period));
	return z/period;
}

void* integral_thread(void* void_pointer_to_data_struct)
{
	struct integral_thread_data* data = (struct integral_thread_data*)void_pointer_to_data_struct;
	//printf("integral thread : %i\t%i\n", data->min_i, data->max_i);
	for(int i = data->min_i; i < data->max_i; ++i)
	{
		data->coefficients_of_exp[i] = integral(i - data->n_max, data->input_points, data->start_time, data->num_input_points);
	}

	return NULL;
}

int i_min(int i, int max_i)
{
	return ceil((double)max_i/num_threads*i);
}

int i_max(int i, int max_i)
{
	return ceil((double)max_i/num_threads*(i + 1));
}

void gen_coefficients_of_exp(complex double* coefficients_of_exp, complex double* input_points, const double* start_time, int num_input_points, int n_max)
{
	pthread_t threads[num_threads];
	struct integral_thread_data data[num_threads];
	for(int i = 0; i < num_threads; ++i)
	{
		struct integral_thread_data initialiser = {n_max, i_min(i, 2*n_max), i_max(i, 2*n_max)*(i != num_threads - 1) + (i == num_threads - 1)*(2*n_max + 1), num_input_points, input_points, start_time, coefficients_of_exp};
		data[i] = initialiser;
		pthread_create(&threads[i], NULL, integral_thread, (void*)&data[i]);
	}
	for(int i = 0; i < num_threads; ++i)
	{
		pthread_join(threads[i], NULL);
	}
	//printf("n_max = %i\nnum_input_points = %i\n", n_max, num_input_points);
	//for(int i = 0; i < 2*n_max + 1; ++i)
	//{
		//printf("coeff_exp : %lf+%lfi\n", creal(coefficients_of_exp[i]), cimag(coefficients_of_exp[i]));
	//}
}

void gen_coefficients_of_power(complex double* coefficients_of_power, int n_max)
{
	for(int i = -n_max; i <= n_max; ++i)
	{
		coefficients_of_power[i + n_max] = CMPLX(0, i*period/(2*M_PI));
	}
	//for(int i = 0; i < 2*n_max + 1; ++i)
	//{
		//printf("coeff_pow%lf+%lfi\n", creal(coefficients_of_power[i]), cimag(coefficients_of_power[i]));
	//}
}

complex double fourier(double t, const complex double* coefficients_of_exp, const complex double* coefficients_of_power, int n_max)
{
	//printf("t = %lf\n", t);
	complex double z = 0;
	for(int i = 0; i < 2*n_max + 1; ++i)
	{
		z += coefficients_of_exp[i]*cexp(coefficients_of_power[i]*t);
	}
	return z;
}

void* fourier_thread(void* void_pointer_to_data_struct)
{
	struct fourier_thread_data* data = (struct fourier_thread_data*)void_pointer_to_data_struct;
	//printf("fourier thread : %i\t%i\n", data->min_i, data->max_i);
	for(int i = data->min_i; i < data->max_i; ++i)
	{
		data->output_points[i] = fourier(i*data->dt, data->coefficients_of_exp, data->coefficients_of_power, data->n_max);
	}

	return NULL;
}

void gen_output_points(complex double* output_points, int num_output_points, const complex double* coefficients_of_exp, const complex double* coefficients_of_power, int n_max)
{
	//printf("num_output_points = %i\n", num_output_points);
	pthread_t threads[num_threads];
	struct fourier_thread_data data[num_threads];
	double dt = period/(num_output_points - 1);
	for(int i = 0; i < num_threads; ++i)
	{
		struct fourier_thread_data initialiser = {i_min(i, num_output_points - 1), i_max(i, num_output_points - 1)*(i != num_threads - 1) + (i == num_threads - 1)*(num_output_points), n_max, dt, coefficients_of_exp, coefficients_of_power, output_points};
		data[i] = initialiser;
		pthread_create(&threads[i], NULL, fourier_thread, (void*)&data[i]);
	}
	for(int i = 0; i < num_threads; ++i)
	{
		pthread_join(threads[i], NULL);
	}
}

