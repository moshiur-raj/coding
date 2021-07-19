#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/sysinfo.h>

#include <variable-types.h>

#define NUM_ARG (11)

static int str_is_number(const char *str)
{
	int temp = 0;
	while (*str != '\0')
	{
		temp += !(isdigit(*str) || *str == '-' || *str == '.' || *str == 'e');
		str++;
	}

	return !temp;
}

static int args_are_numbers(int argc, char *argv[])
{
	int temp = 0;
	for (int i = 1; i < argc; ++i)
	{
		temp += !str_is_number(argv[i]);
	}

	return !temp;
}

static void check_args(int argc, char *argv[])
{
	if (argc != NUM_ARG + 1)
	{
		fprintf(stderr, "ARGUMENT ERROR: There Should Be %i Arguments."
		        " args: boxsize.x boxsize.y boxsize.z num_molecules dt time_limit fps"
		        " relative_mass radius temperature.", NUM_ARG);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	if (!args_are_numbers(argc, argv))
	{
		fprintf(stderr, "ARGUMENT ERROR: Please Use Decimal Numbers For Arguments.\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
}

void check_params(const struct SimulationParameters *restrict param)
{
	if (param->nthread > param->num_molecules)
	{
		fprintf(stderr, "WARNING: number of tasks < number of threads\n");
		fflush(stderr);
	}

	if (param->num_molecules < 0 || param->dt < 0 || param->time_limit < 0 || param->frametime < 0
	     || param->relative_mass < 0 || param->radius < 0 || param->temperature < 0)
	{
		fprintf(stderr, "ERROR: use positive values");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	double min_box_dim = 4*param->radius;
	if (param->boxsize.x < min_box_dim || param->boxsize.y < min_box_dim
	     || param->boxsize.z < min_box_dim)
	{
		fprintf(stderr, "ERROR: boxsize too small.\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
}

void parse_args(int argc, char *argv[], struct SimulationParameters *restrict param)
{
	check_args(argc, argv);

	param->nthread       = (int)strtol(argv[1], NULL, 10);
	param->boxsize.x     = strtod(argv[2], NULL);
	param->boxsize.y     = strtod(argv[3], NULL);
	param->boxsize.z     = strtod(argv[4], NULL);
	param->num_molecules = (int)strtol(argv[5], NULL, 10);
	param->dt            = strtod(argv[6], NULL);
	param->time_limit    = strtod(argv[7], NULL);
	param->frametime		= 1/strtod(argv[8], NULL);
	param->relative_mass = strtod(argv[9], NULL);
	param->radius        = strtod(argv[10], NULL);
	param->temperature   = strtod(argv[11], NULL);

	param->radius_squared = param->radius*param->radius;
	param->mca_factor     = 1e3/param->radius;
	param->wca_factor     = 1e3;

	// Make the number of molecules even. This is done so it is easier to maintain conservation
	// of momentum when generating velocities.
	param->num_molecules += param->num_molecules & 1;


	if (param->nthread <= 0)
	{
		param->nthread = get_nprocs();
	}

	check_params(param);
}
