#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <sys/sysinfo.h>

#include <variable-types.h>

#define NUM_ARG (12)

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
		        " args: nthead nmolecules nrender dt time_limit fps boxsize.x boxsize.y boxsize.z"
		        " relative_mass radius temperature.", NUM_ARG);
		exit(EXIT_FAILURE);
	}

	if (!args_are_numbers(argc, argv))
	{
		fprintf(stderr, "ARGUMENT ERROR: Please Use Decimal Numbers For Arguments.\n");
		exit(EXIT_FAILURE);
	}
}

void check_params(const struct SimulationParameters *restrict param)
{
	if (param->nthread > param->nmolecules)
	{
		fprintf(stderr, "WARNING: number of tasks < number of threads\n");
		exit(EXIT_FAILURE);
	}

	if (param->nmolecules < 0 || param->dt < 0 || param->time_limit < 0 || param->frametime < 0
	     || param->relative_mass < 0 || param->radius < 0 || param->temperature < 0)
	{
		fprintf(stderr, "ERROR: use positive values");
		exit(EXIT_FAILURE);
	}

	double min_box_dim = 4*param->radius;
	if (param->boxsize.x < min_box_dim || param->boxsize.y < min_box_dim
	     || param->boxsize.z < min_box_dim)
	{
		fprintf(stderr, "ERROR: boxsize too small.\n");
		exit(EXIT_FAILURE);
	}

	if (param->nmolecules*M_PI*param->radius*param->radius
	    > param->boxsize.x*param->boxsize.y*param->boxsize.z / 2)
	{
		fprintf(stderr, "ERROR: not enough space for the molecules\n");
		exit(EXIT_FAILURE);
	}
}

void parse_args(int argc, char *argv[], struct SimulationParameters *restrict param)
{
	check_args(argc, argv);

	param->nthread        = (int)strtol(argv[1], NULL, 10);
	param->nmolecules     = (int)strtol(argv[2], NULL, 10);
	param->nrender        = (int)strtol(argv[3], NULL, 10);
	param->dt             = strtod(argv[4], NULL);
	param->time_limit     = strtod(argv[5], NULL);
	param->frametime      = 1/strtod(argv[6], NULL);
	param->boxsize.x      = strtod(argv[7], NULL);
	param->boxsize.y      = strtod(argv[8], NULL);
	param->boxsize.z      = strtod(argv[9], NULL);
	param->relative_mass  = strtod(argv[10], NULL);
	param->radius         = strtod(argv[11], NULL);
	param->temperature    = strtod(argv[12], NULL);

	param->radius_squared = param->radius*param->radius;

	double dspeed         = sqrt(3*8.314*param->temperature / param->relative_mass)*1e-3;
	param->mca_factor     = 5*sqrt(2*8.314*param->temperature / param->relative_mass)/param->radius;
	param->wca_factor     = dspeed/param->dt;
	// debug
	param->compare_radius = param->mca_factor/param->radius;

	// Make the number of molecules even. This is done so it is easier to maintain conservation
	// of momentum when generating velocities.
	param->nmolecules += param->nmolecules & 1;


	if (param->nthread <= 0)
	{
		param->nthread = get_nprocs();
	}

	check_params(param);
}
