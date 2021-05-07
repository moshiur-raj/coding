#include <stdlib.h>
#include <stdio.h>

double Alpha,
	   Length,
	   TimeMin,
	   TimeMax;
int NumFrames,
	NumPositions,
	Nmax;

// making the initial condition somehow work
#define T_MAX 300
#define T_MIN 200

extern inline double initial_temp(double position)
{
	return T_MAX*(position <= Length/2) + T_MIN*(position > Length/2);
}

void initialize(int argc, char **argv)
{
	Alpha = strtod(argv[1], NULL);
	NumFrames = strtol(argv[2], NULL, 10);
	NumPositions = strtol(argv[3], NULL, 10);
	Nmax = strtol(argv[4], NULL, 10);
	TimeMin = strtod(argv[5], NULL);
	TimeMax = strtod(argv[6], NULL);
	Length = strtod(argv[7], NULL);
}
