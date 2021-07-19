#ifndef VARIABLE_TYPES_H
#define VARIABLE_TYPES_H 1

#define X_DATA "xdata.csv"
#define Y_DATA "ydata.csv"
#define Z_DATA "zdata.csv"

struct vec2
{
	double x, y;
};

struct vec3
{
	double x, y, z;
};

#define VEC2D 1
typedef struct vec2 vector_t;

struct SimulationParameters
{
	struct {double x, y, z;} boxsize;
	int num_molecules, nthread;
	double dt, time_limit, frametime;

	double relative_mass, radius, temperature;

	// wca_factor = wall collision acceleration factor
	// mca_factor = molecular collision acceleration factor
	double radius_squared, wca_factor, mca_factor;

	vector_t *restrict position, *restrict velocity, *restrict acceleration;
};

#endif
