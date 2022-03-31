#include <stdio.h>

#include <variable-types.h>

void dump_data(const vector_t *restrict position, int nrender)
{
	FILE *xfptr = fopen(X_DATA, "a"),
		 *yfptr = fopen(Y_DATA, "a");
	#ifdef VEC3D
		FILE *zfptr = fopen(Z_DATA, "a");
	#endif

	for (int i = 0; i < nrender - 1; ++i)
	{
		fprintf(xfptr, "%.6le,", position[i].x);
		fprintf(yfptr, "%.6le,", position[i].y);
		#ifdef VEC3D
			fprintf(zfptr, "%.6le,", position[i].z);
		#endif
	}
	fprintf(xfptr, "%.6le\n", position[nrender - 1].x);
	fprintf(yfptr, "%.6le\n", position[nrender - 1].y);
	fclose(xfptr);
	fclose(yfptr);
	#ifdef VEC3D
		fprintf(zfptr, "%.6le\n", position[nmolecules - 1].z);
		fclose(zfptr);
	#endif
}

// debug
void dump_statistical_data(double t, const vector_t *restrict velocity, int nmolecules)
{
	double rms_speed_squared = 0;
	vector_t dmomentum = {.x = 0, .y = 0};
	for(int i = 0; i < nmolecules; ++i)
	{
		rms_speed_squared += velocity[i].x*velocity[i].x + velocity[i].y*velocity[i].y;
		dmomentum.x += velocity[i].x;
		dmomentum.y += velocity[i].y;
	}
	rms_speed_squared /= nmolecules;
	printf("TIME ==> %7.3lf\tRMS_SQUARED ==> %7.3lf\tMOMENTUM_X ==> %7.3lf\tMOMENTUM_Y ==> %7.3lf\n",
	       t, rms_speed_squared, dmomentum.x, dmomentum.y);
	// for(int i = 0; i < nmolecules; ++i)
	// {
	// 	rms_speed_squared = velocity[i].x*velocity[i].x + velocity[i].y*velocity[i].y;
	// 	fprintf(stderr, "rms speed squared for %ith molecule = %lf\n", i, rms_speed_squared);
	// }
}
