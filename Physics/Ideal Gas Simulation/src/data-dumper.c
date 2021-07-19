#include <stdio.h>

#include <variable-types.h>

void dump_data(vector_t *restrict position, int num_molecules)
{
	FILE *xfptr = fopen(X_DATA, "a"),
		 *yfptr = fopen(Y_DATA, "a");
	#ifdef VEC3D
		FILE *zfptr = fopen(Z_DATA, "a");
	#endif

	for (int i = 0; i < num_molecules - 1; ++i)
	{
		fprintf(xfptr, "%.6le,", position[i].x);
		fprintf(yfptr, "%.6le,", position[i].y);
		#ifdef VEC3D
			fprintf(zfptr, "%.6le,", position[i].z);
		#endif
	}
	fprintf(xfptr, "%.6le\n", position[num_molecules - 1].x);
	fprintf(yfptr, "%.6le\n", position[num_molecules - 1].y);
	fclose(xfptr);
	fclose(yfptr);
	#ifdef VEC3D
		fprintf(zfptr, "%.6le\n", position[num_molecules - 1].z);
		fclose(zfptr);
	#endif
}
