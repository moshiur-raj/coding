#ifndef METROPOLIS_H
#define METROPOLIS_H 1

#include "common.h"

typedef struct
{
	spin_t *lattice_0;
	index_t *row, *col;
	double *drand;
	pointers_t ptr;
} met_arg_t;

extern void metropolis(met_arg_t arg);

#endif
