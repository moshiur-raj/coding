#ifndef TYPES_H
#define TYPES_H 1

typedef double Float_t;
typedef int Int_t;

struct ParticleCount
{
	Int_t boundary, inside, total, plot;
};

struct ParticleIndex
{
	Int_t *restrict boundary;
	Int_t *restrict inside;
};

struct Parameters
{
	void *position, *velocity, *neighbor, *shm;
	Float_t *restrict min, *restrict max;
	const Float_t time_limit, dt, fps, dt_scaled;
	Int_t nframes, di_data_dump;
	const int ndim, ndim_ng, WhatToPlot;
	struct ParticleCount particle_count;
	struct ParticleIndex index;
};

#endif
