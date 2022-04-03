#include "types.h"

#include <math.h>

struct Parameters parse_arguments(int argc, char *const argv[])
{
	Float_t speed = 0.5, length = 1, time_limit = 5, dt = 1e-4;
	struct ParticleCount particle_count_prot;
	particle_count_prot.boundary = 2;
	particle_count_prot.inside = 1e4;
	particle_count_prot.total = particle_count_prot.boundary + particle_count_prot.inside;
	particle_count_prot.plot = 1e3;
	int fps = 30, ndim = 2, ndim_ng = 1, WhatToPlot = 0;

	struct Parameters param = {.time_limit=time_limit, .dt=dt, .dt_data_dump=1.0/30,
	                           .particle_count=particle_count_prot, .fps=fps, .ndim=ndim,
	                           .ndim_ng=ndim_ng, .WhatToPlot=WhatToPlot};
	param.nframes = floor(param.time_limit / param.dt_data_dump);
	param.nframes += param.nframes < param.time_limit / param.dt_data_dump;
	param.di_data_dump = param.particle_count.total / param.particle_count.plot;
	*(Float_t *)&param.dt_scaled = pow(speed, 2) / pow(length / param.particle_count.total, 2)
	                              * param.dt;
	return param;
}
