#include "types.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

void render(struct Parameters param)
{
	pid_t pid = fork();
	if (pid == 0)
	{
		int size = 64;
		char bin_name[size], renderer_path[size], renderPositionVsTime[size], fps[size],
		     nframes[size], ndim[size], nparticles[size], shm_block_name[size],
		     min[param.ndim][size], max[param.ndim][size];
		snprintf(bin_name, size, "%s", "python");
		snprintf(renderer_path, size, "%s", "./renderer.py");
		snprintf(renderPositionVsTime, size, "%i", param.WhatToPlot);
		snprintf(fps, size, "%.6le", param.fps);
		snprintf(nframes, size, "%i", param.nframes);
		snprintf(ndim, size, "%i", param.ndim);
		snprintf(nparticles, size, "%i", param.particle_count.plot);
		snprintf(shm_block_name, size, "%s", SHM_BLOCK_NAME);
		for (int i = 0; i < param.ndim; ++i)
		{
			snprintf(min[i], size, "%.6le", param.min[i]);
			snprintf(max[i], size, "%.6le", param.max[i]);
		}

		size = 8 + 2 * param.ndim + 1;
		char *argv[size];
		argv[0] = bin_name;
		argv[1] = renderer_path;
		argv[2] = renderPositionVsTime;
		argv[3] = fps;
		argv[4] = nframes;
		argv[5] = ndim;
		argv[6] = nparticles;
		argv[7] = shm_block_name;
		for (int i = 0, j = 8; i < param.ndim; ++i)
		{
			argv[j++] = min[i];
			argv[j++] = max[i];
		}
		argv[size - 1] = NULL;

		execv("/usr/bin/python", argv);
	}
	wait(NULL);
}
