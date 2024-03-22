#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

#include "config.h"

static void exec_2d_renderer(void)
{
	pid_t pid = fork();
	if (pid == 0)
	{
		char observed_point_count[16], point_count[16];
		snprintf(observed_point_count, 16, "%i", OBSERVED_POINT_COUNT);
		snprintf(point_count, 16, "%i", POINT_COUNT);
		execlp("./2d-graph-renderer.py", "2d-graph-renderer.py",
		       SHM_BLOCK1_NAME, point_count, SHM_BLOCK2_NAME, observed_point_count,
		       X_LABEL, Y_LABEL, LEGEND, NULL);
	}
	waitpid(pid, NULL, 0);
}

void render_graph(double (*x_out)[POINT_COUNT], double (*y_out)[POINT_COUNT])
{
	if (X_DIM == 1 && Y_DIM == 1)
	{
		exec_2d_renderer();
	}
}
