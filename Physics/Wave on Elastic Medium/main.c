#include "types.h"
#include "argument-parser.h"
#include "initializer.h"
#include "data-points-generator.h"
#include "renderer.h"

int main(int argc, char *argv[])
{
	struct Parameters param = parse_arguments(argc, argv);
	initialize(&param);
	generate_data_points(param);
	render(param);
	cleanup(param);

	return 0;
}
