include_dirs = ['include']

sources = ['main.c', 'argument-parser.c', 'initial-conditions-generator.c', 'collision-handler.c', 'phase-updates.c', 'data-dumper.c']
c_link_args = ['-lm', '-lpthread']

ideal_gas_simulation = executable('Ideal Gas Simulation', sources,
									include_directories: include_dirs,
									link_args: c_link_args,
									install: true,
									install_dir: meson.project_build_root()
								  )

test('data', ideal_gas_simulation,
# args: nthread boxsize.x boxsize.y boxsize.z num_molecules dt time_limit fps relative_mass radius temperature
	  args: ['1', '5', '5', '5', '10', '1e-6', '5', '30', '4', '0.1', '0.05'],
      timeout: 0
	)

renderer = find_program('2d-renderer.py')
test('render', renderer, timeout: 0)
