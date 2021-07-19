project('Ideal Gas Simulation', 'c',
  version: '1.0',
  default_options: [
  'warning_level=3',
  'c_std=gnu18',
  'buildtype=custom',
  'optimization=2',
  'debug=false',
  ]
)

global_c_args = [
  '-fsigned-char',
  '-fPIC',
  '-fPIE',
  '-fno-plt',
  '-march=native',
  '-Winline',
  '-Wconversion',
  '-pipe',
  '-flto'
]
add_project_arguments(global_c_args, language: 'c')

global_c_link_args = [
  '-pie',
  '-flto',
  '-Wl,-z,relro',
  '-Wl,-z,now'
]
add_project_link_arguments(global_c_link_args, language: 'c')

subdir('src')
