project(
    'aep8',
    'c',
    version: run_command('python', '-m', 'setuptools_scm', check: true).stdout().strip(),
    meson_version: '>=1.3.0',
)

add_global_arguments(
    '-DNPY_NO_DEPRECATED_API=NPY_1_19_API_VERSION',
    language: 'c',
)

compiler = meson.get_compiler('c')

if compiler.has_function(
    'exp10',
    prefix: ['#define _GNU_SOURCE', '#include <math.h>'],
)
    add_global_arguments('-D_GNU_SOURCE', '-DHAVE_EXP10', language: 'c')
elif compiler.has_function('__exp10', prefix: '#include <math.h>')
    add_global_arguments('-Dexp10=__exp10', '-DHAVE_EXP10', language: 'c')
endif

is_windows = host_machine.system() == 'windows'
if is_windows
    add_project_link_arguments('-lquadmath', language: 'c')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

py.install_sources(
    'src/aep8/__init__.py',
    'src/aep8/_core.py',
    'src/aep8/py.typed',
    subdir: 'aep8',
)

py.extension_module(
    '_irbem',
    'src/aep8/_irbem.c',
    dependencies: [py_dep, dependency('numpy'), dependency('irbem')],
    install: true,
    limited_api: '3.11',
    subdir: 'aep8',
)
