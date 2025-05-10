project(
    'aep8',
    'c',
    'fortran',
    version: run_command('python', '-m', 'setuptools_scm').stdout().strip(),
    meson_version: '>=0.64.0',
)

add_global_arguments('-DNPY_NO_DEPRECATED_API=NPY_1_19_API_VERSION', language: 'c')

is_windows = host_machine.system() == 'windows'
if is_windows
    add_project_link_arguments('-lquadmath', language : 'c')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_f2py = run_command(
    py,
    [
        '-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())',
    ],
    check: true,
).stdout().strip()

pyext_source = custom_target(
    '_irbem.c',
    input: ['src/aep8/_irbem.pyf'],
    output: ['_irbemmodule.c', '_irbem-f2pywrappers.f'],
    command: [py, '-m', 'numpy.f2py', '@INPUT@', '--lower'],
)

py.install_sources(
    'src/aep8/__init__.py',
    'src/aep8/_core.py',
    'src/aep8/py.typed',
    subdir: 'aep8'
)

py.extension_module(
    '_irbem',
    pyext_source,
    incdir_f2py / 'fortranobject.c',
    dependencies: [py_dep, dependency('numpy'), dependency('irbem')],
    include_directories: incdir_f2py,
    install: true,
    subdir: 'aep8',
)
