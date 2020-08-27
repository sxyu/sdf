from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = '0.1.4'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """
    def __str__(self):
        import pybind11
        return pybind11.get_include()


def find_eigen(min_ver=(3, 2, 0)):
    """Helper to find or download the Eigen C++ library"""
    import re, os
    try_paths = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        os.path.expanduser('~/.local/include/eigen3'),
        'C:/Program Files/eigen3',
        'C:/Program Files (x86)/eigen3',
    ]
    WORLD_VER_STR = "#define EIGEN_WORLD_VERSION"
    MAJOR_VER_STR = "#define EIGEN_MAJOR_VERSION"
    MINOR_VER_STR = "#define EIGEN_MINOR_VERSION"
    EIGEN_WEB_URL = 'https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2'
    TMP_EIGEN_FILE = 'tmp_eigen.tar.bz2'
    TMP_EIGEN_DIR = 'eigen-3.3.7'
    min_ver_str = '.'.join(map(str, min_ver))

    eigen_path = None
    for path in try_paths:
        macros_path = os.path.join(path, 'Eigen/src/Core/util/Macros.h')
        if os.path.exists(macros_path):
            macros = open(macros_path, 'r').read().split('\n')
            world_ver, major_ver, minor_ver = None, None, None
            for line in macros:
                if line.startswith(WORLD_VER_STR):
                    world_ver = int(line[len(WORLD_VER_STR):])
                elif line.startswith(MAJOR_VER_STR):
                    major_ver = int(line[len(MAJOR_VER_STR):])
                elif line.startswith(MINOR_VER_STR):
                    minor_ver = int(line[len(MAJOR_VER_STR):])
            if not world_ver or not major_ver or not minor_ver:
                print('Failed to parse macros file', macros_path)
            else:
                ver = (world_ver, major_ver, minor_ver)
                ver_str = '.'.join(map(str, ver))
                if ver < min_ver:
                    print('Found unsuitable Eigen version', ver_str, 'at',
                          path, '(need >= ' + min_ver_str + ')')
                else:
                    print('Found Eigen version', ver_str, 'at', path)
                    eigen_path = path
                    break

    if eigen_path is None:
        try:
            import urllib.request
            print("Couldn't find Eigen locally, downloading...")
            req = urllib.request.Request(
                EIGEN_WEB_URL,
                data=None,
                headers={
                    'User-Agent':
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'
                })

            with urllib.request.urlopen(req) as resp,\
                  open(TMP_EIGEN_FILE, 'wb') as file:
                data = resp.read()
                file.write(data)
            import tarfile
            tar = tarfile.open(TMP_EIGEN_FILE)
            tar.extractall()
            tar.close()

            eigen_path = TMP_EIGEN_DIR
            os.remove(TMP_EIGEN_FILE)
        except:
            print('Download failed, failed to find Eigen')

    if eigen_path is not None:
        print('Found eigen at', eigen_path)

    return eigen_path


ext_modules = [
    Extension(
        'pysdf',
        # Sort input source files to ensure bit-for-bit reproducible builds
        # (https://github.com/pybind/python_example/pull/53)
        sorted(['src/sdf.cpp', 'pybind.cpp']),
        include_dirs=[
            'include',
            # Path to pybind11 headers
            get_pybind_include(),
            # Eigen 3
            find_eigen()
        ],
        language='c++'),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [
                ('VERSION_INFO',
                 '"{}"'.format(self.distribution.get_version()))
            ]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name='pysdf',
    version=__version__,
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description=
    'SDF: Convert triangle mesh to continuous signed distance function',
    long_description='',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
