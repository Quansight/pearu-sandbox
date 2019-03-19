"""
Prerequisites: xnd, ndtypes, gumath, xndtools, numpy, gcc, gfortran

Usage:
  python setup.py develop     # builds extension module fftpack to CWD
  python -c 'import fftpack'  # test if extension module build is complete

"""
import glob
import os

from setuptools import setup, Extension
from xndtools.kernel_generator import generate_module
from argparse import Namespace
    
package = ''  # package name, when empty then modules are built into CWD
cfg = 'fftpack-kernels.cfg'
source_dir = '.'

# Generate extension module source:
from xndtools.kernel_generator import generate_module
from argparse import Namespace
m = generate_module(Namespace(config_file = cfg,
                              target_language = 'python',
                              package = package,
                              source_dir = source_dir,
                              target_file = None,
                              kernels_source_file = None,
))

# Compile fortran sources (this is a very naive approach, just to get things working)
os.system('cd fftpack && gfortran -c -O2 -fPIC *.f')
os.system('cd dfftpack && gfortran -c -O2 -fPIC *.f')

ext = Extension(m['extname'],
                include_dirs = m['include_dirs'],
                library_dirs = m['library_dirs'],
                depends = m['sources']+[cfg],
                sources = m['sources'] + ['zfft.c'],
                libraries = m['libraries'],
                extra_objects = glob.glob('fftpack/*.o')+glob.glob('dfftpack/*.o')
)

setup(ext_modules=[ext])
