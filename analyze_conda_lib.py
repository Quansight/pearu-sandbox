#!/usr/bin/env python
"""
Analyze libraries in conda packages:
1. determine the used C++ ABI versions
2. if std::string is used
3. is C or C++ library

Usage:

  conda activate <environment-name>
  python analyze_conda_lib.py

  The output will contain library name and analysis results.

For instance,

  python analyze_conda_lib.py  | grep C++ | grep std::string

will show all libraries that use pre-CXX11ABI
"""
# Author: Pearu Peterson
# Created: Janyary 2019


import os
import re
import sys
from glob import glob
from subprocess import Popen, PIPE
    
prefix = os.environ.get('CONDA_PREFIX', '')
if not prefix:
    print('Not in a conda environment (CONDA_PREFIX is missing or empty)')
    sys.exit()
prefix_lib = os.path.join(prefix, 'lib')

def get_symbols(fn, options = [], tags = [], skip_tags = []):
    nm_process = Popen(['nm', '--demangle']+options+[os.path.abspath(fn)], stdout=PIPE)
    out, err = nm_process.communicate()
    if err:
        print(err)
        raise RuntimeError('calling nm on `{}` failed'.format(fn))
    for line in out.splitlines():
        value = line[:16].strip().decode()
        assert line[16:17]==b' ',repr((line, line[16]))
        tag = line[17:18].decode()
        assert line[18:19]==b' ',repr(line)
        symbol = line[19:].rstrip().decode()
        if tag in skip_tags:
            continue
        if tags and tag not in tags:
            continue
        yield value, tag, symbol

get_undefined_symbols = lambda fn: get_symbols(fn, options = ['--undefined-only'], tags = ['U'])
get_defined_symbols = lambda fn: get_symbols(fn, options = ['--defined-only'], skip_tags = 'rgGdbBwW')

cxx_match = re.compile(r'.*[:][:]__cxx(?P<n>\d*)[:][:]').match
std_string_match = re.compile(r'.*std[:][:](basic_|)string').match

for fn in sorted(glob(os.path.join(prefix_lib, '*.so'))):
    clib = True
    cxxlib = False
    cxx_abis = set()
    uses_std_string = False
    uses_cxx11 = False

    for value, tag, symbol in get_defined_symbols(fn):
        if clib and '::' in symbol:
            clib = False
            cxxlib = True
        if not uses_cxx11 and '::__cxx11::' in symbol:
            uses_cxx11 = True
        m = cxx_match(symbol)
        if m:
            cxx_abis.add(m.group('n'))
        if not uses_std_string and std_string_match(symbol):
            uses_std_string = True
        #print(symbol)
    info = []
    if clib:
        info.append('C library')
    if cxxlib:
        info.append('C++ library')
    if cxx_abis:
        info.append('CXXABI={}'.format(','.join(sorted(cxx_abis))))
    if uses_std_string:
        info.append('uses std::string')
    print('{}: {}'.format(fn, ', '.join(sorted(info))))
