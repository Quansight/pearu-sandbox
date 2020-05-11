#!/usr/bin/env python
#
# Author: Pearu Peteson
# Created: May 2020

"""
  class ElapsedTime {
    /*
      Report time elapsed in executing a block of code. Usage::
      {
        ElapsedTime _("code");
        <block code>
      }
      Requires: #include <chrono>
     */
  public:
    ElapsedTime(std::string label) : label(label), start(std::chrono::high_resolution_clock::now()) {}
    ~ElapsedTime() {
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      std::cout << "ElapsedTime: " << label << " took " << duration.count() << " ns" << std::endl;
    }
  private:
    std::string label;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
  };
"""

import re
import sys
from collections import defaultdict

match = re.compile(r'ElapsedTime:\s*(?P<label>.*?)\s*took\s*(?P<count>\d+)\s*(?P<unit>\w+)').match


cummulative_results = defaultdict(int)
unit = None
total = 0
for line in sys.stdin.readlines():
    m = match(line)
    if m is not None:
        cummulative_results[m.group('label')] += int(m.group('count'))
        total += int(m.group('count'))
        if unit is None:
            unit = m.group('unit')
        else:
            assert unit == m.group('unit'), (unit, m.group('unit'))

def new_unit(scale, unit):
    if scale == 1:
        return scale, unit
    if scale == 1000:
        return scale, dict(ns='us', us='ms', ms='s')[unit]
    raise NotImplementedError((scale, unit))
            
scale, unit = new_unit(1000, unit)
            
for label in sorted(cummulative_results):
    print(f'{label:10} took {cummulative_results[label]/scale:10.2f} {unit} - {100*cummulative_results[label]/total:10.0f} %')

print(f'Total time: {total/scale:14.2f} {unit} - {"100":>10} %')
