import os
import time
import subprocess

cmd = 'ps -o vsize -u pearu | tail -n +2 | paste -sd+ - | bc'
max_vsize = 0
min_vsize = 2**32
start_time = time.time()

fn = 'trac_max_memory-%s.log' % (int(start_time))
print('full log will be saved to %s' % (fn), flush=True)
f = open(fn, 'w')
try:
    while True:
        vsize = int(os.popen(cmd).read())
        if max_vsize < vsize:
            print('updated max vsize from %s to %s:\n%s'
                  % (max_vsize, vsize, os.popen('ps -o vsize,args -u pearu --sort=-vsize | head -n 5').read()),
                  flush=True)
            max_vsize = vsize
        min_vsize = min(min_vsize, vsize)
        print('%s %s %s %s' % (time.time() - start_time, min_vsize, vsize, max_vsize), flush=True, file=f)
        
        time.sleep(1)
except KeyboardInterrupt:
    f.close()
    print('min/current/max vsize=%s/%s/%s' % (min_vsize, vsize, max_vsize), flush=True)
    print(fn)
