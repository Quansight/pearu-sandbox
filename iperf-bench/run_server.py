import requests
import socket
import iperf3 as iperf
from iperf_utils import show, dump, get_config
hostname = socket.gethostname()

conf = get_config()

server = iperf.Server()

for section in conf:
    if section.startswith('server'):
        c = conf[section]
        if c.get('skip'):
            continue
        port = int(c.get('port', 5001))
        host = c['host']  # host IP or name
        label = c.get('label', section)
        if c.get('hostname') == hostname:
            server.port = port
            break
else:
    raise ValueError(f'Hostname `{hostname}` needs to be registered in pearu-sandbox/iperf-bench/config.conf')
    

fn = f'server-{hostname}.txt'
print(f'output: {fn}. Press Ctrl-C to stop the server.')
f = open(fn, 'w+')
while 1:
    try:
        result = server.run()
    except KeyboardInterrupt:
        print('Please upload {fn} to https://github.com/Quansight/quansight-internal-support/tree/pearu/iperf-bench/iperf-bench ')
        break
    except Exception as msg:
        print(f'Server failed: {msg}')
        continue
    show(server, result)
    dump(server, result, f)

f.close()
