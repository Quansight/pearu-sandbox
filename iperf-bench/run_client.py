import os
import iperf3 as iperf
import requests
import socket
import tempfile

from tcp_latency import measure_latency
from iperf_utils import show, dump, get_config
hostname = socket.gethostname()
conf = get_config()
servers = []
tests = []
for section in conf:
    if section.startswith('server'):
        c = conf[section]
        if c.get('skip'):
            continue
        port = int(c.get('port', 5001))
        latency_port = int(c.get('latency_port', port))
        host = c['host']  # host IP or name
        label = c.get('label', section)
        servers.append(dict(label=label, port=port, host=host, latency_port=latency_port))
    elif section.startswith('iperf'):
        c = conf[section]
        if c.get('skip'):
            continue
        test = dict(section=section)
        for k, v in c.items():
            if k in ['skip']:
                continue
            if k in ['blksize', 'duration', 'num_streams']:
                value = int(v)
            elif k == 'protocol':
                value = dict(udp='udp', tcp='tcp').get(v.lower())
            elif k in ['reverse', 'zerocopy']:
                value = {'false': False, 'true': True, '0': False, '1': True}.get(v.lower())
            elif k in ['label']:
                value = v
            else:
                print(f'Warning: unrecognized iperf parameter {k}=`{v}`. Skipping.')
                continue
            if value is None:
                print(f'Warning: unsupported iperf parameter value {k}=`{v}`. Skipping.')
                continue
            test[k] = value
        tests.append(test)
    elif section.startswith('latency'):
        c = conf[section]
        if c.get('skip'):
            continue
        test = dict(section=section)
        for k, v in c.items():
            if k in ['skip']:
                continue
            if k in ['runs']:
                value = int(v)
            elif k in ['wait', 'timeout']:
                value = float(v)
            else:
                print(f'Warning: unrecognized latency parameter {k}=`{v}`. Skipping.')
                continue
            test[k] = value
        tests.append(test)

    elif section.startswith('ping'):
        c = conf[section]
        if c.get('skip'):
            continue
        test = dict(section=section)
        for k, v in c.items():
            if k in ['skip']:
                continue
            if k in ['count']:
                value = int(v)
            else:
                print(f'Warning: unrecognized ping parameter {k}=`{v}`. Skipping.')
                continue
            test[k] = value
        tests.append(test)

        
for server in servers:
    fn = f'client-{hostname}-{server["host"]}.txt'
    print(f'SERVER {server["host"]} {server["port"]}, output: {fn}')
    f = open(fn, 'w+')

    tcp_mss_default = None
    with tempfile.TemporaryDirectory() as td:
        for test in tests:
            if test['section'].startswith('iperf'):
                client = iperf.Client(verbose=False)
                for k, v in test.items():
                    setattr(client, k, v)
                if client.protocol == 'udp' and tcp_mss_default is not None:
                    client.blksize = tcp_mss_default
                client.server_hostname = server['host']
                client.port = server['port']
                result = client.run()
                show(client, result)
                f.write(f'{test["section"]},')
                dump(client, result, f)
                if client.protocol == 'tcp' and tcp_mss_default is None and result.error is None:
                    tcp_mss_default = result.tcp_mss_default
                del client
            elif test['section'].startswith('latency'):
                print(f'Measure latency..')
                data = measure_latency(host=server['host'],
                                       port=server['latency_port'],
                                       runs=test.get('runs', 10),
                                       wait=test.get('wait', 1),
                                       timeout=test.get('timeout', 5))
                data = [d for d in data if not d]
                if data:
                    print(f'{test["section"]} min/avg/max: {min(data)}/{sum(data)/len(data)}/{max(data)} ms')
                    f.write(f'{test["section"]},{min(data)},{sum(data)/len(data)},{max(data)}\n')
                    f.flush()
                else:
                    print('Skipping latency test: no data.')
            elif test['section'].startswith('ping'):
                print(f'Ping..')
                tfn = os.path.join(td, 'ping.txt')
                cmd = f'ping {server["host"]} -n -c {test.get("count", 10)} -q > {tfn}'
                status = os.system(cmd)
                if status:
                    print(f'Warning: {cmd} returned {status}')
                tf = open(tfn, 'r')
                ping_text = tf.read().split('\n\n', 1)[-1]
                f.write(ping_text)
                tf.close()
                print(ping_text)
            else:
                raise NotImplementedError(repr(test))

    f.close()

print('Please run this script a couple of times during the working day.')
print('When done, please upload the following files to https://github.com/Quansight/quansight-internal-support/tree/pearu/iperf-bench/iperf-bench ')
for server in servers:
    fn = f'client-{hostname}-{server["host"]}.txt'
    print(f'  {fn}')
