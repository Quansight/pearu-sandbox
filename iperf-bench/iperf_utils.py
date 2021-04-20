import os
import requests
import configparser

def get_config(local=False):
    conf = configparser.ConfigParser()
    if local:
        f = open(os.path.join(os.path.dirname(__file__), 'config.conf'))
        conf.read_string(f.read())
        f.close()
    else:
        try:
            r = requests.get('https://raw.githubusercontent.com/Quansight/pearu-sandbox/master/iperf-bench/config.conf')
            conf.read_string(r.text)
        except Exception as msg:
            print(f'Failed to get config.conf: {msg}. Using local one.')
            return get_config(local=True)
    return conf


def get_ipinfo(ip):
    try:
        import ipinfo
    except ImportError:
        return
    try:
        access_token = open(os.path.join(os.path.dirname(__file__), 'ipinfo_access_token')).read()
    except Exception:
        return
    handler = ipinfo.getHandler(access_token)
    return handler.getDetails(ip).all


def show(target, result):

    if result.error is None:
        label = f'{result.protocol}-{result.blksize}-{result.duration}-{result.num_streams}-{result.reverse}-{result.omit}'
        print(label)
        if target.role == 'c':
            print(f'  {result.local_host}:{result.local_port} connected to {result.remote_host}:{result.remote_port}')
            print(f'  {result.system_info}; time={result.timesecs}; CPU local={result.local_cpu_total:.1f}:{result.local_cpu_user:.1f}:{result.local_cpu_system:.1f}% remote={result.remote_cpu_total:.1f}:{result.remote_cpu_user:.1f}:{result.remote_cpu_system:.1f}%')
            if result.protocol.lower() == 'tcp':
                print(f'  tcp_mss_default={result.tcp_mss_default} retransmits={result.retransmits}')
                print(f'  > {result.sent_bytes} bytes @ {result.sent_Mbps} Mbps')
                print(f'  < {result.received_bytes} bytes @ {result.received_Mbps} Mbps')
            elif result.protocol.lower() == 'udp':
                print(f'  {result.bytes} bytes @ {result.Mbps} Mbps, jitter={result.jitter_ms} ms, packets:lost={result.packets}:{result.lost_packets}')
        elif target.role == 's':
            print(f'  {result.local_host}:{result.local_port} received connection from {result.remote_host}:{result.remote_port}')
            if 0:
                d = get_ipinfo(result.remote_host)
                if d is not None:
                    print('  {ip} info: {country_name}, {region}, {city}, org={org}, loc={loc}'.format_map(d))
            print(f'  {result.system_info}; time={result.timesecs}; CPU local={result.local_cpu_total:.1f}:{result.local_cpu_user:.1f}:{result.local_cpu_system:.1f}% remote={result.remote_cpu_total:.1f}:{result.remote_cpu_user:.1f}:{result.remote_cpu_system:.1f}%')
            if result.protocol.lower() == 'tcp':
                print(f'  tcp_mss_default={result.tcp_mss_default}')
                print(f'  > {result.sent_bytes} bytes @ {result.sent_Mbps} Mbps')
                print(f'  < {result.received_bytes} bytes @ {result.received_Mbps} Mbps')
            elif result.protocol.lower() == 'udp':
                print(f'  {result.bytes} bytes @ {result.Mbps} Mbps, jitter={result.jitter_ms} ms, packets:lost={result.packets}:{result.lost_packets}')
    else:
        print(f'Test failed: {result.error}')


def dump(target, result, f):
    if result.error is None:
        label = f'{result.protocol}-{result.blksize}-{result.duration}-{result.num_streams}-{result.reverse}-{result.omit}'
        f.write(f'{result.timesecs},{result.local_host},{result.remote_host},{label},')
        if target.role == 'c':
            if result.protocol.lower() == 'tcp':
                f.write(f'>{result.sent_Mbps} Mbps,<{result.received_Mbps} Mbps,{result.retransmits}\n')
            else:
                f.write(f'{result.Mbps} Mbps,{result.jitter_ms} ms,{result.packets},{result.lost_packets}\n')
        else:
            if result.protocol.lower() == 'tcp':
                f.write(f'>{result.sent_Mbps} Mbps,<{result.received_Mbps} Mbps\n')
            else:
                f.write(f'{result.Mbps} Mbps,{result.jitter_ms} ms,{result.packets},{result.lost_packets}\n')
    else:
        f.write(f'{result.error}\n')
    f.flush()
