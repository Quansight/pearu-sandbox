
## Running the server

```
git clone git@github.com:Quansight/pearu-sandbox.git
conda env create --file=pearu-sandbox/iperf-bench/iperf-benchmark-server.yaml
conda activate iperf-benchmark-server
pip install iperf3
# make sure that server hostname is registered in pearu-sandbox/iperf-bench/config.conf
python pearu-sandbox/iperf-bench/run_server.py
```
Upload the file `server-<local hostname>.txt` to https://github.com/Quansight/quansight-internal-support/tree/pearu/iperf-bench/iperf-bench

## Running the client

```
git clone git@github.com:Quansight/pearu-sandbox.git
conda env create --file=pearu-sandbox/iperf-bench/iperf-benchmark-client.yaml
conda activate iperf-benchmark-client
pip install iperf3 tcp_latency
python pearu-sandbox/iperf-bench/run_client.py  # it can take a 1-2 minutes to complete
```
Upload the files `client-<local hostname>-<server name>.txt` to https://github.com/Quansight/quansight-internal-support/tree/pearu/iperf-bench/iperf-bench
