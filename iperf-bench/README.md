
# iperf-bench

The aim of the following scripts is to gather information on
client-server connectivity (network throughput and latency). This
information will be used to find an optimal location of development
servers (such as qgpu3).


## iperf service

The candidates of the servers are listed in `server` sections of
[config.conf](config.conf) file. Each server runs `iperf` service (via
[run_server.py](run_server.py)) that logs connections to
`server-<server hostname>.txt` file. To run the server (client users
should ignore the following instructions), one must prepare a
benchmark environment as follows:
1. Clone `Quansight/pearu-sandbox` (if you have not done it yet):
   ```bash
   git clone git@github.com:Quansight/pearu-sandbox.git
   ```

2. Add server information to [config.conf](config.conf) and merge the
   changes.

3. Prepare the server environment:
   ```bash
   cd Quansight/pearu-sandbox/iperf-bench
   conda env create --file=iperf-benchmark-server.yaml
   conda activate iperf-benchmark-server
   pip install iperf3
   ```

4. Run the ipref service:
   ```bash
   python run_server.py
   ```

   The service should run until all clients have completed their
   benchmark runs.  In the case of service crash, just rerun the
   `run_server.py` script that will continue collecting the data.

5. When all benchmark data is collected, upload `server-<server
   hostname>.txt` to
   https://github.com/Quansight/quansight-internal-support/tree/pearu/iperf-bench/iperf-bench
   for futher analysis.  Or just send the files to Pearu via E-mail or
   Slack.


## Client side benchmarking

The clients are hosts that connect to development servers. The client
benchmark script [run_client.py](run_client.py) connects to the
`iperf` service providers and logs benchmark results to
`client-<client hostname>-<server hostname>.txt` file. In addition,
the client benchmark scripts runs `ping` to record the latency data.

To run th client benchmarks, first prepare a benchmark environment as
follows:
1. Clone `Quansight/pearu-sandbox` (if you have not done it yet):
   ```bash
   git clone git@github.com:Quansight/pearu-sandbox.git
   ```

2. Prepare the client environment:
   ```bash
   cd Quansight/pearu-sandbox/iperf-bench
   conda env create --file=iperf-benchmark-client.yaml
   conda activate iperf-benchmark-client
   pip install iperf3 tcp_latency
   ```

3. Run the benchmarks script:
   ```bash
   python run_client.py
   ```

   Please re-run the script several times during your working day
   with, say, 1-4h intervals.

4. Finally, upload all `client-<client hostname>-<server
   hostname>.txt` files to
   https://github.com/Quansight/quansight-internal-support/tree/pearu/iperf-bench/iperf-bench
   for futher analysis. Or just send the files to Pearu via E-mail or Slack.
