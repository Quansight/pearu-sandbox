
# Access VM instance via SSH tunnel

To access vm instance outside the host, on possibility to create a SSH tunnel.
Let's define:
* REMOTE: host where ssh access is attempted
* HOST: KVM server
* GUEST: IP of VM instance seen from HOST

To create a tunnel, run from REMOTE:
```
ssh -f -L 2222:GUEST:22 HOST -N
```
To access vm instance from REMOTE via ssh, run
```
ssh HOST -p 2222 # this should give GUEST prompt
```
For instance, when using tramp mode in Emacs (in REMOTE), you can open GUEST file using
```
  C-x C-f /localhost#2222:<path to GUEST file>
```

# Access VM instance via second NIC

I'd prefer this appoach but have to get a second NIC installed to my workstation, first.
