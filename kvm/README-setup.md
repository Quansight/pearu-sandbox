# My workstation

* CPU: Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz, 4 cores
* MB: MSI B250M PRO-VDH (MS-7A70)
* RAM: 16GB
* GPU: NPY Quadro P2000
* Platform: Ubuntu 16.04
* GCC: 5.5

# Setup

Find instructions from Google how to setup a KVM server in your computer.
For instance, Ubuntu users could start from

  https://help.ubuntu.com/community/KVM

In the following it is assumed that the KVM server is up and running,
and one can create VM instances with properly configured network.
Also assuming that the user has permissions to run `virsh` commands.
See KVM installation guides for details.

# Creating base VM instances

Base VM instances are "read-only" instances with specific environments
that can be cloned to create test or development VM instances.

```
export SERVERS=/servers # base directory for KVM scripts, images, etc.
sudo mkdir -p $SERVERS/images # directory where VM images will situate.
export PATH=/path/to/vm_ip:$PATH
ln -s /path/to/get_vm_ip.sh $SERVERS/
```

## Ubuntu 16.04 (64-bit)

```
sudo virt-install --name ubuntu1604 --ram 6144 --disk path=/servers/images/ubuntu1604-amd64.img,size=50 --vcpus 4 --os-type linux --os-variant ubuntu16.04 --network bridge=virbr0 --nographics  --location http://archive.ubuntu.com/ubuntu/dists/xenial/main/installer-amd64/ --extra-args='console=ttyS0' --noautoconsole

$ virsh console ubuntu1604
```
Hit enter once to get the installation console working and follow instructions.

When in Softwate selection, choose:

*  OpenSSH server
*  Basic Ubuntu server
  
```
virsh start ubuntu1604
export IP=`vm_ip ubuntu1604`
ssh-copy-id $IP
ssh $IP
sudo apt-get update
sudo apt-get install
sudo shutdown -h now
```

To ensure the disk image is qcow2, check the out put of
```
sudo file /servers/images/ubuntu1604.img
```

## CentOS 7 (64-bit)

Fix the location parameter in `virt-install-centos7.sh` and run

```
sudo bash virt-install-centos7.sh
export IP=`vm_ip centos07`
ssh-copy-id $IP
ssh $IP
sudo yum update
sudo yum install wget bzip2 git pciutils nano
sudo shutdown -h now
```
