
# Prerequisites

See [README-vms](README-vms.md)

# Setting up and installing CUDA within a VM instance

In the following, instructions from

  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

are applied within VM instances.

## Cuda 9.2 on Ubuntu 16.04

```
virsh start ubuntu1604-gpu
ssh `vm_ip ubuntu1604-gpu`

sudo apt install gcc
gcc --version
gcc (Ubuntu 5.4.0-6ubuntu1~16.04.10) 5.4.0 20160609
$ uname -r
4.4.0-124-generic
sudo apt-get install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.2.88-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
sudo reboot
ssh `vm_ip ubuntu1604-gpu`
lsmod | grep nouveau # should show nothing
ls /dev/nvidia* # should show nvidia devices
```
Append to `.bashrc`:
```
export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

RECOMMENDED: test driver and samples
```
/usr/bin/nvidia-persistenced --verbose
cuda-install-samples-9.2.sh cuda-9.2-samples
cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  396.26  Mon Apr 30 18:01:39 PDT 2018
GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.10) 
nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Wed_Apr_11_23:16:29_CDT_2018
Cuda compilation tools, release 9.2, V9.2.88
cd cuda-9.2-samples/NVIDIA_CUDA-9.2_Samples/
make
bin/x86_64/linux/release/deviceQuery
bin/x86_64/linux/release/bandwidthTest
```

## Cuda 9.2 on Centos 7

```
virsh start centos70-gpu
ssh `vm_ip centos70-gpu`

uname -m && cat /etc/*release
x86_64
CentOS Linux release 7.5.1804 (Core)
sudo yum install gcc
gcc --version
gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-28)
uname -r
3.10.0-862.3.2.el7.x86_64

sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)

wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-9.2.88-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-9.2.88-1.x86_64.rpm
sudo yum clean all
```

To fix 'Requires: dkms' error
```
sudo yum install epel-release
sudo yum install dkms
```
```
sudo yum install cuda
```

Append to `.bashrc`:
```
export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo reboot

ssh `vm_ip centos70-gpu`
```

Apply https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-verifications
```
nano mk_dev_nvidia.sh # copy the script content from the docs
sudo bash mk_dev_nvidia.sh
ls /dev/nvidia* # should show nvidia devices

sudo /usr/bin/nvidia-persistenced --verbose
cuda-install-samples-9.2.sh .

cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  396.26  Mon Apr 30 18:01:39 PDT 2018
GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-28) (GCC) 
nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Wed_Apr_11_23:16:29_CDT_2018
Cuda compilation tools, release 9.2, V9.2.88

cd NVIDIA_CUDA-9.2_Samples/
make
bin/x86_64/linux/release/deviceQuery
bin/x86_64/linux/release/bandwidthTest
```
