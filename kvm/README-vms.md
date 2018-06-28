# Prerequisites

See [README-setup](README-setup.md)

# Creating development VM instances

```
sudo virt-clone --original <base VM name> --auto-clone -n <development VM name>
```

where base VM name is one node from the following tree:

* ubuntu1604
  * ubuntu1604-mconda36
  * ubuntu1604-gpu
    * ubuntu1604-gpu-mconda36
* centos70
  * centos70-mconda36
  * centos70-gpu
  
# Creating specific base VM instances

## Setup development environment in VM instances

SSH to VM and run
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
git config --global user.email "<YOUR EMAIL AT GITHUB>"
git config --global user.name "<YOUR NAME>"
```

## Miniconda 3.6 for Ubuntu

```
sudo virt-clone --original ubuntu1604 --auto-clone -n ubuntu1604-mconda36
virsh start ubuntu1604-mconda36
ssh `vm_ip ubuntu1604-mconda36`
# Apply 'Setup development environment in VM instances'
sudo shutdown -h now
```

## Miniconda 3.6 for Centos

```
sudo virt-clone --original centos70 --auto-clone -n centos70-mconda36
virsh start centos70-mconda36
ssh `vm_ip centos70-mconda36`
# Apply 'Setup development environment in VM instances'
sudo shutdown -h now
```

## Setup Ubuntu host for CUDA VM instances

To passthrough CUDA enabled GPU card to VM, I followed instructions from
https://arrayfire.com/using-gpus-kvm-virutal-machines/

Enable IOMMU and add Devices for PCI passthrough:
```
/etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on"
sudo grub-mkconfig -o /boot/grub/grub.cfg

/etc/modprobe.d/blacklist-nouveau.conf:
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off

/etc/modprobe.d/nouveau-kms.conf:
options nouveau modeset=0

reboot

dmesg | grep -e DMAR -e IOMMU
[    0.000000] DMAR: IOMMU enabled
```

Enable VT and VT-D in BIOS

```
for iommu_group in $(find /sys/kernel/iommu_groups/ -maxdepth 1 -mindepth 1 -type d); do echo "IOMMU group $(basename "$iommu_group")"; for device in $(ls -1 "$iommu_group"/devices/); do echo -n $'\t'; lspci -nns "$device"; done; done
	01:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:1c30] (rev a1)
	01:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:10f1] (rev a1)

/etc/modprobe.d/vfio.conf:
options vfio-pci ids=10de:1c30,10de:10f1

/etc/initramfs-tools/modules:
vfio
vfio_iommu_type1
vfio_pci
vfio_virqfd

sudo update-initramfs -u

reboot

/etc/libvirt/qemu.conf:
nvram = ["/usr/share/OVMF/OVMF_CODE.fd:/usr/share/OVMF/OVMF_VARS.fd"]
sudo systemctl restart libvirt-bin

sudo virsh nodedev-list
pci_0000_01_00_0 # gpu
pci_0000_01_00_1 # sound

sudo virsh nodedev-detach pci_0000_01_00_0
sudo virsh nodedev-detach pci_0000_01_00_1
```

## Attaching GPU to VM instance

```
sudo virt-clone --original ubuntu1604 --auto-clone -n ubuntu1604-gpu
virsh edit ubuntu1604-gpu
sudo virsh nodedev-dumpxml pci_0000_01_00_0
# add to <devices>:
    <hostdev mode='subsystem' type='pci' managed='yes'>
      <source>
        <address domain='0x0000' bus='0x01' slot='0x00' function='0x0'/>
      </source>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x08' function='0x0'/>
      <rom bar='off'/>
    </hostdev>
```

Got rom bar hint from https://bugs.launchpad.net/ubuntu/+source/seabios/+bug/1181777
that resolved the following problem: clone-edit-start "hanged" vm while host CPU at 100%.

```
virsh start ubuntu1604-gpu
ssh `vm_ip ubuntu1604-gpu`
lspci | grep -i nvidia
00:08.0 VGA compatible controller: NVIDIA Corporation Device 1c30 (rev a1)
sudo shutdown -h now

sudo virt-clone --original centos70 --auto-clone -n centos70-gpu
virsh start centos70-gpu
ssh `vm_ip centos70-gpu`
virsh edit centos70-gpu # add to <devices>, see above
virsh start centos70-gpu
ssh `vm_ip centos70-gpu`
lspci | grep -i nvidia
00:08.0 VGA compatible controller: NVIDIA Corporation GP106GL [Quadro P2000] (rev a1)
```

## Setup development environment in VM instances

SSH to VM and run

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
git config --global user.email "<YOUR EMAIL AT GITHUB>"
git config --global user.name "<YOUR NAME>"
```
