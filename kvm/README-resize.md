# Increasing the disk size of virtual machine

```
# Login to virtual image and shut it down:
sudo shutdown -h now

# In KVM server:
# Set the name of VM that disk is to be resized
export VM=ubuntu1804-gpu-conda37      # edit

# Find the VM image:
virsh domblklist $VM
Target     Source
------------------------------------------------
vda        /servers/images/ubuntu1604-amd64-clone-4-clone-2-clone.img

# Set the name of VM image file:
export VMFILE=/servers/images/ubuntu1604-amd64-clone-4-clone-2-clone.img

# Establish the current size:
sudo qemu-img info $VMFILE

# Resize the disk (add 50GB):
sudo qemu-img resize $VMFILE +50G

# List file systems:
sudo virt-filesystems --all --long -h  -d $VM

# Make a copy
sudo cp $VMFILE $VMFILE-orig

# Expand the file system
sudo virt-resize –expand /dev/sda1 $VMFILE-orig $VMFILE

# Check the results:
sudo qemu-img info $VMFILE
sudo virt-filesystems --all --long -h  -d $VM

# Clean up
rm $VMFILE-orig

# Start the VM and check filesystem sizes
virsh start $VM
ssh `vm_ip $VM`
df -h
logout

# Clean up
rm $VMFILE-orig
```
