# run me as sudo bash ...
#
# Author: Pearu Peterson
# Created: May 2018

NAME=centos70
OSVARIANT=centos7.0
LOCATION=http://ftp.estpak.ee/pub/centos/7/os/x86_64/
CMD="virt-install \
    --name $NAME \
    --ram 6144 \
    --disk path=/servers/images/$NAME-amd64.img,size=50 \
    --vcpus 4 \
    --os-type linux \
    --os-variant $OSVARIANT \
    --network bridge=virbr0 \
    --nographics \
    --location $LOCATION \
    --extra-args='console=ttyS0' \
    --noautoconsole --console pty,target_type=serial" 
echo $CMD
$CMD

echo "Additional instructions:"
echo "  run: virsh console $NAME"
echo "    choose tekst based installation method and fill out all required fields"
echo "  when installation is complete, run: virsh start $NAME"
echo "  determine the IP and do `ssh-copy-id $IP` and `ssh IP`"
echo "  run: yum update"
