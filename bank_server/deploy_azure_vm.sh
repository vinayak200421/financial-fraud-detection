#!/bin/bash
# Local script to provision an Azure VM and open Port 80

RESOURCE_GROUP="BankServerRG"
VM_NAME="BankServerVM"
LOCATION="eastus"
IMAGE="Ubuntu2204"
ADMIN_USER="azureuser"

echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "Creating Virtual Machine (this may take a few minutes)..."
az vm create \
  --resource-group $RESOURCE_GROUP \
  --name $VM_NAME \
  --image $IMAGE \
  --admin-username $ADMIN_USER \
  --generate-ssh-keys \
  --public-ip-sku Standard

echo "Opening Port 80 for Nginx..."
az vm open-port --port 80 --resource-group $RESOURCE_GROUP --name $VM_NAME

echo "Retrieving Public IP Address..."
PUBLIC_IP=$(az vm show -d -g $RESOURCE_GROUP -n $VM_NAME --query publicIps -o tsv)

echo "=========================================="
echo "Deployment Complete!"
echo "Your VM Public IP is: $PUBLIC_IP"
echo "Next Steps:"
echo "1. SCP the bank_server code to the VM: scp -r ../bank_server $ADMIN_USER@$PUBLIC_IP:~/"
echo "2. SSH into the VM: ssh $ADMIN_USER@$PUBLIC_IP"
echo "3. Run the vm_setup.sh script as root (sudo bash vm_setup.sh)"
echo "=========================================="
