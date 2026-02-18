#!/bin/bash
# Update AWS SSH config when IP changes

set -e  # Exit on error

# Check if IP provided
if [ -z "$1" ]; then
    echo "Usage: ./update_aws_ip.sh <new-ip-or-hostname>"
    echo ""
    echo "Examples:"
    echo "  ./update_aws_ip.sh 54.224.93.100"
    echo "  ./update_aws_ip.sh ec2-54-224-93-100.compute-1.amazonaws.com"
    exit 1
fi

NEW_IP="$1"
KEY_PATH="/Users/tianqinmeng/Desktop/Projects/FastText-trainer-finewebedu-climate/WXChatKey.pem"

# Extract just IP if hostname provided
if [[ "$NEW_IP" == ec2-* ]]; then
    HOSTNAME="$NEW_IP"
    # Extract IP from hostname (e.g., ec2-54-224-93-100.compute-1.amazonaws.com -> 54.224.93.100)
    IP=$(echo "$HOSTNAME" | sed 's/ec2-\(.*\)\.compute.*/\1/' | tr '-' '.')
else
    IP="$NEW_IP"
    # Build hostname from IP (e.g., 54.224.93.100 -> ec2-54-224-93-100.compute-1.amazonaws.com)
    HOSTNAME="ec2-$(echo $IP | tr '.' '-').compute-1.amazonaws.com"
fi

echo "========================================="
echo "AWS IP Update Script"
echo "========================================="
echo "New IP: $IP"
echo "Hostname: $HOSTNAME"
echo ""

# Step 1: Remove old SSH keys
echo "[1/3] Removing old SSH keys from known_hosts..."
ssh-keygen -R "$HOSTNAME" 2>/dev/null || true
ssh-keygen -R "$IP" 2>/dev/null || true
# Also remove some common old IPs
ssh-keygen -R "ec2-52-90-123-209.compute-1.amazonaws.com" 2>/dev/null || true
ssh-keygen -R "ec2-98-85-228-60.compute-1.amazonaws.com" 2>/dev/null || true
echo "✓ Old keys removed"
echo ""

# Step 2: Update SSH config
echo "[2/3] Updating SSH config..."
cat > ~/.ssh/config << EOF
Host cs136
  HostName ubuntu2204-006.student.cs.uwaterloo.ca
  User s52meng
  IdentityFile ~/.ssh/id_rsa

Host aws
  HostName $HOSTNAME
  User ec2-user
  IdentityFile $KEY_PATH

Host aws-fasttext
  HostName $HOSTNAME
  User ec2-user
  IdentityFile $KEY_PATH
EOF
echo "✓ SSH config updated"
echo ""

# Step 3: Test connection
echo "[3/3] Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new aws "echo '✓ Connection successful!'" 2>/dev/null; then
    echo ""
    echo "========================================="
    echo "✓ All done! SSH is ready."
    echo "========================================="
    echo ""
    echo "You can now:"
    echo "  - Connect via terminal: ssh aws"
    echo "  - Connect via Cursor: use 'aws-fasttext' host"
    echo "  - Download files: scp aws:~/path/file ."
    echo "  - Upload files: scp file aws:~/path/"
    echo ""
else
    echo ""
    echo "⚠ Warning: Could not connect to AWS instance."
    echo "   Make sure the instance is running and the IP is correct."
    echo "   SSH config has been updated anyway."
    echo ""
fi
