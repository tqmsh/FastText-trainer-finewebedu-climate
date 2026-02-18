#!/bin/bash
# Update AWS SSH config when IP changes (WSL2 + Windows compatible)

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
WSL_KEY_PATH="/home/tqmen/workplace/projects/FastText-trainer-finewebedu-climate/WXChatKey.pem"
WIN_SSH_DIR="/mnt/c/Users/tqmen/.ssh"
WIN_KEY_PATH="$WIN_SSH_DIR/WXChatKey.pem"

# Extract just IP if hostname provided
if [[ "$NEW_IP" == ec2-* ]]; then
    HOSTNAME="$NEW_IP"
    IP=$(echo "$HOSTNAME" | sed 's/ec2-\(.*\)\.compute.*/\1/' | tr '-' '.')
else
    IP="$NEW_IP"
    HOSTNAME="ec2-$(echo $IP | tr '.' '-').compute-1.amazonaws.com"
fi

echo "========================================="
echo "AWS IP Update Script (WSL2 + Windows)"
echo "========================================="
echo "New IP: $IP"
echo "Hostname: $HOSTNAME"
echo ""

# Step 1: Fix key permissions + clear old known_hosts entries
echo "[1/4] Fixing key permissions and clearing stale known_hosts..."
chmod 600 "$WSL_KEY_PATH"
ssh-keygen -R "$HOSTNAME" 2>/dev/null || true
ssh-keygen -R "$IP" 2>/dev/null || true
ssh-keygen -R "ec2-52-90-123-209.compute-1.amazonaws.com" 2>/dev/null || true
ssh-keygen -R "ec2-98-85-228-60.compute-1.amazonaws.com" 2>/dev/null || true
echo "✓ Done"
echo ""

# Step 2: Update WSL SSH config (~/.ssh/config)
echo "[2/4] Updating WSL SSH config..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat > ~/.ssh/config << EOF
Host cs136
  HostName ubuntu2204-006.student.cs.uwaterloo.ca
  User s52meng
  IdentityFile ~/.ssh/id_rsa

Host aws
  HostName $HOSTNAME
  User ec2-user
  IdentityFile $WSL_KEY_PATH

Host aws-fasttext
  HostName $HOSTNAME
  User ec2-user
  IdentityFile $WSL_KEY_PATH
EOF
chmod 600 ~/.ssh/config
echo "✓ WSL SSH config updated"
echo ""

# Step 3: Copy key + update Windows SSH config (C:\Users\tqmen\.ssh\config)
echo "[3/4] Updating Windows SSH config..."
if [ -d "$WIN_SSH_DIR" ]; then
    cp "$WSL_KEY_PATH" "$WIN_KEY_PATH"
    chmod 600 "$WIN_KEY_PATH"
    cat > "$WIN_SSH_DIR/config" << EOF
Host cs136
  HostName ubuntu2204-006.student.cs.uwaterloo.ca
  User s52meng
  IdentityFile ~/.ssh/id_rsa

Host aws
  HostName $HOSTNAME
  User ec2-user
  IdentityFile ~/.ssh/WXChatKey.pem

Host aws-fasttext
  HostName $HOSTNAME
  User ec2-user
  IdentityFile ~/.ssh/WXChatKey.pem
EOF
    echo "✓ Windows SSH config updated (C:\\Users\\tqmen\\.ssh\\config)"
    echo "  Key copied to C:\\Users\\tqmen\\.ssh\\WXChatKey.pem"
else
    echo "⚠ Windows .ssh dir not found at $WIN_SSH_DIR — skipping Windows config"
fi
echo ""

# Step 4: Test connection — must succeed for script to pass
echo "[4/4] Testing SSH connection to $IP..."
echo "      (will retry up to 3 times with 5s timeout each)"
echo ""

CONNECTED=0
for attempt in 1 2 3; do
    echo "  Attempt $attempt/3..."
    if ssh -o ConnectTimeout=5 \
           -o StrictHostKeyChecking=accept-new \
           -o BatchMode=yes \
           -i "$WSL_KEY_PATH" \
           "ec2-user@$HOSTNAME" "echo OK" 2>/tmp/ssh_err; then
        CONNECTED=1
        break
    else
        echo "  Failed: $(cat /tmp/ssh_err | tail -1)"
        [ $attempt -lt 3 ] && sleep 3
    fi
done

echo ""
if [ $CONNECTED -eq 1 ]; then
    echo "========================================="
    echo "✓ All done! SSH is ready."
    echo "========================================="
    echo ""
    echo "  WSL terminal:    ssh aws"
    echo "  Cursor/VSCode:   host 'aws-fasttext'"
    echo "  Copy from EC2:   scp aws:~/file ."
    echo "  Copy to EC2:     scp file aws:~/"
    echo ""
else
    echo "========================================="
    echo "✗ Connection FAILED after 3 attempts."
    echo "========================================="
    echo ""
    echo "  Last SSH error: $(cat /tmp/ssh_err)"
    echo ""
    echo "  Check:"
    echo "    1. Is the EC2 instance running?"
    echo "    2. Is the IP correct? ($IP)"
    echo "    3. Does the security group allow port 22 from your IP?"
    echo ""
    exit 1
fi
