#!/bin/bash
# Must be run as root on the Ubuntu VM
echo "Starting VM Setup for Bank Server..."

# 1. Update and install dependencies
apt update
apt install -y python3-pip python3-venv nginx postgresql postgresql-contrib git fail2ban certbot python3-certbot-nginx

# 2. Setup PostgreSQL
echo "Configuring PostgreSQL..."
sudo -u postgres psql -c "CREATE DATABASE bankdb;"
sudo -u postgres psql -c "CREATE USER bankuser WITH PASSWORD 'bankpass';"
sudo -u postgres psql -c "ALTER ROLE bankuser SET client_encoding TO 'utf8';"
sudo -u postgres psql -c "ALTER ROLE bankuser SET default_transaction_isolation TO 'read committed';"
sudo -u postgres psql -c "ALTER ROLE bankuser SET timezone TO 'UTC';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE bankdb TO bankuser;"

# 3. Setup Application Directory
APP_DIR="/var/www/bank_server"
mkdir -p $APP_DIR
# Assuming code is cloned or copied here. If copying from local, use SCP beforehand.
# cd $APP_DIR

# 4. Setup VirtualEnv and Install
python3 -m venv $APP_DIR/venv
source $APP_DIR/venv/bin/activate
# pip install -r $APP_DIR/requirements.txt
# export DATABASE_URL="postgresql://bankuser:bankpass@localhost/bankdb"
# python $APP_DIR/bootstrap.py

# 5. Configure Systemd Service for Gunicorn
cat <<EOF > /etc/systemd/system/bank_server.service
[Unit]
Description=Gunicorn instance to serve Bank Server
After=network.target

[Service]
User=root
Group=www-data
WorkingDirectory=$APP_DIR
Environment="DATABASE_URL=postgresql://bankuser:bankpass@localhost/bankdb"
Environment="SECRET_KEY=prod-secret-key-change-me"
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/gunicorn --workers 3 --bind unix:$APP_DIR/bank_server.sock -m 007 app:app

[Install]
WantedBy=multi-user.target
EOF

systemctl start bank_server
systemctl enable bank_server

# 6. Configure Nginx
cat <<EOF > /etc/nginx/sites-available/bank_server
server {
    listen 80;
    server_name _;

    location / {
        include proxy_params;
        proxy_pass http://unix:$APP_DIR/bank_server.sock;
    }
}
EOF

ln -s /etc/nginx/sites-available/bank_server /etc/nginx/sites-enabled
rm /etc/nginx/sites-enabled/default
systemctl restart nginx

echo "VM Setup Complete! The server should be running on Port 80."
