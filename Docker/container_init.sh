#modify mavros
sed -i 's|/dev/ttyACM0:57600|/dev/ttyTHS0:921600|1' /opt/ros/noetic/share/mavros/launch/px4.launch

#LCM Config
ifconfig wlan1 multicast
route add -net 224.0.0.0 netmask 240.0.0.0 dev wlan1

#start service
service ssh start
/daemon/nvargus-daemon &

#login as root
su - root
