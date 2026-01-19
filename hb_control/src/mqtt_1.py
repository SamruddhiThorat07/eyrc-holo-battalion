#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from hb_interfaces.msg import BotCmdArray
import paho.mqtt.client as mqtt
from std_msgs.msg import Bool
from hb_interfaces.msg import BotIrState

class MQTTBridge(Node):
    def __init__(self):
        super().__init__('mqtt_bridge')
        
        self.broker_ip = "10.21.92.148"
        self.max_vel = 90.0
        
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        self.ir_publishers = {}
        
        try:
            self.mqtt_client.connect(self.broker_ip, 1883, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            self.get_logger().error(f"MQTT connection failed: {e}")
        
        # ROS2 Sub/Pub
        self.cmd_sub = self.create_subscription(BotCmdArray, '/bot_cmd', self.cmd_callback, 10)
        self.ir_pub = self.create_publisher(BotIrState, '/ir_sensor_state', 10)
                
        self.get_logger().info("MQTT Bridge initialized with Dynamic ID Support")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            # Subscribe to ALL bot IR sensors using wildcard
            self.mqtt_client.subscribe("esp/sensor/ir/+")
            self.get_logger().info("Connected and Subscribed to esp/sensor/ir/+")

    def on_mqtt_message(self, client, userdata, msg):
        if "esp/sensor/ir/" in msg.topic:
            try:
                bot_id = int(msg.topic.split('/')[-1]) # Convert ID to int
                ir_state = bool(int(msg.payload.decode()))
                
                ros_msg = BotIrState()
                ros_msg.id = bot_id
                ros_msg.state = ir_state
                self.ir_pub.publish(ros_msg)
                
            except Exception as e:
                self.get_logger().error(f"IR Parse Error: {e}")

    def cmd_callback(self, msg):
        for cmd in msg.cmds:
            # Clamp and format
            m1 = max(-self.max_vel, min(self.max_vel, cmd.m1))
            m2 = max(-self.max_vel, min(self.max_vel, cmd.m2))
            m3 = max(-self.max_vel, min(self.max_vel, cmd.m3))
            
            payload = f"{cmd.id},{m1:.2f},{m2:.2f},{m3:.2f},{cmd.base:.2f},{cmd.elbow:.2f}"
            topic = f"esp/cmd/{cmd.id}"
            
            self.mqtt_client.publish(topic, payload, qos=1)

def main(args=None):
    rclpy.init(args=args)
    bridge = MQTTBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    bridge.mqtt_client.disconnect()
    bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()