#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from hb_interfaces.msg import BotCmdArray
import paho.mqtt.client as mqtt
from std_msgs.msg import Bool

class MQTTBridge(Node):
    def __init__(self):
        super().__init__('mqtt_bridge')
        
        # MQTT Setup - UPDATED IP
        self.broker_ip = "10.21.92.148"  # Updated to match ESP32's network
        self.max_vel = 75.0
        
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        try:
            self.get_logger().info(f"Connecting to MQTT broker at {self.broker_ip}:1883")
            self.mqtt_client.connect(self.broker_ip, 1883, 60)
            self.mqtt_client.loop_start()
            self.get_logger().info("MQTT client loop started")
        except Exception as e:
            self.get_logger().error(f"MQTT connection failed: {e}")
        
        # ROS2 Subscription
        self.cmd_sub = self.create_subscription(
            BotCmdArray,
            '/bot_cmd',
            self.cmd_callback,
            10
        )
        
        self.ir_pub = self.create_publisher(Bool, '/ir_sensor_state', 10)
        self.mqtt_client.subscribe("esp/sensor/ir")
                
        self.get_logger().info("MQTT Bridge initialized")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.get_logger().info("Connected to MQTT broker")
        else:
            self.get_logger().error(f"Connection failed: {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        # self.get_logger().info(f"[{msg.topic}] {msg.payload.decode()}")
        if msg.topic == "esp/sensor/ir":
            try:
                # Parse the MQTT payload ("1" or "0")
                ir_state = bool(int(msg.payload.decode()))
                
                # Create the ROS2 message
                ros_msg = Bool()
                ros_msg.data = ir_state
                
                # Publish to the ROS2 network
                self.ir_pub.publish(ros_msg)
                
                # Log for debugging
                state_text = "OBJECT DETECTED" if ir_state else "CLEAR"
                # self.get_logger().info(f"IR Status: {state_text}")
            except Exception as e:
                self.get_logger().error(f"Failed to parse IR MQTT: {e}")
    
    def cmd_callback(self, msg):
        for cmd in msg.cmds:
            if cmd.id == 0 or cmd.id == 2 or cmd.id == 4:
                # Clamp to max_vel
                m1 = max(-self.max_vel, min(self.max_vel, cmd.m1))
                m2 = max(-self.max_vel, min(self.max_vel, cmd.m2))
                m3 = max(-self.max_vel, min(self.max_vel, cmd.m3))
                arm = cmd.base
                solenoid = cmd.elbow
                
                # Create payload
                mqtt_payload = f"{m1:.2f},{m2:.2f},{m3:.2f},{arm:.2f},{solenoid:.2f}"
                
                # Publish
                result = self.mqtt_client.publish("esp/cmd/0", mqtt_payload, qos=1)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self.get_logger().info(f"→ {mqtt_payload}")

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




# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from hb_interfaces.msg import BotCmdArray
# import paho.mqtt.client as mqtt
# from std_msgs.msg import Bool

# class MQTTBridge(Node):
#     def __init__(self):
#         super().__init__('mqtt_bridge')
        
#         # MQTT Setup - UPDATED IP
#         self.broker_ip = "10.21.92.148"  # Updated to match ESP32's network
#         self.max_vel = 90.0
        
#         self.mqtt_client = mqtt.Client()
#         self.mqtt_client.on_connect = self.on_mqtt_connect
#         self.mqtt_client.on_message = self.on_mqtt_message
        
#         try:
#             self.get_logger().info(f"Connecting to MQTT broker at {self.broker_ip}:1883")
#             self.mqtt_client.connect(self.broker_ip, 1883, 60)
#             self.mqtt_client.loop_start()
#             self.get_logger().info("MQTT client loop started")
#         except Exception as e:
#             self.get_logger().error(f"MQTT connection failed: {e}")
        
#         # ROS2 Subscription
#         self.cmd_sub = self.create_subscription(
#             BotCmdArray,
#             '/bot_cmd',
#             self.cmd_callback,
#             10
#         )
        
#         self.ir_pub = self.create_publisher(Bool, '/ir_sensor_state', 10)
#         self.mqtt_client.subscribe("esp/sensor/ir")
                
#         self.get_logger().info("MQTT Bridge initialized")
    
#     def on_mqtt_connect(self, client, userdata, flags, rc):
#         if rc == 0:
#             self.get_logger().info("Connected to MQTT broker")
#         else:
#             self.get_logger().error(f"Connection failed: {rc}")
    
#     def on_mqtt_message(self, client, userdata, msg):
#         # self.get_logger().info(f"[{msg.topic}] {msg.payload.decode()}")
#         if msg.topic == "esp/sensor/ir":
#             try:
#                 # Parse the MQTT payload ("1" or "0")
#                 ir_state = bool(int(msg.payload.decode()))
                
#                 # Create the ROS2 message
#                 ros_msg = Bool()
#                 ros_msg.data = ir_state
                
#                 # Publish to the ROS2 network
#                 self.ir_pub.publish(ros_msg)
                
#                 # Log for debugging
#                 state_text = "OBJECT DETECTED" if ir_state else "CLEAR"
#                 # self.get_logger().info(f"IR Status: {state_text}")
#             except Exception as e:
#                 self.get_logger().error(f"Failed to parse IR MQTT: {e}")
    
#     def cmd_callback(self, msg):
#         for cmd in msg.cmds:
#             if cmd.id in [0, 2, 4]:
#                 # Clamp to max_vel
#                 m1 = max(-self.max_vel, min(self.max_vel, cmd.m1))
#                 m2 = max(-self.max_vel, min(self.max_vel, cmd.m2))
#                 m3 = max(-self.max_vel, min(self.max_vel, cmd.m3))
#                 arm = cmd.base
#                 solenoid = cmd.elbow
                
#                 # NEW PAYLOAD: Includes Bot ID as the first value
#                 mqtt_payload = f"{cmd.id},{m1:.2f},{m2:.2f},{m3:.2f},{arm:.2f},{solenoid:.2f}"
                
#                 # DYNAMIC TOPIC: esp/cmd/0, esp/cmd/2, etc.
#                 topic = f"esp/cmd/{cmd.id}"
                
#                 # Publish
#                 result = self.mqtt_client.publish(topic, mqtt_payload, qos=1)
                
#                 if result.rc == mqtt.MQTT_ERR_SUCCESS:
#                     self.get_logger().info(f"[{topic}] → {mqtt_payload}")
                    
# def main(args=None):
#     rclpy.init(args=args)
#     bridge = MQTTBridge()
    
#     try:
#         rclpy.spin(bridge)
#     except KeyboardInterrupt:
#         pass
    
#     bridge.mqtt_client.disconnect()
#     bridge.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

