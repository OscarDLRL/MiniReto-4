#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time
import math


class WaypointPIDController(Node):
    def __init__(self):
        super().__init__('waypoint_pid_controller')

        # Publishers
        self.pub_L = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_R = self.create_publisher(Float32, '/VelocitySetR', 10)

        # Subscribers
        self.sub_L = self.create_subscription(Float32, '/VelocityEncL', self.cb_L, 10)
        self.sub_R = self.create_subscription(Float32, '/VelocityEncR', self.cb_R, 10)

        # PID parameters
        self.declare_parameter('kp', 2.4)
        self.declare_parameter('ki', 0.03)
        self.declare_parameter('kd', 0.001)

        # Robot params
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_base', 0.19)
        self.declare_parameter('invert_wheels', True)

        # Velocities
        self.vel_L = 0.0
        self.vel_R = 0.0

        self.err_L_prev = 0.0
        self.err_R_prev = 0.0
        self.int_L = 0.0
        self.int_R = 0.0

        self.last_time = time.time()

        self.get_logger().info("Waypoint PID Controller listo")

    def cb_L(self, msg):
        self.vel_L = msg.data

    def cb_R(self, msg):
        self.vel_R = msg.data

    def compute_pid(self, set_L, set_R):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value

        err_L = set_L - self.vel_L
        err_R = set_R - self.vel_R

        self.int_L += err_L * dt
        self.int_R += err_R * dt

        der_L = (err_L - self.err_L_prev) / dt if dt > 0 else 0
        der_R = (err_R - self.err_R_prev) / dt if dt > 0 else 0

        out_L = kp * err_L + ki * self.int_L + kd * der_L
        out_R = kp * err_R + ki * self.int_R + kd * der_R

        self.err_L_prev = err_L
        self.err_R_prev = err_R

        return out_L, out_R

    def send_body_velocity(self, v, w, duration):
        r = self.get_parameter('wheel_radius').value
        b = self.get_parameter('wheel_base').value
        invert = self.get_parameter('invert_wheels').value

        end_time = time.time() + duration

        while time.time() < end_time:
            v_left = v - (w * b / 2.0)
            v_right = v + (w * b / 2.0)

            set_L = v_left / r
            set_R = v_right / r

            if invert:
                set_L = -set_L
                set_R = -set_R

            u_L, u_R = self.compute_pid(set_L, set_R)

            self.pub_L.publish(Float32(data=float(u_L)))
            self.pub_R.publish(Float32(data=float(u_R)))

            time.sleep(0.02)

        self.stop()

    def stop(self):
        self.pub_L.publish(Float32(data=0.0))
        self.pub_R.publish(Float32(data=0.0))
        time.sleep(0.3)

    def move_straight(self, distance_m=0.4, v=0.3):
        t = distance_m / v
        self.get_logger().info(f"Avanzando {distance_m} m")
        self.send_body_velocity(v, 0.0, t)

    def turn(self, angle_rad, w=1.2):
        t = abs(angle_rad) / w
        direction = 1.0 if angle_rad >= 0 else -1.0

        self.get_logger().info(f"Girando {math.degrees(angle_rad)}°")
        self.send_body_velocity(0.0, direction * w, t)

    def run(self):
        self.get_logger().info("Iniciando trayectoria de waypoints")

        # W1
        self.move_straight(0.4)

        # Turn +45°
        self.turn(math.pi / 4)

        # W2
        self.move_straight(0.4)

        # Turn -45°
        self.turn(-math.pi / 4)

        # W3
        self.move_straight(0.4)

        self.stop()


def main():
    rclpy.init()
    node = WaypointPIDController()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()