#!/bin/python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Int16
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time
import math

import numpy as np


class WaypointPIDController(Node):
    def __init__(self):
        super().__init__('waypoint_traffic_controller')

        # Publishers
        self.pub_L = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_R = self.create_publisher(Float32, '/VelocitySetR', 10)

        # Subscribers
        qos_encoders = QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_L = self.create_subscription(Float32,'/VelocityEncL',self.cb_L,qos_encoders)
        self.sub_R = self.create_subscription(Float32,'/VelocityEncR',self.cb_R,qos_encoders)
        self.sub_traffic_light = self.create_subscription(String, '/Traffic_light', self.cb_color, 10) 
        self.sub_center_error = self.create_subscription(Float32, '/line_detector_error', self.cb_line, 10)  # typo corregido

        # Robot params
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_base', 0.19)
        self.declare_parameter('invert_left',  True)   # ajusta según tu hardware
        self.declare_parameter('invert_right', True)  # ajusta según tu hardware

        self.declare_parameter('v_max', 0.5)
        self.declare_parameter('follow_speed', 0.1)
        self.declare_parameter('omega_max', 2.5)
        self.v_max = self.get_parameter('v_max').value
        self.w_max = self.get_parameter('omega_max').value

        # PID freq
        self.declare_parameter('frequency', 100) # In Hz
        self.freq = self.get_parameter('frequency').value
        self.create_timer(1/self.freq, self.update)

        # PID angular
        self.declare_parameter('kp_ang', 0.8)
        self.declare_parameter('ki_ang', 0.0)
        self.declare_parameter('kd_ang', 0.6)

        # PID lineal
        self.declare_parameter('kp_dist', 0.65)
        self.declare_parameter('ki_dist', 0.05)  # subido de 0.01
        self.declare_parameter('kd_dist', 0.0)

        # Tolerancia
        self.declare_parameter('pos_tol', 0.05)

        # Velocity
        self.vel_L = 0.0
        self.vel_R = 0.0

        # Odometry
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.last_odom_time = time.time()

        # Traffic Light
        self.traffic_light = "none"

        # Line Follower
        self.line_error = 0.0

        # Ramp limiter
        self.prev_v_cmd = 0.0
        self.prev_w_cmd = 0.0

        self.was_red = False

        # Lineal memoria
        self.int_dist = 0.0
        self.prev_dist_error = 0.0

        # Angular memoria
        self.int_ang = 0.0
        self.prev_ang_error = 0.0
        self.last_pid_time = time.time()

        self.get_logger().info("Line follow PID Controller listo")

    def cb_L(self, msg):
        self.vel_L = msg.data

    def cb_R(self, msg):
        self.vel_R = msg.data

    def cb_color(self, msg):
        self.traffic_light = msg.data

    def cb_line(self, msg):
        self.line_error = msg.data

    def update_odometry(self):
        now = time.time()
        dt = now - self.last_odom_time
        self.last_odom_time = now

        if dt <= 0:
            return

        r        = self.get_parameter('wheel_radius').value
        b        = self.get_parameter('wheel_base').value
        inv_L    = self.get_parameter('invert_left').value
        inv_R    = self.get_parameter('invert_right').value

        wL = -self.vel_L if inv_L else self.vel_L
        wR = -self.vel_R if inv_R else self.vel_R

        vL = wL * r
        vR = wR * r

        v = (vR + vL) / 2.0
        w = (vR - vL) / b

        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.v = v
        self.theta += w * dt
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def send_velocity(self, v, w):
        r     = self.get_parameter('wheel_radius').value
        b     = self.get_parameter('wheel_base').value
        inv_L = self.get_parameter('invert_left').value
        inv_R = self.get_parameter('invert_right').value

        # Traffic light logic
        if self.traffic_light == "green":
            self.was_red = False
        elif self.traffic_light == "red" or self.was_red:
            v = 0.0
            w = 0.0
            self.was_red = True
        elif self.traffic_light == "yellow":
            v *= 1.1
            w *= 1.1

        # Ramp limiter
        max_accel = 0.2
        max_alpha = 4.0
        dt_loop   =  1/self.freq

        dv = max_accel * dt_loop
        dw = max_alpha * dt_loop

        v = max(self.prev_v_cmd - dv, min(v, self.prev_v_cmd + dv))
        w = max(self.prev_w_cmd - dw, min(w, self.prev_w_cmd + dw))

        self.prev_v_cmd = v
        self.prev_w_cmd = w

        # Diferencial
        v_left  = v - w * b / 2.0
        v_right = v + w * b / 2.0

        set_L = v_left  / r
        set_R = v_right / r

        # Inversión individual por rueda
        if inv_L:
            set_L = -set_L
        if inv_R:
            set_R = -set_R

        self.pub_L.publish(Float32(data=float(set_L)))
        self.pub_R.publish(Float32(data=float(set_R)))

    def update(self):
        self.update_odometry()

        e = float(self.line_error)

        now = time.time()
        dt  = now - self.last_pid_time
        self.last_pid_time = now

        if dt <= 0.0:
            return

        kp_a = self.get_parameter('kp_ang').value
        ki_a = self.get_parameter('ki_ang').value
        kd_a = self.get_parameter('kd_ang').value

        self.get_logger().info(f"{kp_a}")

        self.int_ang += e * dt
        int_limit     = self.w_max / (ki_a + 1e-9)
        self.int_ang  = max(-int_limit, min(self.int_ang, int_limit))

        de_ang = (e - self.prev_ang_error) / dt
        self.prev_ang_error = e

        omega = kp_a*e + ki_a*self.int_ang + kd_a*de_ang
        self.get_logger().info(f"{omega}")
        omega = max(-self.w_max, min(omega, self.w_max))


        v_base = self.get_parameter('follow_speed').value

        if abs(omega) > 0.15:
            v_base -= 0.05 * abs(omega)
            self.get_logger().info("frenanding...")

        self.send_velocity(v_base, omega)

        self.get_logger().info(
            f"e={e:+.3f}  v={v_base:.3f}  ω={omega}  "
            f"x={self.x:.2f}  y={self.y:.2f}  θ={math.degrees(self.theta):.1f}°"
        )

    def destroy_node(self):
        target = time.time() + 1
        while time.time() < target:
            rclpy.spin_once(self)
            self.send_velocity(0.0, 0.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = WaypointPIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception as e:
            print(e)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
