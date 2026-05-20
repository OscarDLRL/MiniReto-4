#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
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


        # Robot params
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_base', 0.19)
        self.declare_parameter('invert_wheels', True)

        self.declare_parameter('v_max',            1.5)
        self.declare_parameter('omega_max',        2.5)
        self.v_max   = self.get_parameter('v_max').value
        self.w_max   = self.get_parameter('omega_max').value

        # PID angular
        self.declare_parameter('kp_ang', 1.8)
        self.declare_parameter('ki_ang', 0.0)
        self.declare_parameter('kd_ang', 0.0)

        # PID lineal
        self.declare_parameter('kp_dist', 0.65)
        self.declare_parameter('ki_dist', 0.01)
        self.declare_parameter('kd_dist', 0.0)

        # Tolerancia
        self.declare_parameter('pos_tol', 0.05)


        #Velocity
        self.vel_L = 0.0
        self.vel_R = 0.0

        #Odometry
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_odom_time = time.time()

        # Traffic Light
        self.traffic_light = "none"

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

        self.get_logger().info("Waypoint PID Controller listo")

    def cb_L(self, msg):
        self.vel_L = msg.data

    def cb_R(self, msg):
        self.vel_R = msg.data
    
    def cb_color(self, msg):
        self.traffic_light = msg.data

    
    def update_odometry(self):
        now = time.time()
        dt = now - self.last_odom_time
        self.last_odom_time = now

        if dt <= 0:
            return

        r = self.get_parameter('wheel_radius').value
        b = self.get_parameter('wheel_base').value
        invert = self.get_parameter('invert_wheels').value

        wL = self.vel_L
        wR = self.vel_R

        if invert:
            wL = -wL
            wR = -wR

        vL = wL * r
        vR = wR * r

        v = (vR + vL) / 2.0
        w = (vR - vL) / b

        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt

        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
    


    def send_velocity(self, v, w):
        r = self.get_parameter('wheel_radius').value
        b = self.get_parameter('wheel_base').value
        invert = self.get_parameter('invert_wheels').value


        if self.traffic_light == "green":
            self.was_red = False

        elif self.traffic_light == "red" or self.was_red:
            v = 0.0
            w = 0.0
            self.was_red = True


        elif self.traffic_light == "yellow":
            v *= 1.1
            w *= 1.1

        max_accel = 0.2
        max_alpha = 4.0
        dt_loop = 0.02

        dv = max_accel * dt_loop
        dw = max_alpha * dt_loop

        v = max(self.prev_v_cmd - dv, min(v, self.prev_v_cmd + dv))
        w = max(self.prev_w_cmd - dw, min(w, self.prev_w_cmd + dw))

        self.prev_v_cmd = v
        self.prev_w_cmd = w

        v_left = v - w * b / 2.0
        v_right = v + w * b / 2.0

        set_L = v_left / r
        set_R = v_right / r

        if invert:
            set_L = -set_L
            set_R = -set_R


        self.pub_L.publish(Float32(data=float(set_L)))
        self.pub_R.publish(Float32(data=float(set_R)))
    
    def stop(self):
        self.prev_v_cmd = 0.0
        self.prev_w_cmd = 0.0
        self.pub_L.publish(Float32(data=0.0))
        self.pub_R.publish(Float32(data=0.0))
        time.sleep(0.3)
        
    def go_to_waypoint(self, goal_x, goal_y):
        rate_dt = 0.02
        #Reinicio de error
        self.int_dist = 0.0
        self.prev_dist_error = 0.0
        self.int_ang = 0.0
        self.prev_ang_error = 0.0
        self.last_pid_time = time.time()
        # PID lineal
        kp_dist = self.get_parameter('kp_dist').value
        ki_dist = self.get_parameter('ki_dist').value
        kd_dist = self.get_parameter('kd_dist').value
        # PID angular
        kp_ang = self.get_parameter('kp_ang').value
        ki_ang = self.get_parameter('ki_ang').value
        kd_ang = self.get_parameter('kd_ang').value

        pos_tol = self.get_parameter('pos_tol').value
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            self.update_odometry()

            dx = goal_x - self.x
            dy = goal_y - self.y

            distance_error = math.sqrt(dx**2 + dy**2)
            desired_theta = math.atan2(dy, dx)

            angle_error = desired_theta - self.theta
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

            if distance_error < pos_tol:
                break

            now = time.time()
            dt = now - self.last_pid_time
            self.last_pid_time = now

            if dt <= 0:
                dt = rate_dt

            self.int_dist += distance_error * dt
            self.int_dist = np.clip(self.int_dist, -1.0, 1.0)
            der_dist = (distance_error - self.prev_dist_error) / dt

            self.int_ang += angle_error * dt
            self.int_ang = np.clip(self.int_ang, -1.0, 1.0)
            der_ang = (angle_error - self.prev_ang_error) / dt

            v_cmd = (
                kp_dist * distance_error
                + ki_dist * self.int_dist
                + kd_dist * der_dist
            )

            w_cmd = (
                kp_ang * angle_error
                + ki_ang * self.int_ang
                + kd_ang * der_ang
            )

            self.prev_dist_error = distance_error
            self.prev_ang_error = angle_error

            v_cmd = float(np.clip(v_cmd, -self.v_max, self.v_max))
            w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

            if abs(angle_error) > 0.35:
                v_cmd = 0.0
            
            print(
                f"goal=({goal_x:.2f},{goal_y:.2f}) "
                f"x={self.x:.2f} y={self.y:.2f} th={self.theta:.2f} "
                f"dist={distance_error:.2f} ang={angle_error:.2f}"
            )

            self.send_velocity(v_cmd, w_cmd)

            time.sleep(rate_dt)

        self.stop()

    def run(self):
        self.get_logger().info("Iniciando trayectoria por odometría")

        waypoints = [
            (0.4, 0.0),
            (0.6, 0.346),
            (0.4, 0.692),
            (0.0, 0.692),
            (-0.2, 0.346),
            (0.0, 0.0),
        ]

        for wx, wy in waypoints:
            self.go_to_waypoint(wx, wy)

        self.stop()

def main():
    rclpy.init()
    node = WaypointPIDController()
    """
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
    """
    node.run()
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


