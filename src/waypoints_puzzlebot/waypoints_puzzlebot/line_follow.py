import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Int16, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time
import math
from enum import Enum
from collections import Counter

import numpy as np

class STATES(Enum):
    LINE_FOLLOW = 1
    STOP = 2
    TURN_RIGHT = 3
    TURN_LEFT = 4
    STRAIGHT = 5
    GIVE_WAY = 6
    ADVANCE_TO_CROSS = 7

class PuzzlebotController(Node):
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
        self.sub_center_error = self.create_subscription(Float32, '/line_detector_error', self.cb_line, 10)
        self.sub_yolo_state = self.create_subscription(String,'/traffic_sign',self.cb_yolo_state,10)
        self.sub_cross_bool = self.create_subscription(Bool,'/crosswalk_bool',self.cb_cross_bool,10)
        self.sub_cross_ang = self.create_subscription(Float32,'/crosswalk_ang',self.cb_cross_ang,10)
        self.sub_stop_area = self.create_subscription(Float32,'/stop_area',self.cb_stop_area,10)

        # YOLO states
        self.state = STATES.LINE_FOLLOW
        self.yolo_state = "none"
        self.state_start_time = time.time()

        # Robot params
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_base', 0.19)
        self.declare_parameter('invert',  True)

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

        # Debug and other utilities
        self.declare_parameter('console_debug', False)
        self.console_debug = self.get_parameter('console_debug').value

        # Velocity
        self.vel_L = 0.0
        self.vel_R = 0.0

        # Odometry
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.last_odom_time = time.time()
        self.lastw = 0.0

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

        # Cruze peatonal
        self.cross_bool = False
        self.cross_ang = 0
        self.last_detected_sign = "none"
        self.advance_distance = 0.0
        self.advance_goal = 0.26
        self.aangle_goal = 0.0
        self.aangle_theta = 0.0 # RADIANS
        self.advance_finished_time = None
        self.cross_ang_tolerance = 0.03
        self.cross_w = 0.0

        # Señal de alto 
        self.stop_area = 0.0
        self.stop_area_threshold = 26100.0

        # ── Work Ahead (flag, no state) ──────────────────────────────────────
        # En lugar de cambiar el estado, solo se activa este booleano.
        # Reduce la velocidad base mientras esté activo, sin interrumpir
        # ninguna otra maniobra (giros, cruce, etc.).
        self.work_ahead = False
        self.work_ahead_start = 0.0
        self.work_ahead_duration = 2.0   # segundos que dura el efecto
        self.work_ahead_speed = 0.06     # velocidad reducida mientras está activo
        # ─────────────────────────────────────────────────────────────────────

        # Counter 
        self.sign_counter = Counter()
        self.sign_counter_min_votes = 3

        self.get_logger().info("Line follow PID Controller listo")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def cb_L(self, msg):
        self.vel_L = msg.data

    def cb_R(self, msg):
        self.vel_R = msg.data

    def cb_color(self, msg):
        self.traffic_light = msg.data

    def cb_line(self, msg):
        self.line_error = msg.data
    
    def cb_stop_area(self, msg):
        self.stop_area = msg.data
    
    def cb_cross_ang(self, msg):
        self.cross_ang = msg.data

    def cb_cross_bool(self, msg):
        new_cross = msg.data

        # Detecta flanco de subida: antes False, ahora True.
        if new_cross and not self.cross_bool and self.state == STATES.LINE_FOLLOW:
            if self.console_debug:
                self.get_logger().info("Crosswalk detectado!")

            # Si había una señal guardada, avanza antes de ejecutar la acción.
            if self.last_detected_sign != "none":
                if len(self.sign_counter) > 0:
                    self.last_detected_sign = self.sign_counter.most_common(1)[0][0]

                if self.console_debug:
                    self.get_logger().info("EMPEZANDO A AVANZAR AL CROSSWALK")

                if self.last_detected_sign == "give_way":
                    self.state = STATES.GIVE_WAY
                else:
                    self.state = STATES.ADVANCE_TO_CROSS

                self.state_start_time = time.time()
                self.advance_distance = 0.0
                self.advance_finished_time = None
                self.aangle_goal = np.deg2rad(self.cross_ang)
                self.aangle_theta = 0.0
                self.cross_w = self.lastw
                if abs(self.cross_w) < 0.01 and self.aangle_goal > self.cross_ang_tolerance * 10:
                    self.cross_w = 0.25 * self.aangle_goal / abs(self.aangle_goal)

        self.cross_bool = new_cross

    def cb_yolo_state(self, msg):
        new_yolo_state = msg.data

        if new_yolo_state == "none":
            self.yolo_state = "none"
            return

        # STOP siempre se permite inmediatamente.
        if new_yolo_state == "stop":
            self.yolo_state = new_yolo_state
            self.last_detected_sign = new_yolo_state
            self.state = STATES.STOP
            self.state_start_time = time.time()
            return

        # ── WORK_AHEAD: solo activa el flag, no cambia el estado ─────────────
        # Así respeta cualquier maniobra que ya esté en curso.
        if new_yolo_state == "work_ahead":
            self.yolo_state = new_yolo_state
            self.work_ahead = True
            self.work_ahead_start = time.time()
            return
        # ─────────────────────────────────────────────────────────────────────

        # Cuenta señales vistas antes del crosswalk.
        self.sign_counter[new_yolo_state] += 1

        most_common_sign, votes = self.sign_counter.most_common(1)[0]
        if votes >= self.sign_counter_min_votes:
            self.last_detected_sign = most_common_sign

        self.yolo_state = new_yolo_state

        if not self.cross_bool:
            return

        if self.state != STATES.LINE_FOLLOW:
            return

        self.apply_saved_sign()

    # ── Odometry ─────────────────────────────────────────────────────────────

    def update_odometry(self):
        now = time.time()
        dt = now - self.last_odom_time
        self.last_odom_time = now

        if dt <= 0:
            return

        r   = self.get_parameter('wheel_radius').value
        b   = self.get_parameter('wheel_base').value
        inv = self.get_parameter('invert').value

        wL = -self.vel_L if inv else self.vel_L
        wR = -self.vel_R if inv else self.vel_R

        vL = wL * r
        vR = wR * r

        v = (vR + vL) / 2.0
        w = (vR - vL) / b

        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.v      = v
        self.theta += w * dt
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Solo usado en crosswalk
        self.advance_distance += v * dt
        self.aangle_theta     += w * dt

    # ── Send velocity ─────────────────────────────────────────────────────────

    def send_velocity(self, v, w):
        r   = self.get_parameter('wheel_radius').value
        b   = self.get_parameter('wheel_base').value
        inv = self.get_parameter('invert').value

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
        dt_loop   = 1 / self.freq

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

        if inv:
            set_L = -set_L
            set_R = -set_R

        self.pub_L.publish(Float32(data=float(set_L)))
        self.pub_R.publish(Float32(data=float(set_R)))

        self.lastw = w

    # ── Apply sign ───────────────────────────────────────────────────────────

    def apply_saved_sign(self):
        if self.console_debug:
            self.get_logger().info(f"APLICANDO {self.last_detected_sign}")

        if self.last_detected_sign == "turn_right":
            self.state = STATES.TURN_RIGHT
        elif self.last_detected_sign == "turn_left":
            self.state = STATES.TURN_LEFT
        elif self.last_detected_sign == "straight":
            self.state = STATES.STRAIGHT
        elif self.last_detected_sign == "give_way":
            self.state = STATES.GIVE_WAY
        elif self.last_detected_sign == "stop":
            self.state = STATES.STOP
        else:
            self.state = STATES.LINE_FOLLOW

        self.state_start_time = time.time()
        self.sign_counter.clear()
        self.last_detected_sign = "none"

    # ── Main loop ────────────────────────────────────────────────────────────

    def update(self):
        self.update_odometry()

        elapsed = time.time() - self.state_start_time

        # ── Apagar work_ahead si ya expiró ───────────────────────────────────
        if self.work_ahead and (time.time() - self.work_ahead_start) > self.work_ahead_duration:
            self.work_ahead = False
        # ─────────────────────────────────────────────────────────────────────

        # Estado STOP
        if self.state == STATES.STOP:
            if self.stop_area > self.stop_area_threshold:
                self.send_velocity(0.0, 0.0)
                return
            else:
                self.state = STATES.LINE_FOLLOW
                self.yolo_state = "none"

        # Estado ADVANCE_TO_CROSS
        if self.state == STATES.ADVANCE_TO_CROSS:
            dist = self.advance_goal - self.advance_distance

            if dist > 0:
                if abs(self.aangle_goal - self.aangle_theta) < self.cross_ang_tolerance:
                    self.send_velocity(0.10, 0.0)
                else:
                    self.send_velocity(0.10, self.cross_w)
            else:
                self.send_velocity(0.0, 0.0)

                if self.advance_finished_time is None:
                    self.advance_finished_time = time.time()

                if time.time() - self.advance_finished_time >= 2.0:
                    self.advance_finished_time = None
                    self.get_logger().info("stopping...")

                    # Second read removed
                    """
                    if self.cross_bool:
                        self.apply_saved_sign()
                        if self.console_debug:
                            self.get_logger().info("found second Cross sign!")
                    else:
                        self.get_logger().info("huh")
                        self.state = STATES.LINE_FOLLOW
                    """

                    self.apply_saved_sign()
                    if self.console_debug:
                        self.get_logger().info("Positioned in crosswalk!")

            return

        # Estado TURN_RIGHT
        # work_ahead reduce la velocidad lineal incluso durante el giro.
        if self.state == STATES.TURN_RIGHT:
            v_turn = 0.03 if self.work_ahead else 0.05
            self.send_velocity(v_turn, 1.0)

            if elapsed > 1.2:
                self.state = STATES.LINE_FOLLOW
                self.yolo_state = "none"

            return

        # Estado TURN_LEFT
        if self.state == STATES.TURN_LEFT:
            v_turn = 0.03 if self.work_ahead else 0.05
            self.send_velocity(v_turn, -1.0)

            if elapsed > 1.2:
                self.state = STATES.LINE_FOLLOW
                self.yolo_state = "none"

            return

        # Estado STRAIGHT
        if self.state == STATES.STRAIGHT:
            v_straight = 0.07 if self.work_ahead else 0.12
            self.send_velocity(v_straight, 0.0)

            if elapsed > 1.0:
                self.state = STATES.LINE_FOLLOW
                self.yolo_state = "none"

            return

        # Estado GIVE_WAY
        if self.state == STATES.GIVE_WAY:
            self.send_velocity(0.0, 0.0)

            if elapsed > 2.0:
                self.last_detected_sign = "none"
                self.sign_counter.clear()

                self.state = STATES.ADVANCE_TO_CROSS
                self.state_start_time = time.time()
                self.advance_distance = 0.0
                self.advance_finished_time = None

            return

        # ── LINE_FOLLOW con PID angular ───────────────────────────────────────
        # La velocidad base se reduce automáticamente si work_ahead está activo.
        v_base = self.work_ahead_speed if self.work_ahead else self.get_parameter('follow_speed').value

        e = float(self.line_error)

        now = time.time()
        dt = now - self.last_pid_time
        self.last_pid_time = now

        if dt <= 0.0:
            return

        kp_a = self.get_parameter('kp_ang').value
        ki_a = self.get_parameter('ki_ang').value
        kd_a = self.get_parameter('kd_ang').value

        self.int_ang += e * dt
        int_limit = self.w_max / (ki_a + 1e-9)
        self.int_ang = max(-int_limit, min(self.int_ang, int_limit))

        de_ang = (e - self.prev_ang_error) / dt
        self.prev_ang_error = e

        omega = kp_a * e + ki_a * self.int_ang + kd_a * de_ang
        omega = max(-self.w_max, min(omega, self.w_max))

        if abs(omega) > 0.15:
            v_base -= 0.05 * abs(omega)

        self.send_velocity(v_base, omega)

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def destroy_node(self):
        target = time.time() + 1
        while time.time() < target:
            rclpy.spin_once(self)
            self.send_velocity(0.0, 0.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = PuzzlebotController()
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
