#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time
import csv


class CollectWheelData(Node):
    def __init__(self):
        super().__init__('collect_wheel_data')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.pub_L = self.create_publisher(Float32, '/VelocitySetL', qos)
        self.pub_R = self.create_publisher(Float32, '/VelocitySetR', qos)

        self.sub_L = self.create_subscription(Float32, '/VelocityEncL', self.cb_L, qos)
        self.sub_R = self.create_subscription(Float32, '/VelocityEncR', self.cb_R, qos)

        self.vel_L = 0.0
        self.vel_R = 0.0

        self.filename = 'wheel_data.csv'

    def cb_L(self, msg):
        self.vel_L = msg.data

    def cb_R(self, msg):
        self.vel_R = msg.data

    def send(self, left, right):
        self.pub_L.publish(Float32(data=float(left)))
        self.pub_R.publish(Float32(data=float(right)))

    def stop(self):
        self.send(0.0, 0.0)
        time.sleep(0.5)

    def run_test(self):
        tests = [
            ("stop", 0.0, 0.0, 1.0),
            ("straight_slow", -0.3, -0.3, 3.0),
            ("stop", 0.0, 0.0, 1.0),
            ("straight_medium", -0.5, -0.5, 3.0),
            ("stop", 0.0, 0.0, 1.0),
            ("turn_left", 0.3, -0.3, 3.0),
            ("stop", 0.0, 0.0, 1.0),
            ("turn_right", -0.3, 0.3, 3.0),
            ("stop", 0.0, 0.0, 1.0),
            ("curve", -0.3, -0.6, 3.0),
            ("stop", 0.0, 0.0, 1.0),
        ]

        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'test',
                't',
                'set_L',
                'set_R',
                'vel_L',
                'vel_R'
            ])

            global_start = time.time()

            for name, set_L, set_R, duration in tests:
                print(f"\nPRUEBA: {name}  L={set_L}  R={set_R}")

                start = time.time()

                while time.time() - start < duration and rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.0)

                    self.send(set_L, set_R)

                    t = time.time() - global_start

                    writer.writerow([
                        name,
                        round(t, 4),
                        set_L,
                        set_R,
                        round(self.vel_L, 4),
                        round(self.vel_R, 4)
                    ])

                    print(
                        f"t={t:.2f}, "
                        f"setL={set_L:.2f}, setR={set_R:.2f}, "
                        f"velL={self.vel_L:.2f}, velR={self.vel_R:.2f}"
                    )

                    time.sleep(0.02)

                self.stop()

        print(f"\nDatos guardados en: {self.filename}")


def main():
    rclpy.init()
    node = CollectWheelData()

    try:
        node.run_test()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
