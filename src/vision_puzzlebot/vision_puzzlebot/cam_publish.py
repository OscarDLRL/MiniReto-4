#!/usr/bin/env python3
"""cam_publish.py

Nodo ROS 2 q publica la camara al topic

- publica std_msgs/Int16 en `/line_detector_error`
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from std_msgs.msg import Int16, Float32
from sensor_msgs.msg import Image


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=4032,
    capture_height=3040,
    display_width=640,
    display_height=480,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor_mode=0 sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


class CameraNode(Node):
    def __init__(self):
        super().__init__("CameraNode")

        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 20)
        self.declare_parameter("camera_sensor_id", 0)
        self.declare_parameter("flip_method", 0)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("topic", '/cam/img_raw')
        self.cam_w          = int(self.get_parameter("camera_width").value)
        self.cam_h          = int(self.get_parameter("camera_height").value)
        self.camera_fps     = float(self.get_parameter("camera_fps").value)
        self.camera_sensor_id = int(self.get_parameter("camera_sensor_id").value)
        self.flip_method    = int(self.get_parameter("flip_method").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.cam_topic = self.get_parameter('topic').value
        
        self.cam_publisher = self.create_publisher(Image, self.cam_topic, 10)
        self.bridge = CvBridge()
        
        self.get_logger().info("Leyendo imágenes desde pipeline GStreamer")
        self.cap = cv2.VideoCapture(
            gstreamer_pipeline(
                sensor_id=self.camera_sensor_id,
                display_width=self.cam_w,
                display_height=self.cam_h,
                framerate=int(self.camera_fps),
                flip_method=self.flip_method,
            ),
            cv2.CAP_GSTREAMER,
        )
        if not self.cap.isOpened():
            self.get_logger().error("No se pudo abrir el stream de cámara")

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._timer_cb)

    def _timer_cb(self):
        if self.cap is None:
            return
        if not self.cap.isOpened():
            self.get_logger().error("El stream de cámara está cerrado")
            return
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("No se pudo leer frame de la cámara")
            return

        try:
            img = self.bridge.cv2_to_imgmsg(frame,'bgr8')
            self.cam_publisher.publish(img)
        except Exception as exc:
            self.get_logger().error(f"Error de conversión de imagen: {exc}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

