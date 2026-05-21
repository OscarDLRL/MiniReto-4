#!/usr/bin/env python3
"""line_follower_camera.py

Nodo ROS 2 para seguir una línea usando una cámara real o una imagen
recibida por topic.

Lógica de visión:
- Recorta el 1/4 inferior de la imagen
- Umbralización Otsu invertida + cierre morfológico
- Selecciona el contorno más cercano al centro con área mínima
- Calcula el error lateral firmado en [-100, 100]

Salida principal:
- publica std_msgs/Int16 en `/line_detector_error`
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from std_msgs.msg import Int16
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


class LineFollowerCamera(Node):
    def __init__(self):
        super().__init__("line_detector_error")

        # --- Parámetros ---
        self.declare_parameter("topic_im_read", False)
        self.declare_parameter("im_topic", "/cam/img_raw")
        self.declare_parameter("line_error_topic", "/line_detector_error")
        self.declare_parameter("image_debug", True)
        self.declare_parameter("console_debug", False)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 20)
        self.declare_parameter("camera_sensor_id", 0)
        self.declare_parameter("flip_method", 0)
        self.declare_parameter("publish_rate_hz", 20.0)

        # Parámetros de visión (CenterLineDetector)
        self.declare_parameter("min_contour_area", 260)
        self.declare_parameter("morph_kernel_size", 5)

        self.im_from_topic  = self.get_parameter("topic_im_read").value
        self.im_topic       = self.get_parameter("im_topic").value
        self.line_error_topic = self.get_parameter("line_error_topic").value
        self.im_debug       = self.get_parameter("image_debug").value
        self.debug          = self.get_parameter("console_debug").value
        self.cam_w          = int(self.get_parameter("camera_width").value)
        self.cam_h          = int(self.get_parameter("camera_height").value)
        self.camera_fps     = float(self.get_parameter("camera_fps").value)
        self.camera_sensor_id = int(self.get_parameter("camera_sensor_id").value)
        self.flip_method    = int(self.get_parameter("flip_method").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.min_contour_area = int(self.get_parameter("min_contour_area").value)
        self.morph_kernel_size = int(self.get_parameter("morph_kernel_size").value)

        self.bridge = CvBridge()
        self.cap    = None
        self.timer  = None
        self.sub    = None

        # Publisher
        self.line_error_pub = self.create_publisher(Int16, self.line_error_topic, 10)

        # Fuente de imagen
        if self.im_from_topic:
            self.get_logger().info(f"Leyendo imágenes del topic ROS2: {self.im_topic}")
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.sub = self.create_subscription(Image, self.im_topic, self._im_cb, qos)
        else:
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

        if self.im_debug:
            cv2.namedWindow("line_follower", cv2.WINDOW_AUTOSIZE)
            self.get_logger().info("Debug de imagen habilitado")

    # ------------------------------------------------------------------
    # Lógica de visión (extraída de CenterLineDetector)
    # ------------------------------------------------------------------

    def _detect_line(self, image):
        """
        Detecta la línea central en el cuarto inferior de la imagen.

        Returns dict con:
          - cx, cy        : centroide del mejor contorno (coords imagen completa)
          - binary        : imagen binarizada del ROI (para debug)
          - contours      : todos los contornos encontrados
          - best_contour  : contorno seleccionado (o None)
          - line_error    : int en [-100, 100]
          - center_error  : float en [-1.0, 1.0]
          - crop_point    : fila donde empieza el ROI
        """
        height, width = image.shape[:2]
        crop_point = 3 * height // 4

        # --- ROI: cuarto inferior ---
        roi = image[crop_point:height, 0:width]

        # --- Preprocesamiento ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # --- Contornos ---
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_cx = width // 2
        min_area  = self.min_contour_area

        def contour_score(contour):
            area = cv2.contourArea(contour)
            if area < min_area:
                return float("inf")
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return float("inf")
            cx = int(M["m10"] / M["m00"])
            return abs(cx - frame_cx)

        best_contour = None
        cx = frame_cx   # fallback: sin detección → error 0
        cy = crop_point + (height - crop_point) // 2

        if contours:
            candidate = min(contours, key=contour_score)
            M = cv2.moments(candidate)
            if M["m00"] != 0 and cv2.contourArea(candidate) >= min_area:
                best_contour = candidate
                cx_roi = int(M["m10"] / M["m00"])
                cy_roi = int(M["m01"] / M["m00"])
                cx = cx_roi
                cy = cy_roi + crop_point

        center_error = (cx - width / 2) / (width / 2)
        line_error   = int(np.clip(np.round(center_error * 100.0), -100, 100))

        return {
            "cx": cx,
            "cy": cy,
            "binary": binary,
            "contours": contours,
            "best_contour": best_contour,
            "line_error": line_error,
            "center_error": center_error,
            "crop_point": crop_point,
        }

    # ------------------------------------------------------------------
    # Debug visual
    # ------------------------------------------------------------------

    def _draw_debug(self, image, result):
        vis = image.copy()
        height, width = vis.shape[:2]

        # Superposición de la máscara binaria en verde sobre el ROI
        crop_point = result["crop_point"]
        bin_bgr = cv2.cvtColor(result["binary"], cv2.COLOR_GRAY2BGR)
        bin_bgr[:, :, 0] = 0   # quitar canal R
        bin_bgr[:, :, 2] = 0   # quitar canal B  → solo verde
        vis[crop_point:height, :] = cv2.addWeighted(
            vis[crop_point:height, :], 1.0, bin_bgr, 0.4, 0
        )

        # Línea de corte del ROI
        cv2.line(vis, (0, crop_point), (width, crop_point), (255, 255, 0), 1)

        # Contornos detectados (azul claro) y el mejor (rojo)
        if result["contours"]:
            cv2.drawContours(
                vis, result["contours"], -1, (200, 200, 0), 1,
                offset=(0, crop_point)
            )
        if result["best_contour"] is not None:
            cv2.drawContours(
                vis, [result["best_contour"]], -1, (0, 0, 255), 2,
                offset=(0, crop_point)
            )

        # Centro detectado vs centro de imagen
        cx, cy = result["cx"], result["cy"]
        cv2.line(vis, (width // 2, height - 1), (width // 2, crop_point), (220, 220, 220), 1)
        cv2.line(vis, (cx, height - 1), (cx, crop_point), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

        # Texto de error
        text = (
            f"line_error={result['line_error']:+d}  "
            f"center_err={result['center_error']:+.3f}"
        )
        cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 1, cv2.LINE_AA)

        return vis

    # ------------------------------------------------------------------
    # Pipeline de procesamiento
    # ------------------------------------------------------------------

    def _process_frame(self, frame):
        if frame is None or frame.size == 0:
            return

        if frame.shape[1] != self.cam_w or frame.shape[0] != self.cam_h:
            frame = cv2.resize(frame, (self.cam_w, self.cam_h))

        result = self._detect_line(frame)

        # Publicar error
        self.line_error_pub.publish(Int16(data=int(result["line_error"])))

        if self.debug:
            self.get_logger().info(
                f"line_error={result['line_error']:+d}  "
                f"center={result['center_error']:+.3f}  "
                f"cx={result['cx']}  cy={result['cy']}"
            )

        if self.im_debug:
            vis = self._draw_debug(frame, result)
            cv2.imshow("line_follower", vis)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _im_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._process_frame(frame)
        except Exception as exc:
            self.get_logger().error(f"Error de conversión de imagen: {exc}")

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
        self._process_frame(frame)

    def closeall(self):
        if self.cap is not None:
            self.cap.release()
        if self.im_debug:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.closeall()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()