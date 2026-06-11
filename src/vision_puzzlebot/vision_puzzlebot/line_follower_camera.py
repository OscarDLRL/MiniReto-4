#!/usr/bin/env python3
"""line_follower_camera.py

Nodo ROS 2 para seguir una línea usando una cámara real o una imagen
recibida por topic. También detecta cruces peatonales.

Lógica de visión (línea):
- Recorta el 1/4 inferior de la imagen
- Umbralización Otsu invertida + cierre morfológico
- Selecciona el contorno más cercano al centro con área mínima
- Calcula el error lateral firmado en [-1, 1]

Lógica de visión (cruce peatonal):
- Recorta el 1/3 inferior de la imagen
- Busca grupos de 6 contornos similares alineados horizontalmente
- Publica True/False en /crosswalk_bool

Salidas:
- /line_detector_error  → std_msgs/Float32
- /crosswalk_bool       → std_msgs/Bool
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from std_msgs.msg import Float32, Bool
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


# Crosswalk detection helpers

def _get_contour_features(c):
    """Return (cx, cy, area, aspect_ratio) for a contour."""
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    area = cv2.contourArea(c)
    aspect = w / h if h > 0 else 0
    return cx, cy, area, aspect

def _fit_slope(contours):
    """
    Fit a line through the centroids of the given contours using cv2.fitLine.
    Returns (angle_deg, p1, p2) where p1/p2 are draw endpoints.
    Returns None if fewer than 2 contours.
    """

    pts = np.array(
        [[_get_contour_features(c)[0], _get_contour_features(c)[1]] for c in contours],
        dtype=np.float32
    )
    if len(pts) < 2:
        return None
 
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
 
    x_min = int(pts[:, 0].min())
    x_max = int(pts[:, 0].max())
    span  = max(x_max - x_min, 1)
 
    p1 = (int(x0 - vx * span), int(y0 - vy * span))
    p2 = (int(x0 + vx * span), int(y0 + vy * span))
 
    angle = float(np.degrees(np.arctan2(vy, vx)))

    perp_distances = vy * (pts[:, 0] - x0) - vx * (pts[:, 1] - y0)
    mse = float(np.mean(perp_distances ** 2))

    return angle, p1, p2, mse




class LineFollowerCamera(Node):
    def __init__(self):
        super().__init__("line_follower_camera")

        # Parameters  
        # Valgame dios
        self.declare_parameter("topic_im_read", False)
        self.declare_parameter("im_topic", "/cam/img_raw")
        self.declare_parameter("line_error_topic", "/line_detector_error")
        self.declare_parameter("crosswalk_topic",  "/crosswalk_ang")
        self.declare_parameter("crosswalk_topic_bool",  "/crosswalk_bool")
        self.declare_parameter("image_debug", True) # For the line part
        self.declare_parameter("crosswalk_debug", True) # For the crosswalk
        self.declare_parameter("console_debug", False)
        
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 20)
        self.declare_parameter("camera_sensor_id", 0)
        self.declare_parameter("flip_method", 0)
        self.declare_parameter("publish_rate_hz", 20.0)

        # Parámetros de detección de línea
        self.declare_parameter("morph_kernel_size",   5)
        self.declare_parameter("min_contour_area",380 )


        # Parámetros de detección de cruce peatonal
        self.declare_parameter("cluster_minarea", 100)
        self.declare_parameter("cluster_maxarea", 4000)
        self.declare_parameter("cluster_MSE", 1000)
        self.declare_parameter("cluster_ASR", 3.7) # AREA SIMILARITY ERROR
        self.declare_parameter("cluster_AsSR", 3.8) # Aspect Similarity Ratio

        self.im_from_topic   = self.get_parameter("topic_im_read").value
        self.im_topic = self.get_parameter("im_topic").value
        self.line_error_topic= self.get_parameter("line_error_topic").value
        self.crosswalk_topic = self.get_parameter("crosswalk_topic").value
        self.crosswalkb_topic = self.get_parameter("crosswalk_topic_bool").value
        self.im_debug = self.get_parameter("image_debug").value
        self.crosswalk_debug = self.get_parameter("crosswalk_debug").value
        self.debug = self.get_parameter("console_debug").value
        self.cam_w = int(self.get_parameter("camera_width").value)
        self.cam_h = int(self.get_parameter("camera_height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)
        self.camera_sensor_id= int(self.get_parameter("camera_sensor_id").value)
        self.flip_method = int(self.get_parameter("flip_method").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.min_contour_area= int(self.get_parameter("min_contour_area").value)
        self.morph_kernel_size=int(self.get_parameter("morph_kernel_size").value)

        self.cluster_min_area = self.get_parameter("cluster_minarea").value
        self.cluster_max_area = self.get_parameter("cluster_maxarea").value
        self.cluster_maxslopeerror = self.get_parameter("cluster_MSE").value
        self.cluster_area_ratio_max = self.get_parameter("cluster_ASR").value # AREA SIMILARITY ERROR
        self.cluster_aspect_ratio_max = self.get_parameter("cluster_AsSR").value # Aspect Similarity Ratio

        self.bridge      = CvBridge()
        self.cap         = None
        self.timer       = None
        self.sub         = None
        self.last_centre = (-1, -1)

        # Publishers
        self.line_error_pub  = self.create_publisher(Float32, self.line_error_topic, 10)
        self.crosswalk_pub   = self.create_publisher(Float32,    self.crosswalk_topic,  10)
        self.crosswalk_bool_pub   = self.create_publisher(Bool,    self.crosswalkb_topic,  10)

        # Image  source
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
            self.get_logger().info("???")
            if not self.cap.isOpened():
                self.get_logger().error("No se pudo abrir el stream de cámara")
            self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._timer_cb)

        if self.im_debug:
            cv2.namedWindow("line_follower", cv2.WINDOW_AUTOSIZE)
            self.get_logger().info("Debug de imagen habilitado")

        if self.crosswalk_debug:
            cv2.namedWindow("crosswalk_debug", cv2.WINDOW_AUTOSIZE)
            self.get_logger().info("Debug de crosswalk habilitado")


    # Line Detection
    def _detect_line(self, image):
        height, width = image.shape[:2]
        crop_point = 3 * height // 4

        roi  = image[crop_point:height, 0:width]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 5, 5))
        dilation = cv2.dilate(blur, kernel, iterations=1)

        _, binary = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            cy = int(M["m01"] / M["m00"])
            score = abs(cx - frame_cx)
            if self.last_centre[0] != -1:
                score += (((cx - self.last_centre[0]) ** 2 + (cy + self.last_centre[1]) ** 2) ** 0.5)
            return score

        best_contour = None
        cx = frame_cx
        cy = crop_point + (height - crop_point) // 2

        if contours:
            candidate = min(contours, key=contour_score)
            M = cv2.moments(candidate)
            if M["m00"] != 0 and cv2.contourArea(candidate) >= min_area:
                best_contour = candidate
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) + crop_point

        center_error = (cx - width / 2) / (width / 2)
        line_error   = float(np.clip(center_error, -1, 1))

        return {
            "cx": cx, "cy": cy,
            "binary": binary,
            "contours": contours,
            "best_contour": best_contour,
            "line_error": line_error,
            "center_error": center_error,
            "crop_point": crop_point,
        }

    # Crosswalk Detection Function

    def _detect_crosswalk(self, image):
        height, width = image.shape[:2]
        crop_point = 2 * height // 4

        # PRE PROCESSING
        roi = image[crop_point:height, 0:width]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11,11), 0)
        kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 5, 5))
        dilation = cv2.dilate(blur, kernel, iterations=2)
        
        _, binary = cv2.threshold( dilation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # CONTOURS
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(0,crop_point))

        frame_cx = width // 2
        min_area  = self.cluster_min_area
        max_area = self.cluster_max_area

        candidates = [ c for c in contours if cv2.contourArea(c) >= min_area and cv2.contourArea(c) <= max_area]

        # --- Cluster by Y and look for a group of 6 ---
        clusters = self._similarity_clusters(candidates)
        if len(clusters) == 0:
            return False, -1.0
        validfound = len(clusters[0]) >= 4
        validangle = 0.0
        ret = image
        colors = [ (0,0,200), (0,200,0), (200,0,0) ]
     
        i = 0
        for cluster in clusters:
            try: 
                a, p1,p2,mse = _fit_slope(cluster)
            except TypeError:
                mse = self.cluster_maxslopeerror +1
                pass

            if i == 0:
                if mse > self.cluster_maxslopeerror:
                    #self.get_logger().info(f"Too ucho error ({mse}<{self.cluster_maxslopeerror}")
                    validfound = False
                else:
                    validangle = a
            if self.crosswalk_debug:
                if validfound and i == 0 :
                    # Draw the matched group in green with bounding boxes
                    cv2.drawContours(ret, cluster, -1, (0, 255, 0), 2)
                    cv2.line(image, p1, p2, (255, 128, 0), 2)
                elif len(cluster) >= 4:
                    # Draw other candidates in red so you can debug
                    cv2.drawContours(ret, cluster, -1, colors[i%3], 1)
                    cv2.line(image, p1, p2, colors[i%3], 1)
                else:
                    cv2.drawContours(ret, cluster, -1, colors[i%3], 1)

            i += 1

        if self.crosswalk_debug:
            #self.get_logger().info(f"Total clusters: {i}")
            cv2.imshow("crosswalk_debug", ret)
            cv2.waitKey(1)

        return validfound, validangle

    # HELPER FUNCTIONS FOR CROSSWALK DETECTION
    
    def _similarity_clusters(self, contours):
        clusters = []
        for c in contours:
            _, _, area, aspect = _get_contour_features(c)
            placed = False
     
            for cl in clusters:
                med_area   = float(np.median(cl["areas"]))
                med_aspect = float(np.median(cl["aspects"]))
     
                area_ok   = max(area, med_area)   / min(area, med_area)   < self.cluster_area_ratio_max
                aspect_ok = (
                    max(aspect, med_aspect) / min(aspect, med_aspect) < self.cluster_aspect_ratio_max
                    if aspect > 0 and med_aspect > 0 else False
                )
     
                if area_ok and aspect_ok:
                    cl["contours"].append(c)
                    cl["areas"].append(area)
                    cl["aspects"].append(aspect)
                    placed = True
                    break
     
            if not placed:
                clusters.append({
                    "contours": [c],
                    "areas":    [area],
                    "aspects":  [aspect],
                })
     
        return sorted([cl["contours"] for cl in clusters], key=len, reverse=True)

    # Visual Debug for line
    def _draw_debug(self, image, line_result):
        vis = image.copy()
        height, width = vis.shape[:2]
        crop_point = line_result["crop_point"]

        bin_bgr = cv2.cvtColor(line_result["binary"], cv2.COLOR_GRAY2BGR)
        bin_bgr[:, :, 0] = 0
        bin_bgr[:, :, 2] = 0
        vis[crop_point:height, :] = cv2.addWeighted(
            vis[crop_point:height, :], 1.0, bin_bgr, 0.4, 0
        )

        cv2.line(vis, (0, crop_point), (width, crop_point), (255, 255, 0), 1)

        if line_result["contours"]:
            cv2.drawContours(vis, line_result["contours"], -1, (200, 200, 0), 1,
                             offset=(0, crop_point))
        if line_result["best_contour"] is not None:
            cv2.drawContours(vis, [line_result["best_contour"]], -1, (0, 0, 255), 2,
                             offset=(0, crop_point))

        cx, cy = line_result["cx"], line_result["cy"]
        cv2.line(vis, (width // 2, height - 1), (width // 2, crop_point), (220, 220, 220), 1)
        cv2.line(vis, (cx, height - 1), (cx, crop_point), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

        text = f"center_err={line_result['center_error']:+.3f}"
        cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 1, cv2.LINE_AA)
        return vis

    # ====================================================================== #
    # Pipeline principal                                                       #
    # ====================================================================== #

    def _process_frame(self, frame):
        if frame is None or frame.size == 0:
            return

        if frame.shape[1] != self.cam_w or frame.shape[0] != self.cam_h:
            frame = cv2.resize(frame, (self.cam_w, self.cam_h))

        # --- Detección de línea ---
        line_result = self._detect_line(frame)
        self.last_centre = (line_result["cx"], line_result["cy"])
        self.line_error_pub.publish(Float32(data=line_result["line_error"]))

        # --- Detección de cruce peatonal ---
        crosswalk_detected,crosswalk_angle = self._detect_crosswalk(frame)
        self.crosswalk_bool_pub.publish(Bool(data=crosswalk_detected))
        self.crosswalk_pub.publish(Float32(data=crosswalk_angle))

        """
        if self.debug:
            self.get_logger().info(
                f"line_error={line_result['line_error']:+.3f}  "
                f"center={line_result['center_error']:+.3f}  "
                f"cx={line_result['cx']}  cy={line_result['cy']}  "
                f"crosswalk={crosswalk_detected}"
            )
        """

        if self.im_debug:
            # Superponer debug de línea sobre el frame con contornos de cruce
            vis = self._draw_debug(cw_vis, line_result)
            cv2.imshow("line_follower", vis)
            cv2.waitKey(1)

    # ====================================================================== #
    # Callbacks                                                                #
    # ====================================================================== #

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
