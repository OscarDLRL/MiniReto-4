#!/usr/bin/env python3 
"""
trafficlight_detect.py
Detecta luces de trafico, las envia a el topico {TODO: definir el topico xdd}
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

import cv2
from cv_bridge import CvBridge
import numpy as np

# Video Capturer
def gstreamer_pipeline(
        sensor_id = 0,
        capture_width=4032,
        capture_height=3040,
        display_width=504,
        display_height=380,
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


# NODE
class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_detector')
        
        #Parameters 
        self.declare_parameter('topic_im_read', False) # Specify if the image is obtained via node subscription, or if the pipeline is called
        self.im_from_topic = self.get_parameter('topic_im_read').value
        self.declare_parameter('im_topic', '/cam/img_raw') # Only use if the image is read from a topic
        self.im_topic = self.get_parameter('im_topic').value
        self.declare_parameter('image_debug', True) 
        self.im_debug = self.get_parameter('image_debug').value
        self.declare_parameter('console_debug',False)
        self.debug = self.get_parameter('console_debug').value

        self.declare_parameter('camera_width', 640)
        self.cam_w = self.get_parameter('camera_width').value
        self.declare_parameter('camera_height', 480)
        self.cam_h = self.get_parameter('camera_height').value

        self.declare_parameter('half', 'left')
        # 'left' -> Only process left side of the image (we know the traffic light is in this halve)
        # 'right' -> Only process right side of the image (we know the traffic light is in this halve)
        # 'both' -> Process whole image (We don't know where the traffic light will be)
        self.half = self.get_parameter('half').value
        """klajsdlkj"""

        # HSV BOUNDS
        self.declare_parameter('red_hsv_lower', [0,150,70])
        l = self.get_parameter('red_hsv_lower').value
        self.redlower = np.array(l ,dtype=np.uint8)
        self.declare_parameter('red_hsv_upper', [5,255,255])
        l = self.get_parameter('red_hsv_upper').value
        self.redupper = np.array(l, dtype=np.uint8)

        # Red needs 2 bounds ...
        self.declare_parameter('red2_hsv_lower', [165,150,70])
        l = self.get_parameter('red2_hsv_lower').value
        self.redlower2 = np.array(l ,dtype=np.uint8)
        self.declare_parameter('red2_hsv_upper', [180,255,255])
        l = self.get_parameter('red2_hsv_upper').value
        self.redupper2 = np.array(l, dtype=np.uint8)

        self.declare_parameter('yellow_hsv_lower', [25,40,140])
        l = self.get_parameter('yellow_hsv_lower').value
        self.yellowlower = np.array(l, dtype=np.uint8)
        self.declare_parameter('yellow_hsv_upper', [49,255,255])
        l = self.get_parameter('yellow_hsv_upper').value
        self.yellowupper = np.array(l, dtype=np.uint8)

        self.declare_parameter('green_hsv_lower', [50,50,70])
        l = self.get_parameter('green_hsv_lower').value
        self.greenlower = np.array(l, dtype=np.uint8)
        self.declare_parameter('green_hsv_upper', [80,255,255])
        l = self.get_parameter('green_hsv_upper').value
        self.greenupper = np.array(l, dtype=np.uint8)

        self.declare_parameter('min_pixels', 2000) # minimum amount of pixels to detet an image
        self.minscore = self.get_parameter('min_pixels').value

        self.colorpub = self.create_publisher(String, '/Traffic_light', 10)
        self.timer = None
        self.sub = None
        self.cap = None
        self.bridge = None
        self.cal_mat = None

        if self.im_from_topic:
            self.get_logger().info(f"READING IMAGE FROM ROS2 TOPIC: {self.im_topic}")
            self.sub = self.create_subscription( Image, self.im_topic, self._im_cb, 10 )
            self.bridge = CvBridge()
        else:
            self.get_logger().info("READING IMAGE THROUGH GSTREAM")
            self.timer = self.create_timer(1.0/60, self._timer_cb)
            self.cap = cv2.VideoCapture(
                    gstreamer_pipeline(framerate=10, display_width=self.cam_w, display_height=self.cam_h),
                    cv2.CAP_GSTREAMER
                    )

            try: 
                share_dir = get_package_share_directory('vision_puzzlebot')
                filepath = os.path.join(share_dir, 'data', 'colorCalibration.npz')
                with np.load(filepath) as data:
                    self.cal_mat = cv2.resize(data['arr_0'], (self.cam_w,self.cam_h))
            except Exception as e:
                self.get_logger().info(f"WARNING: Unable to correct image colors: {e}")
                self.cal_mat = None
            

        if self.im_debug:
            self.get_logger().info("IMAGE DEBUG ACTIVE")
            cv2.namedWindow("demo",cv2.WINDOW_AUTOSIZE)

    def closeall(self):
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def _detect_traffic(self, image):
        h,w = image.shape[:2]

        if self.half in ('left', 'right'):
            crop_point = w // 2
            if self.half == 'left':
                img = image[0:h, 0:crop_point]
            else:
                img = image[0:h, crop_point:w]
        elif self.half == 'both':
            img = image
        else:
            self.get_logger().error("unvalid 'half' value, only 'left', 'right' and 'both' are accepted")
            return
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply filters
        redmask = cv2.inRange(hsv, self.redlower, self.redupper)
        redmask2 = cv2.inRange(hsv, self.redlower2, self.redupper2)
        yellowmask = cv2.inRange(hsv, self.yellowlower, self.yellowupper)
        greenmask = cv2.inRange(hsv, self.greenlower, self.greenupper)

        values = {
                "red" : cv2.countNonZero(redmask),
                "red2" : cv2.countNonZero(redmask2),
                "yellow" : cv2.countNonZero(yellowmask),
                "green" : cv2.countNonZero(greenmask),
                "none" : self.minscore
                }

        state = max(values, key=values.get)

  #      self.get_logger().info(f"{state}, {values[state]}")

        # Publish to ros
        if state == "red2":
            statemsg = "red"
        else:
            statemsg = state
        if self.debug:
            self.get_logger().info("Publishing data...")
        self.colorpub.publish(String(data=statemsg))

        # Show Image
        if self.im_debug:
            if state == "none":
                cv2.imshow("demo", image)
                cv2.waitKey(1)
                return
            elif state == "red":
                mask = redmask
            elif state == "red2":
                mask = redmask2
                state = "red" # Send just "red to image debug
            elif state == "yellow":
                mask = yellowmask
            elif state == "green":
                mask = greenmask

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
                )
            if not contours:
                cv2.imshow("demo", image)
                cv2.waitKey(1)
                return

            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            self.get_logger().info(f"{area}")
            cv2.drawContours(image, [best], -1, (255,255,0),2)
            cv2.putText(image, f'{state}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("demo", image)
            cv2.waitKey(1)



    # CALLBACK FOR ROS2 TOPIC IMAGE
    def _im_cb(self, im : Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(im, "bgr8")
            self._detect_traffic(frame)
        except Exception as e:
            self.get_logger().error(f"Conversion error: {e}")

    # CALLBACK FOR GSTREAM IMAGE
    def _timer_cb(self):
        if not self.cap.isOpened():
            self.get_logger().error("Tried to read video capture, but stream is closed!")
            return
        _,frame = self.cap.read()

        if self.cal_mat is not None:
            frame = np.uint8(self.cal_mat*frame)


        self._detect_traffic(frame)



def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.closeall()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
