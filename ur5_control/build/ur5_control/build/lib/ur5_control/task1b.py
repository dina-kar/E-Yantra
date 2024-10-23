#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Logistic coBot (LB) Theme (eYRC 2024-25)
*        		===============================================
*
*  This script should be used to implement Task 1B of Logistic coBot (LB) Theme (eYRC 2024-25).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1b_boiler_plate.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image


##################### FUNCTION DEFINITIONS #######################

def calculate_rectangle_area(coordinates):
    '''
    Description:    Function to calculate the area of detected ArUco marker.

    Args:
        coordinates (list):     Coordinates of detected ArUco (4 sets of (x, y) coordinates).

    Returns:
        area        (float):    Area of detected ArUco marker.
        width       (float):    Width of detected ArUco marker.
    '''

    # Assuming the coordinates are in the following order:
    # coordinates = [top-left, top-right, bottom-right, bottom-left]

    # Extract coordinates
    top_left = coordinates[0]
    top_right = coordinates[1]
    bottom_right = coordinates[2]
    bottom_left = coordinates[3]

    # Calculate width (distance between top-left and top-right)
    width = np.linalg.norm(np.array(top_right) - np.array(top_left))

    # Calculate height (distance between top-left and bottom-left)
    height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))

    # Calculate area (width * height)
    area = width * height

    return area, width


def detect_aruco(image):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''

    # Constants and Variables
    aruco_area_threshold = 1500
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
    dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    size_of_aruco_m = 0.15

    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []

    # 1. Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Load the dictionary and parameters for ArUco detection
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # 3. Detect the ArUco markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # 4. Loop over all detected markers
        for i in range(len(ids)):
            corner = corners[i][0]  # Get the 4 corners of the marker
            marker_id = ids[i][0]   # Get the marker ID

            # Calculate the area and width of the marker
            area, width = calculate_rectangle_area(corner)

            # Filter out markers based on area threshold
            if area < aruco_area_threshold:
                continue  # Skip markers that are too far away or too small

            # Calculate the center point of the marker
            center_x = np.mean(corner[:, 0])
            center_y = np.mean(corner[:, 1])
            center_point = (center_x, center_y)

            # Pose estimation for the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], size_of_aruco_m, cam_mat, dist_mat)

            # Distance to the marker (using the translation vector)
            distance = np.linalg.norm(tvec[0][0])

            # Angle of the marker (using the rotation vector)
            angle = np.degrees(np.arctan2(tvec[0][0][0], tvec[0][0][2]))  # Angle in degrees

            # Draw the detected marker and axis on the image
            cv2.aruco.drawDetectedMarkers(image, corners)
            cv2.drawFrameAxes(image, cam_mat, dist_mat, rvec, tvec, 0.1)

            # Append the calculated values to the lists
            center_aruco_list.append(center_point)
            distance_from_rgb_list.append(distance)
            angle_aruco_list.append(angle)
            width_aruco_list.append(width)
            ids.append(marker_id)

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids


##################### CLASS DEFINITION #######################

class aruco_tf(Node):
    '''
    ___CLASS___

    Description:    Class which servers purpose to define process for detecting aruco marker and publishing tf on pose estimated.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class aruco_tf
                        All classes have a function called __init__(), which is always executed when the class is being initiated.
                        The __init__() function is called automatically every time the class is being used to create a new object.
                        You can find more on this topic here -> https://www.w3schools.com/python/python_classes.asp
        '''

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())


    def depthimagecb(self, data):
         '''
        Description: Callback function for aligned depth camera topic. 
                 Use this function to receive image depth data and convert to CV2 image

        Args:
            data (Image): Input depth image frame received from aligned depth camera topic
            Returns:
        '''
    try:
        # Convert depth image to OpenCV format
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    except CvBridgeError as e:
        self.get_logger().error(f"Error converting depth image: {str(e)}")


    def colorimagecb(self, data):
        '''
    Description: Callback function for color camera raw topic.
                 Use this function to receive raw image data and convert to CV2 image

    Args:
        data (Image): Input colored raw image frame received from image_raw camera topic
    Returns:
    '''
    try:
        # Convert color image to OpenCV format
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Optionally flip or rotate the image based on the camera orientation
        # Example: self.cv_image = cv2.rotate(self.cv_image, cv2.ROTATE_90_CLOCKWISE)
    except CvBridgeError as e:
        self.get_logger().error(f"Error converting color image: {str(e)}")


    def process_image(self):
        '''
        Description:    Timer function used to detect aruco markers and publish tf on estimated poses.

        Args:
        Returns:
        '''

        ############ Function VARIABLES ############

        # These are the variables defined from camera info topic such as image pixel size, focalX, focalY, etc.
        # Make sure you verify these variable values once. As it may affect your result.
        # You can find more on these variables here -> http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375
            

        
        def process_image(self):
            '''
            Description: Timer function used to detect aruco markers and publish tf on estimated poses.
            Args:
            Returns:
            '''
            if self.cv_image is None or self.depth_image is None:
                return  # If either image is not available, skip processing

            # Detect ArUco markers in the color image
            center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = detect_aruco(self.cv_image)

            if ids is None or len(ids) == 0:
                return  # No ArUco markers detected, skip processing

            for i, marker_id in enumerate(ids):
                # Get ArUco marker data
                cX, cY = center_aruco_list[i]
                distance_from_rgb = distance_from_rgb_list[i]
                angle_aruco = angle_aruco_list[i]

                # Correct the angle using the given formula
                angle_aruco = (0.788 * angle_aruco) - ((angle_aruco**2) / 3160)

                # Calculate real-world coordinates (x, y, z) using the focal length, image size, and center
                x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
                y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
                z = distance_from_rgb / 1000  # Convert from mm to meters

                # Calculate quaternions from roll, pitch, yaw (roll and pitch are 0, yaw is angle_aruco)
                yaw = np.radians(angle_aruco)  # Convert to radians
                q = tf_transformations.quaternion_from_euler(0, 0, yaw)

                # Publish transform from camera_link to ArUco marker (cam_<marker_id>)
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "camera_link"
                t.child_frame_id = f"cam_{marker_id}"

                t.transform.translation.x = x
                t.transform.translation.y = y
                t.transform.translation.z = z
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]

                self.br.sendTransform(t)

                # Draw the ArUco marker's center point on the image
                cv2.circle(self.cv_image, (int(cX), int(cY)), 5, (0, 0, 255), -1)

                # Now, publish the transform w.r.t. base_link
                try:
                    # Lookup the transform between base_link and camera_link
                    trans = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
                    
                    # Apply the transform to get the ArUco marker's position w.r.t base_link
                    obj_transform = tf2_geometry_msgs.do_transform_transform(t.transform, trans)

                    # Publish the transform between base_link and ArUco marker (obj_<marker_id>)
                    t_base = TransformStamped()
                    t_base.header.stamp = self.get_clock().now().to_msg()
                    t_base.header.frame_id = "base_link"
                    t_base.child_frame_id = f"obj_{marker_id}"

                    t_base.transform = obj_transform.transform
                    self.br.sendTransform(t_base)

                except tf2_ros.LookupException as e:
                    self.get_logger().error(f"Transform lookup failed: {str(e)}")

            # Display the image with detected ArUco markers
            cv2.imshow("ArUco Detection", self.cv_image)
            cv2.waitKey(1)

##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
                    You can find more on this here -> https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/
    '''
    print("Running")
    main()
