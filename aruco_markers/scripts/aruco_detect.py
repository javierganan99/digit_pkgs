#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge
import csv
import yaml
from yaml.loader import SafeLoader

ARUCO_DICT = {  # Dictionary with different types of aruco
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11,
}


def get_image_APS(data: Image):
    """
    Get the APS images in data argument, undistort them, detect the arucos and show
    the image with detections, and save the timestamp and the number of detected arucos.
    """
    global store_time_APS
    global store_APS

    cont_APS = 0
    frame = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    if calibration_path:
        mapx, mapy = cv.initUndistortRectifyMap(
            K, D, None, P, (frame.shape[1], frame.shape[0]), 5)
        frame = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
    store_time_APS.append(data.header.stamp.to_sec())
    markerCorners, markerIds, _ = cv.aruco.detectMarkers(
        frame, dictionary, parameters=parameters
    )
    if frame.shape[-1] == 3:
        color = frame
    else:
        color = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    if markerIds is not None:
        for i, corner in zip(markerIds, markerCorners):
            cont_APS += 1
            color = cv.putText(
                color,
                "id {0}".format(i),
                (int(corner[0][0][0]) + 20, int(corner[0][0][1]) + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    store_APS.append(cont_APS)
    if show:
        color = cv.aruco.drawDetectedMarkers(
            color, markerCorners, borderColor=(0, 255, 0)
        )
        cv.imshow("APS", color)
        cv.waitKey(1)


def get_image_reconstructed(data: Image):
    """
    Get the reconstructed images in data argument, detect the arucos and show
    the image with detections, and save the timestamp and the number of detected arucos.
    """
    global store_time_R
    global store_R
    cont_R = 0
    frame = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    store_time_R.append(data.header.stamp.to_sec())

    markerCorners, markerIds, _ = cv.aruco.detectMarkers(
        frame, dictionary, parameters=parameters
    )
    color = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    if markerIds is not None:
        for i, corner in zip(markerIds, markerCorners):
            cont_R += 1
            color = cv.putText(
                color,
                "id {0}".format(i),
                (int(corner[0][0][0]) + 20, int(corner[0][0][1]) + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    store_R.append(cont_R)
    if show:
        color = cv.aruco.drawDetectedMarkers(
            color, markerCorners, borderColor=(0, 255, 0)
        )
        cv.imshow("Reconstructed", color)
        cv.waitKey(1)


def init_calib(path: str):
    """
    Function to load the calibration parameters.
    """
    global K, P, D, R
    # Open the yaml file and load the file
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    K = np.array(data["camera_matrix"]["data"]).reshape((3, 3))
    D = np.array(data["distortion_coefficients"]["data"])
    R = np.array(data["rectification_matrix"]["data"]).reshape((3, 3))
    P = np.array(data["projection_matrix"]["data"]).reshape((3, 4))[:, :3]


if __name__ == "__main__":
    # To convert from ros Image to opencv image
    bridge = CvBridge()

    # To save the time and number of detected arucos
    store_time_APS = []
    store_time_R = []
    store_APS = []
    store_R = []

    # Init node to detect arucos from topics
    rospy.init_node("detectAruco", anonymous=True)

    # Get params
    show = rospy.get_param("~show")
    camera = rospy.get_param("~camera")
    aruco_type = rospy.get_param("aruco_type")
    if camera == "blue":
        calibration_path = rospy.get_param("blue")["calibration_path"]
        aps_frames_topic = rospy.get_param("blue")["aps_frames_topic"]
        reconstructed_frames_topic = rospy.get_param(
            "blue")["reconstructed_frames_topic"]
        output_file_arucos = rospy.get_param("blue")["output_file_arucos"]
    elif camera == "elp":
        calibration_path = rospy.get_param("elp")["calibration_path"]
        aps_frames_topic = rospy.get_param("elp")["aps_frames_topic"]
        reconstructed_frames_topic = rospy.get_param(
            "elp")["reconstructed_frames_topic"]
        output_file_arucos = rospy.get_param("elp")["output_file_arucos"]
    elif camera == "zed":
        calibration_path = rospy.get_param("zed")["calibration_path"]
        aps_frames_topic = rospy.get_param("zed")["aps_frames_topic"]
        reconstructed_frames_topic = rospy.get_param(
            "zed")["reconstructed_frames_topic"]
        output_file_arucos = rospy.get_param("zed")["output_file_arucos"]
    elif camera == "rs":
        calibration_path = rospy.get_param("rs")["calibration_path"]
        aps_frames_topic = rospy.get_param("rs")["aps_frames_topic"]
        reconstructed_frames_topic = rospy.get_param(
            "rs")["reconstructed_frames_topic"]
        output_file_arucos = rospy.get_param("rs")["output_file_arucos"]
    elif camera == "ecap":
        calibration_path = rospy.get_param("ecap")["calibration_path"]
        aps_frames_topic = rospy.get_param("ecap")["aps_frames_topic"]
        reconstructed_frames_topic = rospy.get_param(
            "ecap")["reconstructed_frames_topic"]
        output_file_arucos = rospy.get_param("ecap")["output_file_arucos"]
    else:
        rospy.logerr("Not the correct camera")
        raise Exception("Select the correct camera")

    # Dictionary for arUco detection
    dictionary = cv.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    parameters = cv.aruco.DetectorParameters_create()

    # Subscribers
    rospy.Subscriber(
        aps_frames_topic, Image, get_image_APS
    )
    if reconstructed_frames_topic:
        rospy.Subscriber(
            reconstructed_frames_topic, Image, get_image_reconstructed
        )
    if calibration_path:
        # Load the calibration of the camera to undistort the APS frames
        init_calib(calibration_path)

    # Blocks until rosnode is shutdown
    rospy.spin()

    # When the node is shutdown, the time vs detected arucos is written to a csv file
    with open(output_file_arucos, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(store_time_APS)
        writer.writerow(store_APS)
        writer.writerow(store_time_R)
        writer.writerow(store_R)

    cv.destroyAllWindows()
