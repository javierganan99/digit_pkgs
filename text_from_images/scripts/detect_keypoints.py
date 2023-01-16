#!/usr/bin/python3

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from helper import Homography


def image_callback(image: Image):
    """
    Function to detect the 4 corners.
    """
    global initialization
    img = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
    if initialization:
        color = n.get_points(img)
        if color is not None:
            initialization = False
            cv2.imshow("detected", color)
            cv2.waitKey(1000)
    else:
        color = n.get_normalize(img)
        color_show = color.copy()
        if show:  # We show the digits the model is recognizing
            # We loop over the 4 digits
            for i in range(4):
                color_show = cv2.rectangle(color_show, (boxes[i][0], boxes[i][2]), (
                    boxes[i][1], boxes[i][3]), (255, 0, 0), 1)
            cv2.imshow("detected", color_show)
            cv2.waitKey(1)
        pub_cropped.publish(bridge.cv2_to_imgmsg(
            color_show[boxes[0][2]:boxes[3][3], boxes[0][0]:boxes[3][1]], "bgr8"))
    return


if __name__ == "__main__":
    initialization = True
    # Normalizer to extract the images
    n = Homography()
    # From ROS images to Opencv images
    bridge = CvBridge()
    # Init node
    rospy.init_node("detect_keypoints")
    # Load parameters
    show = rospy.get_param("~show")
    camera = rospy.get_param("~camera")
    model_file = rospy.get_param("model_file")
    boxes = rospy.get_param("boxes")
    if camera == "blue":
        topic_digits = rospy.get_param("blue")["topic_digits"]
    elif camera == "elp":
        topic_digits = rospy.get_param("elp")["topic_digits"]
    elif camera == "zed":
        topic_digits = rospy.get_param("zed")["topic_digits"]
    elif camera == "rs":
        topic_digits = rospy.get_param("rs")["topic_digits"]
    elif camera == "ecap":
        topic_digits = rospy.get_param("ecap")["topic_digits"]
    else:
        rospy.logerr("Not the correct camera")
        raise Exception("Select the correct camera")
    rospy.loginfo("Ready to read and extract keypoints!")
    # Subscriber to the topic of the images with the digits
    sub = rospy.Subscriber(topic_digits,
                           Image, callback=image_callback)
    pub_cropped = rospy.Publisher("/cropped", Image, queue_size=10)
    rospy.spin()
