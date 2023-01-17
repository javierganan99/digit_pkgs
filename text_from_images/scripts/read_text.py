#!/usr/bin/python3

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from helper import Homography, normalize
from tensorflow import keras
import tensorflow as tf

# Needed to asign memory to GPUs ###############################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
################################################################################


def open_model_file():
    """
    Open the model and the file to write the detected digits vs timestamp.
    """
    global file, model
    file = open(output_file_luxes, "w+")
    file.write("")
    file.close()
    model = keras.models.load_model(
        model_file)


def image_callback(image: Image):
    """
    Function to detect the digits and write them to a file.
    """
    global file
    # Read image from which text needs to be extracted
    img = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """
    GRADED:
    Call the get_normalize method of Homography class (passing img as a parameter) and save the returned image in img variable 
    ~ 1 line
    """
    if show:
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    file = open(output_file_luxes, "a")
    # To save the 4 digits
    n_4 = ""
    # We loop over the 4 digits
    for i in range(4):
        cropped = img[
            boxes[i][2]: boxes[i][3],
            boxes[i][0]: boxes[i][1],
        ]
        # Denoise image
        cropped = cv2.fastNlMeansDenoising(cropped, None, 10, 7, 21)
        # Min max normalization
        cropped = normalize(cropped, (0, 1))
        # Resize the image
        cropped = cv2.resize(
            cropped, (28, 28), interpolation=cv2.INTER_NEAREST)
        # Predict the digit
        number = np.argmax(model.predict(np.expand_dims(cropped, axis=0), verbose = 0))
        if show:  # We show the digits the model is recognizing
            ishow = cv2.putText(
                color,
                str(number),
                (boxes[i][0], boxes[i][2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color=(255, 0, 0),
            )
        n_4 += str(number)
    if show:
        cv2.imshow("detected", ishow)
        cv2.waitKey(1)
    # To write the detected digits along with the timestamp
    file.write(
        str(float(n_4) / 10))
    file.write("\t")
    file.write(str(image.header.stamp.to_sec()))
    file.write("\n")
    file.close


if __name__ == "__main__":
    # Homography to extract the digits from the image
    n = Homography()
    # From ROS images to Opencv images
    bridge = CvBridge()
    # Init node
    rospy.init_node("read_text")
    # Read parameters from launch
    show = rospy.get_param("~show")
    camera = rospy.get_param("~camera")
    catkin_path = rospy.get_param("catkin_path")
    if not catkin_path[-1] == "/":
        catkin_path += "/"
    model_file = catkin_path + rospy.get_param("model_file")
    boxes = rospy.get_param("boxes")
    cameras_list = ["blue", "elp", "zed", "rs", "ecap"]
    if camera in cameras_list:
        topic_digits = rospy.get_param(camera)["topic_digits"]
        n.int_points = rospy.get_param(camera)["int_points"]
        n.dst_points = rospy.get_param(camera)["pts_dst"]
        output_file_luxes = catkin_path + rospy.get_param(camera)["output_file_luxes"]
    else:
        rospy.logerr("Not the correct camera")
        raise Exception("Select the correct camera")
    rospy.loginfo("Ready to read images and detect the digits!")
    # Open the file to write the data and load the model
    open_model_file()
    # Subscriber to the topic of the images with the digits
    sub = rospy.Subscriber(topic_digits,
                           Image, callback=image_callback)
    rospy.spin()
