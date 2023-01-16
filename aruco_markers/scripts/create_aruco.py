import argparse
import cv2
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="ArUco.png", help="path to output image containing ArUCo tag")
ap.add_argument("-i", "--id", type=int, default=0, help="ID of first ArUCo tag to generate")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
ap.add_argument("-d", "--dpi", type=str, default="72", help="the DPI of the output print")
ap.add_argument("-s", "--size", type=int, default=40, help="the size in mm of the ArUco tag")
ap.add_argument("-m", "--margin", type=int, default=5, help="the size in mm of the margins between the ArUco tags")
ap.add_argument("-x", "--x", type=int, default=3, help="number of ArUco tags in the X direction")
ap.add_argument("-y", "--y", type=int, default=4, help="number of ArUco tags in the Y direction")
ap.add_argument("-f", "--format", type=int, default=4, help="format of the paper")
args = vars(ap.parse_args())

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
	sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
tag_type = args["type"]

if args["format"] == 4:
	A_width = 210
	A_height = 297
elif args["format"] == 1:
	A_width = 594
	A_height = 841
else:
	raise Exception("The format must be A4 or A1. Enter number 4 or 1 in the format parameter")

x = args["x"]
y = args["y"]
size = args["size"]
margin = args["margin"]


if x < 1 or y < 1:
	print(f"[INFO] Please make sure that the grid contains at least one tag - i.e. (x > 0) and (y > 0). Currently, x = {x} and y = {y}.")
	sys.exit(0)


rest_x = A_width - (x * size + (x - 1) * margin)
rest_y = A_height - (y * size + (y - 1) * margin)


stop = False

if rest_x < 0:
	print(f"[INFO] Please ensure that the grid fits on the page. Consider reducing the number of tags in the x-direction. Currently, x = {x}.")
	stop = True

if rest_y < 0:
	print(f"[INFO] Please ensure that the grid fits on the page. Consider reducing the number of tags in the y-direction. Currently, y = {y}.")
	stop = True

if stop:
	sys.exit(0)


half_rest_x = int(np.floor(rest_x/2))
half_rest_y = int(np.floor(rest_y/2))

if args["format"] == 4:
	A_DICT = {
		"72": (595, 842),
		"96": (794, 1123)
	}
else:
	A_DICT = {
	"72": (1684, 2384),
	"96": (7016, 9933)
}


if A_DICT.get(args["dpi"], None) is None:
	print("[INFO] A4 print of {} DPI is not supported. Please try one of the following: 72, 96.".format(args["dpi"]))
	sys.exit(0)
	
dpi = A_DICT[args["dpi"]]
	
page = np.ones((dpi[1],dpi[0],3), dtype="uint8")*255

multiplier = np.min([dpi[0]/A_width, dpi[1]/A_height])


size_m = int(np.floor(size * multiplier))
margin_m = int(np.floor(margin * multiplier))
half_rest_x_m = int(np.floor(half_rest_x * multiplier))
half_rest_y_m = int(np.floor(half_rest_y * multiplier))

tag_id = args["id"]

print(f"[INFO] creating {x*y} tags from the {tag_type} dictionary. Starting with id:{tag_id}")
for i in range(0, y):
	for j in range(0, x):
		img = np.ones((size_m,size_m,3), dtype="uint8")*255
		i_val = half_rest_y_m + i*size_m + i*margin_m
		j_val = half_rest_x_m + j*size_m + j*margin_m
		tag = np.zeros((size_m, size_m, 1), dtype="uint8")
		cv2.aruco.drawMarker(arucoDict, tag_id, size_m, tag, 1)
		page[i_val:i_val+size_m, j_val:j_val+size_m] = tag
		tag_id += 1

cv2.imwrite(args["output"], page)
cv2.imshow("ArUCo Tags Page", page)
cv2.waitKey(10)