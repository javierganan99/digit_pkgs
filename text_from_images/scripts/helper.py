import cv2
import numpy as np
import rospy


def rotate_image(image, angle):
    """
    Rotate an image.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def resize_image(img, perc):
    """
    Resize the image percentage "perc".
    """
    width = int(img.shape[1] * perc / 100)
    height = int(img.shape[0] * perc / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    return img


def normalize(img, limits):
    """
    Normalize img between limits of intensity: limits = (min, max).
    """
    min_v, max_v = limits
    img = (img - img.min()) / (img.max() - img.min()) * (max_v - min_v) + min_v
    return img


def dilate_erode(img, kernel_dilatation, kernel_erosion):
    """
    This function consequtively dilate and erode with the kernels kernel_dilatation and kernel_erosion.
    """
    kernel_dil = np.ones((kernel_dilatation, kernel_dilatation), np.uint8)
    kernel_ero = np.ones((kernel_erosion, kernel_erosion), np.uint8)
    img = cv2.dilate(img, kernel_dil, iterations=1)
    img = cv2.erode(img, kernel_ero, iterations=1)
    return img


class Homography:
    """
    This class performs homography of a detected square in source image,
    and center it on another square image of dimensions (size,size).
    --get_points() detect 4 corner points based on hough transform.
    --get_normalize() performs the homography once the points are detected.
    """

    def __init__(self, size=100, th=100, h_r=1):
        self.int_points = []  # Source points to perform homography
        self.size = size  # Size of the destiny square image
        self.th = th  # Threshold to binarize the image
        self.h_r = h_r  # Threshold to rho for Hough Transform
        self.img_dst = np.zeros((self.size, self.size))  # Destiny square image
        self.dst_points = []  # Destiny points for the homography

    def get_points(self, img_src):
        """
        This function gets "img_src" that contains an square and detect its corners.
        """
        color = img_src.copy()
        if img_src.shape[-1] == 3:
            img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        _, img_src = cv2.threshold(
            img_src, self.th, 255, cv2.THRESH_BINARY)
        img_src = cv2.Canny(img_src, 10, 10, None, 3)
        lines = cv2.HoughLines(img_src, self.h_r, np.pi / 180, 50,
                               None, 0, 0)  # Hough line detector

        # Get the 4 best lines that define the square
        hor_angle = 90
        vert_angle = [180, 0]
        hor_lines = []
        vert_lines = []
        rh_ant = -100
        rv_ant = -100
        if lines is not None and len(lines) >= 4:
            for i in range(len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                if abs((theta / np.pi * 180) - hor_angle) < 20 and abs(rh_ant - rho) > 20 and len(hor_lines) < 2:
                    hor_lines.append([pt1, pt2])
                    rh_ant = rho
                    cv2.line(color, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                elif any(abs((theta / np.pi * 180) - vert_angle)) < 20 and abs(rv_ant - rho) > 20 and len(vert_lines) < 2:
                    rv_ant = rho
                    vert_lines.append([pt1, pt2])
                    cv2.line(color, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    rospy.logwarn("Not the correct line!")
                if len(hor_lines) == len(vert_lines) == 2:
                    break
        else:
            rospy.logwarn("Not sufficient lines detected!")
            return None

        # Check if the corrected lines corrected
        if not (len(hor_lines) == len(vert_lines) == 2):
            rospy.logwarn("Not the correct lines detected!")
            return None

        # Get the intersecction points
        self.int_points = []
        for h in hor_lines:
            A = h[0]
            B = h[1]
            a1 = B[1] - A[1]
            b1 = A[0] - B[0]
            c1 = a1*(A[0]) + b1*(A[1])

            for v in vert_lines:
                C = v[0]
                D = v[1]
                a2 = D[1] - C[1]
                b2 = C[0] - D[0]
                c2 = a2*(C[0]) + b2*(C[1])
                determinant = a1*b2 - a2*b1
                if (determinant == 0):
                    raise Exception(
                        "Lines are paralel! Check Hough function parameters")
                else:
                    x = (b2*c1 - b1*c2)/determinant
                    y = (a1*c2 - a2*c1)/determinant
                    self.int_points.append((int(x), int(y)))
        self.int_points.sort()  # Sort the points based on x coordinate
        rospy.loginfo("The detected points are: ")
        rospy.loginfo(self.int_points)

        # Destinatary points
        o = 0
        for i in range(0, len(self.int_points), 2):
            if self.int_points[i][1] > self.int_points[i+1][1]:
                self.dst_points.append((o, self.size))
                self.dst_points.append((o, 0))
            else:
                self.dst_points.append((o, 0))
                self.dst_points.append((o, self.size))
            o += self.size
        rospy.loginfo("The destiny points are: ")
        rospy.loginfo(self.dst_points)

        # Draw the points on the source image
        for pt in self.int_points:
            color = cv2.circle(color, pt, radius=4,
                               color=(255, 0, 0), thickness=-1)
        return color

    def get_normalize(self, img_src):
        """
        This function perfoms homography based on previous calculated corners.
        It returns the destiny image.
        """
        # Calculate Homography
        h, _ = cv2.findHomography(
            np.array(self.int_points), np.array(self.dst_points))

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(
            img_src, h, (self.img_dst.shape[1], self.img_dst.shape[0]))

        return im_out
