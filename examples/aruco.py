import numpy as np
import cv2

data = np.load('frame.npz', allow_pickle=True)
color_image = data['color_image']
depth_color_image = data['depth_color_image']
image = color_image[:, :, :3]

# cv2.imshow('Color Image', color_image)
# cv2.waitKey(0)

# image = cv2.imread('aruco.png')
image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)
# Print the detected markers
print("Detected markers:", ids)
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    cv2.imshow('Detected Markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    cv2.imshow('No Markers Detected', image)
    cv2.waitKey(0)