import cv2
import pykinect_azure as pykinect
import numpy as np
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	# print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	parameters = cv2.aruco.DetectorParameters()

	# Create the ArUco detector
	detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
	# Detect the markers

	# cv2.namedWindow('Transformed Color Image',cv2.WINDOW_NORMAL)
	while True:
		
		# Get capture
		capture = device.update()

		# Get the color image from the capture
		ret, color_image = capture.get_transformed_color_image()

		if not ret:
			continue

		color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

		ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

		_, image = capture.get_color_image()
		image = image[:,:,:3]
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
		corners, ids, rejected = detector.detectMarkers(gray)
		# ret, depth_image = capture.get_depth_image()
		if ids is not None:
			transformed_corners = []
			for j in range(len(corners)):
				for i in range(corners[j].shape[1]):
					pix = corners[j][0][i].astype(int)
					pix_x = pix[0]
					pix_y = pix[1]
					pixels = k4a_float2_t((pix_x, pix_y))
					c = device.calibration.convert_2d_to_2d(pixels, transformed_depth_image[pix_y, pix_x], K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
					transformed_corners.append(cv2.KeyPoint(c.xy.x, c.xy.y, 1))
			cv2.aruco.drawDetectedMarkers(image, corners, ids)
			cv2.drawKeypoints(color_image, transformed_corners, color_image, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		# Get the colored depth

		# Combine both images
		# combined_image = cv2.addWeighted(color_image[:,:,:3], 0.7, depth_image, 0.3, 0)

		# Overlay body segmentation on depth image
		# cv2.imshow('Transformed Color Image',combined_image)
		# double the image size for better visibility

		color_image = cv2.resize(color_image, (0,0), fx=1.7, fy=1.7)
		# cv2.imshow('Depth Image',transformed_depth_image)
		cv2.imshow('Color Image',color_image)
		cv2.imshow('ArUco Image',image)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):
			break