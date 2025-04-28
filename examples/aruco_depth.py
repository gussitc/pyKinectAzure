#%%
import cv2
import pykinect_azure as pykinect
import numpy as np
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t, k4a_float3_t
import matplotlib.pyplot as plt

if __name__ == "__main__":
# if True:

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
			corner_dict_2d = {}
			corner_dict_3d = {}
			for j in range(len(corners)):
				transformed_corners_2d = []
				transformed_corners_3d = []
				for i in range(corners[j].shape[1]):
					pix = corners[j][0][i].astype(int)
					pix_x = pix[0]
					pix_y = pix[1]
					pixels = k4a_float2_t((pix_x, pix_y))
					c2 = device.calibration.convert_2d_to_2d(pixels, transformed_depth_image[pix_y, pix_x], K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
					transformed_corners_2d.append(cv2.KeyPoint(c2.xy.x, c2.xy.y, 1))

					c3 = device.calibration.convert_2d_to_3d(pixels, transformed_depth_image[pix_y, pix_x], K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
					transformed_corners_3d.append(np.array([c3.xyz.x, c3.xyz.y, c3.xyz.z]))

				corner_dict_2d[ids[j][0]] = transformed_corners_2d
				corner_dict_3d[ids[j][0]] = transformed_corners_3d

			cv2.aruco.drawDetectedMarkers(image, corners, ids)
			
			if 8 in corner_dict_2d:
				cv2.drawKeypoints(color_image, corner_dict_2d[8], color_image, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			if 9 in corner_dict_2d:
				cv2.drawKeypoints(color_image, corner_dict_2d[9], color_image, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

			if 8 in corner_dict_3d and 9 in corner_dict_3d:
				crn8 = corner_dict_3d[8]
				vec0 = (crn8[1] - crn8[0] + crn8[2] - crn8[3]) * 0.5
				c0 = crn8[0] + vec0 * 0.5

				# depth is 2.5 cm
				crn9 = corner_dict_3d[9]
				vec1 = (crn9[1] - crn9[0] + crn9[2] - crn9[3]) * 0.5
				len1 = np.linalg.norm(vec1)
				c0 = c0 + vec1/len1 * 30

				# height is 8.5 cm
				vec2 = (crn8[0] - crn8[3] + crn8[1] - crn8[2]) * 0.5
				len2 = np.linalg.norm(vec2)
				c0 = c0 + vec2/len2 * 85

				pixel = k4a_float3_t((c0[0], c0[1], c0[2]))
				c0_2d = device.calibration.convert_3d_to_2d(pixel, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)

				try:
					image = cv2.circle(image, (int(c0_2d.xy.x), int(c0_2d.xy.y)), 10, (0, 255, 0), -1)
				except:
					print("NaN value in c0_2d")
				
			# plt.imshow(color_image, cmap='gray')
			# plt.show()
			# break

			# if 8 in corner_dict_3d and 9 in corner_dict_3d:
			# 	# cv2.imshow('Color Image',color_image)
			# 	# cv2.imshow('ArUco Image',image)

			# 	break

#%%

		# Get the colored depth
		ret, depth_image = capture.get_colored_depth_image()

		# Combine both images
		# combined_image = cv2.addWeighted(color_image[:,:,:3], 0.7, depth_image, 0.3, 0)

		# Overlay body segmentation on depth image
		# cv2.imshow('Transformed Color Image',combined_image)
		# double the image size for better visibility

		color_image = cv2.resize(color_image, (0,0), fx=1.7, fy=1.7)
		cv2.imshow('Depth Image',depth_image)
		cv2.imshow('Color Image',color_image)
		cv2.imshow('ArUco Image',image)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):
			break

#%%
# from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t, k4a_float3_t

# crn8 = corner_dict_3d[8]
# c0 = crn8[0] + (crn8[1] - crn8[0]) * 0.5

# # depth is 2.5 cm
# crn9 = corner_dict_3d[9]
# c0 = c0 + (crn9[1] - crn9[0])/np.linalg.norm(crn9[1] - crn9[0]) * 25

# # height is 8.5 cm
# c0 = c0 + (crn8[0] - crn8[3])/np.linalg.norm(crn8[0] - crn8[3]) * 85

# # c0 = crn[0]

# pixel = k4a_float3_t((c0[0], c0[1], c0[2]))
# c0_2d = device.calibration.convert_3d_to_2d(pixel, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)

# new_image = image.copy()
# new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
# new_image = cv2.circle(new_image, (int(c0_2d.xy.x), int(c0_2d.xy.y)), 15, (0, 255, 0), -1)
# plt.imshow(new_image, cmap='gray')