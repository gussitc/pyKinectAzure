#%%
import cv2
import pykinect_azure as pykinect
import numpy as np
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t, k4a_float3_t
import matplotlib.pyplot as plt
from rotation import closest_rotation_matrix

USE_PLAYBACK = False

def get_aruco_box_frame(x_face, y_face, y_dist, z_dist, y_mirror=False):

	# in the middle of the marker 8
	x_vec = (x_face[1] - x_face[0] + x_face[2] - x_face[3]) * 0.5
	x_norm = np.linalg.norm(x_vec)
	x_vec_norm = x_vec/x_norm
	c0 = x_face[0] + x_vec * 0.5

	# depth is 2.5 cm
	y_vec = (y_face[0] - y_face[1] + y_face[3] - y_face[2]) * 0.5
	if y_mirror:
		y_vec = -y_vec
	y_norm = np.linalg.norm(y_vec)
	y_vec_norm = y_vec/y_norm
	c0 = c0 + y_vec_norm * y_dist

	# height is 8.5 cm
	z_vec = (x_face[0] - x_face[3] + x_face[1] - x_face[2] + y_face[0] - y_face[3] + y_face[1] - y_face[2]) * 0.25
	z_norm = np.linalg.norm(z_vec)
	z_vec_norm = z_vec/z_norm
	c0 = c0 + z_vec_norm * z_dist

	return c0, x_vec_norm, y_vec_norm, z_vec_norm

if __name__ == "__main__":
# if True:

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()

	if USE_PLAYBACK:
		video_filename = "calibration.mkv"

		# Initialize the library, if the library is not found, add the library path as argument
		pykinect.initialize_libraries()

		# Start playback
		device = pykinect.start_playback(video_filename)

		playback_config = device.get_record_configuration()
	else:
		# Modify camera configuration
		device_config = pykinect.default_configuration
		device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
		# device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
		device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
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
	emblo_pos = []
	x_vecs = []
	y_vecs = []
	z_vecs = []
	while True:
		
		# Get capture
		if USE_PLAYBACK:
			ret, capture = device.update()
			if not ret:
				break
		else:
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

			c0 = None
			if 8 in corner_dict_3d and 9 in corner_dict_3d:
				y_dist = -25
				z_dist = 85

				c0, x_vec_norm, y_vec_norm, z_vec_norm = get_aruco_box_frame(corner_dict_3d[8], corner_dict_3d[9], y_dist, z_dist)
			elif 8 in corner_dict_3d and 11 in corner_dict_3d:
				y_dist = -25
				z_dist = 85

				c0, x_vec_norm, y_vec_norm, z_vec_norm = get_aruco_box_frame(corner_dict_3d[8], corner_dict_3d[11], y_dist, z_dist, y_mirror=True)

			if c0 is not None:
				emblo_pos.append(c0)
				x_vecs.append(x_vec_norm)
				y_vecs.append(y_vec_norm)
				z_vecs.append(z_vec_norm)

				R = np.array([x_vec_norm, y_vec_norm, z_vec_norm]).T
				R = closest_rotation_matrix(R)

				s = 150 # 20 cm

				# coordinate frame in emblo coordinates
				p0_e = np.array([s, 0, 0])
				p1_e = np.array([0, s, 0])
				p2_e = np.array([0, 0, s])

				p0 = R @ p0_e + c0
				p1 = R @ p1_e + c0
				p2 = R @ p2_e + c0

				p0_2d = device.calibration.convert_3d_to_2d(k4a_float3_t((p0[0], p0[1], p0[2])), K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)
				p1_2d = device.calibration.convert_3d_to_2d(k4a_float3_t((p1[0], p1[1], p1[2])), K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)
				p2_2d = device.calibration.convert_3d_to_2d(k4a_float3_t((p2[0], p2[1], p2[2])), K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)

				pixel = k4a_float3_t((c0[0], c0[1], c0[2]))
				c0_2d = device.calibration.convert_3d_to_2d(pixel, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)

				try:
					image = cv2.circle(image, (int(c0_2d.xy.x), int(c0_2d.xy.y)), 10, (0, 255, 0), -1)

					image = cv2.line(image, (int(p0_2d.xy.x), int(p0_2d.xy.y)), (int(c0_2d.xy.x), int(c0_2d.xy.y)), (0, 0, 255), 2)
					image = cv2.line(image, (int(p1_2d.xy.x), int(p1_2d.xy.y)), (int(c0_2d.xy.x), int(c0_2d.xy.y)), (0, 255, 0), 2)
					image = cv2.line(image, (int(p2_2d.xy.x), int(p2_2d.xy.y)), (int(c0_2d.xy.x), int(c0_2d.xy.y)), (255, 0, 0), 2)
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

		scale = 0.7
		image = cv2.resize(image, (0,0), fx=scale, fy=scale)
		cv2.imshow('Depth Image',depth_image)
		cv2.imshow('Color Image',color_image)
		cv2.imshow('ArUco Image',image)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):
			break

if USE_PLAYBACK:
	emblo_pos_avg = np.average(emblo_pos, axis=0)
	x_vec_avg = np.average(x_vecs, axis=0)
	y_vec_avg = np.average(y_vecs, axis=0)
	z_vec_avg = np.average(z_vecs, axis=0)

	print("emblo_pos_avg: ", emblo_pos_avg)
	print("x_vec_avg: ", x_vec_avg)
	print("y_vec_avg: ", y_vec_avg)
	print("z_vec_avg: ", z_vec_avg)

	np.savez('kinect_calibration', emblo_pos=emblo_pos_avg, x_vec=x_vec_avg, y_vec=y_vec_avg, z_vec=z_vec_avg)

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