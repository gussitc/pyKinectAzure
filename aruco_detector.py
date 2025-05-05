import cv2
import numpy as np
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t, k4a_float3_t
import matplotlib.pyplot as plt
from rotation import closest_rotation_matrix

class ArucoDetector:
	def __init__(self):
		aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
		parameters = cv2.aruco.DetectorParameters()

		self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

	def get_aruco_box_frame(self, x_face, y_face, y_dist, z_dist, x_mirror=False, y_mirror=False):
		x_vec = (x_face[1] - x_face[0] + x_face[2] - x_face[3]) * 0.5
		x_norm = np.linalg.norm(x_vec)
		x_vec_norm = x_vec/x_norm
		c0 = x_face[0] + x_vec * 0.5
		if x_mirror:
			x_vec_norm = -x_vec_norm

		y_vec = (y_face[0] - y_face[1] + y_face[3] - y_face[2]) * 0.5
		if y_mirror:
			y_vec = -y_vec
		y_norm = np.linalg.norm(y_vec)
		y_vec_norm = y_vec/y_norm
		c0 = c0 + y_vec_norm * y_dist

		z_vec = (x_face[0] - x_face[3] + x_face[1] - x_face[2] + y_face[0] - y_face[3] + y_face[1] - y_face[2]) * 0.25
		z_norm = np.linalg.norm(z_vec)
		z_vec_norm = z_vec/z_norm
		c0 = c0 + z_vec_norm * z_dist

		return c0, x_vec_norm, y_vec_norm, z_vec_norm

	def get_reference_frame(self, calibration, ids, corners, depth_image):
		if ids is not None:
			corner_dict_3d = {}
			for j in range(len(corners)):
				transformed_corners_3d = []
				for i in range(corners[j].shape[1]):
					pix = corners[j][0][i].astype(int)
					pix_x = pix[0]
					pix_y = pix[1]
					pixels = k4a_float2_t((pix_x, pix_y))

					c3 = calibration.convert_2d_to_3d(pixels, depth_image[pix_y, pix_x], K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
					transformed_corners_3d.append(np.array([c3.xyz.x, c3.xyz.y, c3.xyz.z]))

				corner_dict_3d[ids[j][0]] = transformed_corners_3d

			c0 = None
			x_face = None
			y_face = None
			x_vec_norm = None
			y_vec_norm = None
			z_vec_norm = None
			y_length = 150
			y_offset = 25
			z_dist = 85
			if 8 in corner_dict_3d and 9 in corner_dict_3d:
				y_dist = 0 - y_offset
				x_face = corner_dict_3d[8]
				y_face = corner_dict_3d[9]
				y_mirror = False
				x_mirror = False
			elif 8 in corner_dict_3d and 11 in corner_dict_3d:
				y_dist = 0 - y_offset
				x_face = corner_dict_3d[8]
				y_face = corner_dict_3d[11]
				x_mirror = False
				y_mirror = True
			elif 10 in corner_dict_3d and 11 in corner_dict_3d:
				y_dist = y_length - y_offset
				x_face = corner_dict_3d[10]
				y_face = corner_dict_3d[11]
				x_mirror = True
				y_mirror = True
			elif 10 in corner_dict_3d and 9 in corner_dict_3d:
				y_dist = y_length - y_offset
				x_face = corner_dict_3d[10]
				y_face = corner_dict_3d[9]
				x_mirror = True
				y_mirror = False

			if x_face is not None and y_face is not None:
				c0, x_vec_norm, y_vec_norm, z_vec_norm = self.get_aruco_box_frame(x_face, y_face, y_dist, z_dist, x_mirror, y_mirror)
			
			return c0, x_vec_norm, y_vec_norm, z_vec_norm
		return None, None, None, None

	def draw_coordinate_frame(self, image, calibration, c0, x_vec_norm, y_vec_norm, z_vec_norm):
		R = np.array([x_vec_norm, y_vec_norm, z_vec_norm]).T
		try:
			R_SVD = closest_rotation_matrix(R)
			R = R_SVD
		except:
			print("SVD did not converge")

		s = 150 # 15 cm

		# coordinate frame in emblo coordinates
		p0_e = np.array([s, 0, 0])
		p1_e = np.array([0, s, 0])
		p2_e = np.array([0, 0, s])

		p0 = R @ p0_e + c0
		p1 = R @ p1_e + c0
		p2 = R @ p2_e + c0

		p0_2d = calibration.convert_3d_to_2d(k4a_float3_t((p0[0], p0[1], p0[2])), K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)
		p1_2d = calibration.convert_3d_to_2d(k4a_float3_t((p1[0], p1[1], p1[2])), K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)
		p2_2d = calibration.convert_3d_to_2d(k4a_float3_t((p2[0], p2[1], p2[2])), K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)

		pixel = k4a_float3_t((c0[0], c0[1], c0[2]))
		c0_2d = calibration.convert_3d_to_2d(pixel, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR)

		if not np.isnan(c0_2d.xy.x) and not np.isnan(c0_2d.xy.y):
			image = cv2.circle(image, (int(c0_2d.xy.x), int(c0_2d.xy.y)), 10, (0, 255, 0), -1)

			try:
				image = cv2.line(image, (int(p0_2d.xy.x), int(p0_2d.xy.y)), (int(c0_2d.xy.x), int(c0_2d.xy.y)), (0, 0, 255), 2)
				image = cv2.line(image, (int(p1_2d.xy.x), int(p1_2d.xy.y)), (int(c0_2d.xy.x), int(c0_2d.xy.y)), (0, 255, 0), 2)
				image = cv2.line(image, (int(p2_2d.xy.x), int(p2_2d.xy.y)), (int(c0_2d.xy.x), int(c0_2d.xy.y)), (255, 0, 0), 2)
			except:
				pass
		return image


	def detect(self, calibration, color_image, depth_image):
		gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
		corners, ids, rejected = self.detector.detectMarkers(gray)

		c0, x_vec_norm, y_vec_norm, z_vec_norm = self.get_reference_frame(calibration, ids, corners, depth_image)

		cv2.aruco.drawDetectedMarkers(image, corners, ids)

		if c0 is not None and not np.isnan(c0[0]):
			image = self.draw_coordinate_frame(image, calibration, c0, x_vec_norm, y_vec_norm, z_vec_norm)
		
		return image, c0, x_vec_norm, y_vec_norm, z_vec_norm