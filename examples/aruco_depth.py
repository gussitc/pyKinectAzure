#%%
import pykinect_azure as pykinect
from aruco_detector import ArucoDetector
import cv2
import numpy as np

USE_PLAYBACK = True

def main():

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

	aruco_detector = ArucoDetector()

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

		ret, color_image = capture.get_transformed_color_image()

		if not ret:
			continue

		ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

		_, image = capture.get_color_image()
		image = image[:,:,:3]
		image, c0, x_vec_norm, y_vec_norm, z_vec_norm = aruco_detector.detect(device.calibration, image, transformed_depth_image)

		emblo_pos.append(c0)
		x_vecs.append(x_vec_norm)
		y_vecs.append(y_vec_norm)
		z_vecs.append(z_vec_norm)

		scale = 0.7
		image = cv2.resize(image, (0,0), fx=scale, fy=scale)
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

if __name__ == "__main__":
	main()