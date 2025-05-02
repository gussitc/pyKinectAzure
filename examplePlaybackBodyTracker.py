import cv2
import numpy as np
import pykinect_azure as pykinect
import pykinect_azure.k4a._k4a as _k4a

if __name__ == "__main__":

	video_filename = "output.mkv"

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Start playback
	playback0 = pykinect.start_playback('walking_0.mkv')
	playback1 = pykinect.start_playback('walking_1.mkv')


	# playback_config = playback0.get_record_configuration()
	# print(playback_config)

	playback_calibration0 = playback0.get_calibration()
	playback_calibration1 = playback1.get_calibration()

	# Start body tracker
	bodyTracker0 = pykinect.start_body_tracker(calibration=playback_calibration0)
	bodyTracker1 = pykinect.start_body_tracker(calibration=playback_calibration1)

	naval_pos0 = []
	naval_pos1 = []

	naval_confidence0 = []
	naval_confidence1 = []

	timestamp0 = []
	timestamp1 = []

	# cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)
	while True:

		# Get camera capture
		ret, capture = playback0.update()
		ret1, capture1 = playback1.update()

		if not ret:
			break

		# Get body tracker frame
		body_frame = bodyTracker0.update(capture=capture)
		body_frame1 = bodyTracker1.update(capture=capture1)

		# Get color image
		ret_color, color_image = capture.get_transformed_color_image()
		ret_color1, color_image1 = capture1.get_transformed_color_image()

		# Get the colored depth
		ret_depth, depth_color_image = capture.get_colored_depth_image()
		ret_depth1, depth_color_image1 = capture1.get_colored_depth_image()

		# Get the colored body segmentation
		ret_seg, body_image_color = body_frame.get_segmentation_image()
		ret_seg1, body_image_color1 = body_frame1.get_segmentation_image()
		
		if not ret_color or not ret_depth or not ret_seg:
			continue

		device_timestamp = _k4a.k4a_image_get_device_timestamp_usec(_k4a.k4a_capture_get_depth_image(capture.handle()))
		device_timestamp1 = _k4a.k4a_image_get_device_timestamp_usec(_k4a.k4a_capture_get_depth_image(capture1.handle()))
			
		# Combine both images
		combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
		combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

		# Draw the skeletons
		combined_image = body_frame.draw_bodies(combined_image)

		combined_image1 = cv2.addWeighted(depth_color_image1, 0.6, body_image_color1, 0.4, 0)
		combined_image1 = cv2.addWeighted(color_image1[:, :, :3], 0.7, combined_image1, 0.3, 0)

		# Draw the skeletons
		combined_image1 = body_frame1.draw_bodies(combined_image1)

		try:
			naval_3d = body_frame.get_body_skeleton().joints[1].position
			naval_3d_confidence = body_frame.get_body_skeleton().joints[1].confidence_level
			naval_2d = body_frame.calibration.convert_3d_to_2d(naval_3d, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			combined_image = cv2.circle(combined_image, (int(naval_2d.xy.x), int(naval_2d.xy.y)), 5, (0, 0, 255), -1)
			naval_3d = np.array([naval_3d.xyz.x, naval_3d.xyz.y, naval_3d.xyz.z])
		except:
			naval_3d = [0, 0, 0]
			naval_3d_confidence = 0

		try:
			naval_3d1 = body_frame1.get_body_skeleton().joints[1].position
			naval_3d_confidence1 = body_frame1.get_body_skeleton().joints[1].confidence_level
			naval_2d1 = body_frame1.calibration.convert_3d_to_2d(naval_3d1, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
			combined_image1 = cv2.circle(combined_image1, (int(naval_2d1.xy.x), int(naval_2d1.xy.y)), 5, (0, 0, 255), -1)
			naval_3d1 = np.array([naval_3d1.xyz.x, naval_3d1.xyz.y, naval_3d1.xyz.z])
		except:
			naval_3d1 = [0, 0, 0]
			naval_3d_confidence1 = 0

		naval_pos0.append(naval_3d)
		naval_pos1.append(naval_3d1)
		naval_confidence0.append(naval_3d_confidence)
		naval_confidence1.append(naval_3d_confidence1)
		timestamp0.append(device_timestamp)
		timestamp1.append(device_timestamp1)

		# Overlay body segmentation on depth image
		cv2.imshow('Depth image with skeleton',combined_image)
		cv2.imshow('Depth image with skeleton1',combined_image1)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):
			break

	# Save the data to a file
	np.savez('kinect_position', cam0=naval_pos0, cam1=naval_pos1, cam0_confidence=naval_confidence0, cam1_confidence=naval_confidence1, timestamp0=timestamp0, timestamp1=timestamp1)