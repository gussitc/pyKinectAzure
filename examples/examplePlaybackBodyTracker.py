import cv2

import pykinect_azure as pykinect

if __name__ == "__main__":

	video_filename = "calibration.mkv"

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Start playback
	playback = pykinect.start_playback(video_filename)

	playback_config = playback.get_record_configuration()
	# print(playback_config)

	playback_calibration = playback.get_calibration()

	# Start body tracker
	bodyTracker = pykinect.start_body_tracker(calibration=playback_calibration)

	cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)
	while True:

		# Get camera capture
		ret, capture = playback.update()

		if not ret:
			break

		# Get body tracker frame
		body_frame = bodyTracker.update(capture=capture)

		# Get color image
		ret_color, color_image = capture.get_transformed_color_image()

		# Get the colored depth
		ret_depth, depth_color_image = capture.get_colored_depth_image()

		# Get the colored body segmentation
		ret_seg, body_image_color = body_frame.get_segmentation_image()
		
		if not ret_color or not ret_depth or not ret_seg:
			continue
			
		# Combine both images
		combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
		combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

		# Draw the skeletons
		combined_image = body_frame.draw_bodies(combined_image)

		pelvis = body_frame.calibration.convert_3d_to_2d(body_frame.get_body_skeleton().joints[0].position, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		combined_image = cv2.circle(combined_image, (int(pelvis.xy.x), int(pelvis.xy.y)), 5, (0, 255, 0), -1)

		naval = body_frame.calibration.convert_3d_to_2d(body_frame.get_body_skeleton().joints[1].position, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		combined_image = cv2.circle(combined_image, (int(naval.xy.x), int(naval.xy.y)), 5, (0, 0, 255), -1)

		# Overlay body segmentation on depth image
		cv2.imshow('Depth image with skeleton',combined_image)

		# Press q key to stop
		if cv2.waitKey(0) == ord('q'):
			break