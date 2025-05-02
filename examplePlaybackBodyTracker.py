import cv2

import pykinect_azure as pykinect

if __name__ == "__main__":

	video_filename = "output.mkv"

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries(track_body=True)

	# Start playback
	playback0 = pykinect.start_playback('output_0.mkv')
	playback1 = pykinect.start_playback('output_1.mkv')


	# playback_config = playback0.get_record_configuration()
	# print(playback_config)

	playback_calibration0 = playback0.get_calibration()
	playback_calibration1 = playback1.get_calibration()

	# Start body tracker
	bodyTracker0 = pykinect.start_body_tracker(calibration=playback_calibration0)
	bodyTracker1 = pykinect.start_body_tracker(calibration=playback_calibration1)

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
			
		# Combine both images
		combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
		combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

		# Draw the skeletons
		combined_image = body_frame.draw_bodies(combined_image)

		combined_image1 = cv2.addWeighted(depth_color_image1, 0.6, body_image_color1, 0.4, 0)
		combined_image1 = cv2.addWeighted(color_image1[:, :, :3], 0.7, combined_image1, 0.3, 0)

		# Draw the skeletons
		combined_image1 = body_frame1.draw_bodies(combined_image1)

		# Overlay body segmentation on depth image
		cv2.imshow('Depth image with skeleton',combined_image)
		cv2.imshow('Depth image with skeleton1',combined_image1)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):
			break