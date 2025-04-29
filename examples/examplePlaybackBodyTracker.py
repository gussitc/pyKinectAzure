import cv2
import numpy as np
import pykinect_azure as pykinect

if __name__ == "__main__":

	# video_filename = "calibration.mkv"
	video_filename = "output.mkv"
	# video_filename = "yz.mkv"

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

	kinect_pos = []

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

		pelvis_3d = body_frame.get_body_skeleton().joints[0].position
		pelvis_2d = body_frame.calibration.convert_3d_to_2d(pelvis_3d, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		combined_image = cv2.circle(combined_image, (int(pelvis_2d.xy.x), int(pelvis_2d.xy.y)), 5, (0, 255, 0), -1)

		naval_3d = body_frame.get_body_skeleton().joints[1].position
		naval_2d = body_frame.calibration.convert_3d_to_2d(naval_3d, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		combined_image = cv2.circle(combined_image, (int(naval_2d.xy.x), int(naval_2d.xy.y)), 5, (0, 0, 255), -1)

		head_3d = body_frame.get_body_skeleton().joints[26].position
		head_2d = body_frame.calibration.convert_3d_to_2d(head_3d, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		combined_image = cv2.circle(combined_image, (int(head_2d.xy.x), int(head_2d.xy.y)), 5, (255, 0, 0), -1)

		# foot_3d = body_frame.get_body_skeleton().joints[21].position
		# foot_2d = body_frame.calibration.convert_3d_to_2d(foot_3d, pykinect.K4A_CALIBRATION_TYPE_DEPTH, pykinect.K4A_CALIBRATION_TYPE_DEPTH)
		# combined_image = cv2.circle(combined_image, (int(foot_2d.xy.x), int(foot_2d.xy.y)), 5, (255, 255, 0), -1)

		print("naval_3d: ", naval_3d)

		R = np.array([[ 0.65830269, -0.75175884,  0.03868087],
					  [ 0.16960551,  0.09806402, -0.98062094],
					  [-0.73339725, -0.65210589, -0.19205825]])

		# emblo pos in room coordinates
		c_r = np.array([1.336217 , 2.7460349, 1.1113696]) * 1000

		# emblo pos in camera frame
		c_k = np.array([-118.62810442, 301.21279806, 1652.79677597])

		# c_r = np.array([0, 0, 0])

		naval_k = np.array([naval_3d.xyz.x, naval_3d.xyz.y, naval_3d.xyz.z])
		naval_r = R.T @ (naval_k - c_k)
		naval_r[0] = -naval_r[0]
		naval_r += c_r
		naval_r /= 1000
		print("naval_r: ", naval_r)
		kinect_pos.append(naval_r)

		# head_k = np.array([head_3d.xyz.x, head_3d.xyz.y, head_3d.xyz.z])
		# head_r = R.T @ head_k + c_r
		# print("head_r: ",  head_r/1000)

		# foot_k = np.array([foot_3d.xyz.x, foot_3d.xyz.y, foot_3d.xyz.z])
		# foot_r = R.T @ foot_k + c_r
		# print("foot_r: ",  foot_r/1000)

		# Overlay body segmentation on depth image
		cv2.imshow('Depth image with skeleton',combined_image)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):
			break
	
	# np.save("kinect_pos.npy", kinect_pos)