import cv2

import pykinect_azure as pykinect

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Start playback
    playback_cam0 = pykinect.start_playback('output_0.mkv')
    playback_cam1 = pykinect.start_playback('output_1.mkv')

    # playback_config = playback_cam0.get_record_configuration()
    # print(playback_config)

    cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)
    while True:

        # Get camera capture
        ret0, capture0 = playback_cam0.update()
        ret1, capture1 = playback_cam1.update()

        if not ret0 or not ret1:
            break

        # Get color image
        ret_color, color_image = capture0.get_transformed_color_image()
        ret_color1, color_image1 = capture1.get_transformed_color_image()

        # Get the colored depth
        ret_depth, depth_color_image = capture0.get_colored_depth_image()
        ret_color1, depth_color_image1 = capture1.get_colored_depth_image()

        if not ret_color or not ret_depth:
            continue

        # Plot the image
        combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_color_image, 0.3, 0)
        cv2.imshow('Depth Image', combined_image)

        combined_image1 = cv2.addWeighted(color_image1[:, :, :3], 0.7, depth_color_image1, 0.3, 0)
        cv2.imshow('Depth Image1', combined_image1)

        # Press q key to stop
        if cv2.waitKey(30) == ord('q'):
            break
