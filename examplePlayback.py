import cv2
from aruco_detector import ArucoDetector
import pykinect_azure as pykinect
import numpy as np

if __name__ == "__main__":

    aruco_detector = ArucoDetector()

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Start playback
    playback_cam0 = pykinect.start_playback('calib_0.mkv')
    playback_cam1 = pykinect.start_playback('calib_1.mkv')

    # playback_config = playback_cam0.get_record_configuration()
    # print(playback_config)

    c0_list = []
    c1_list = []
    x_vec_list = []
    y_vec_list = []
    z_vec_list = []
    x_vec1_list = []
    y_vec1_list = []
    z_vec1_list = []

    # cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)
    while True:

        # Get camera capture
        ret0, capture0 = playback_cam0.update()
        ret1, capture1 = playback_cam1.update()

        if not ret0 or not ret1:
            break

        # Get color image
        ret_color, color_image = capture0.get_color_image()
        ret_color1, color_image1 = capture1.get_color_image()

        # Get the colored depth
        ret_depth, depth_color_image = capture0.get_colored_depth_image()
        ret_color1, depth_color_image1 = capture1.get_colored_depth_image()

        if not ret_color or not ret_depth:
            continue

        ret_depth, depth_image = capture0.get_transformed_depth_image()
        ret_depth1, depth_image1 = capture1.get_transformed_depth_image()

        color_image = color_image[:, :, :3]
        color_image1 = color_image1[:, :, :3]

        color_image, c0, x_vec, y_vec, z_vec = aruco_detector.detect(playback_cam0.calibration, color_image, depth_image)
        color_image1, c1, x_vec1, y_vec1, z_vec1 = aruco_detector.detect(playback_cam1.calibration, color_image1, depth_image1)

        # append if not None
        if c0 is not None:
            c0_list.append(c0)
            x_vec_list.append(x_vec)
            y_vec_list.append(y_vec)
            z_vec_list.append(z_vec)
        if c1 is not None:
            c1_list.append(c1)
            x_vec1_list.append(x_vec1)
            y_vec1_list.append(y_vec1)
            z_vec1_list.append(z_vec1)

        scale = 0.25
        color_image = cv2.resize(color_image, (0, 0), fx=scale, fy=scale)
        color_image1 = cv2.resize(color_image1, (0, 0), fx=scale, fy=scale)

        cv2.imshow('Color Image', color_image)
        cv2.imshow('Color Image1', color_image1)

        # Plot the image
        # combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_color_image, 0.3, 0)
        # cv2.imshow('Depth Image', combined_image)

        # combined_image1 = cv2.addWeighted(color_image1[:, :, :3], 0.7, depth_color_image1, 0.3, 0)
        # cv2.imshow('Depth Image1', combined_image1)

        # Press q key to stop
        if cv2.waitKey(30) == ord('q'):
            break

    # replace None with 0
    # for i in range(len(c0_list)):
    #     if c0_list[i] is None:
    #         c0_list[i] = np.array([0, 0, 0])
    #     if c1_list[i] is None:
    #         c1_list[i] = np.array([0, 0, 0])
    #     if x_vec_list[i] is None:
    #         x_vec_list[i] = np.array([0, 0, 0])
    #     if y_vec_list[i] is None:
    #         y_vec_list[i] = np.array([0, 0, 0])
    #     if z_vec_list[i] is None:
    #         z_vec_list[i] = np.array([0, 0, 0])
    #     if x_vec1_list[i] is None:
    #         x_vec1_list[i] = np.array([0, 0, 0])
    #     if y_vec1_list[i] is None:
    #         y_vec1_list[i] = np.array([0, 0, 0])
    #     if z_vec1_list[i] is None:
    #         z_vec1_list[i] = np.array([0, 0, 0])
    
    np.savez("kinect_calibration", c0=c0_list, c1=c1_list, x_vec=x_vec_list, y_vec=y_vec_list, z_vec=z_vec_list, x_vec1=x_vec1_list, y_vec1=y_vec1_list, z_vec1=z_vec1_list)