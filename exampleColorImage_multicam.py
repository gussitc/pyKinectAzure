# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:29:09 2024

@author: 28731
"""

import pykinect_azure as pykinect
import cv2
from aruco_detector import ArucoDetector

record = False

# Start single camera
def start_camera(device_info):
    
    device = device_info['device']
    device.start(device_info['config'], record=record, record_filepath=f"output_{device_info['index']}.mkv")
    print(
        f"Successfully started camera for device {device_info['index']} ({device_info['type']})")
    
# Close all the devices
def close_devices(devices):
        
    for device_info in devices:
        device_info['device'].close()
        
        
if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()
    
    # A list to store the devices
    devices = []
    
    # The number of your devices
    num_devices = pykinect.k4a_device_get_installed_count() 
    
    # Modify camera configuration and start devices
    for i in range(num_devices):
        device = pykinect.Device(i)
        device_config, device_type = device.device_configinit()
        # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        devices.append({
            'device': device,
            'type': device_type,
            'config': device_config,
            'index': i,
            'rgb_image': None,
            'depth_image': None})
        
        # cv2.namedWindow(f'Color Image_{i}',cv2.WINDOW_NORMAL)

	# Start cameras
    master_devices = [d for d in devices if d['type'] == 'Master']
    sub_devices = [d for d in devices if d['type'] == 'Sub']
    stan_devices = [d for d in devices if d['type'] == 'Standalone']

    for device_info in stan_devices:
        start_camera(device_info)
    for device_info in sub_devices:
        start_camera(device_info)
    # Finally open the master camera 
    for device_info in master_devices:
        start_camera(device_info)

    if len(master_devices) == 1 and len(sub_devices) == 0:
        close_devices()
        raise Exception(
            "NO Sub device detected but detected Master device, please check the sync cable!")
        
    elif len(master_devices) > 1:
        close_devices()
        raise Exception(
            "The Master device cannot be more than one, please check the sync cable!")
    
    elif len(master_devices) == 0 and len(sub_devices) != 0:
        close_devices()
        raise Exception(
            "NO Master device detected but detected Sub device, please check the sync cable!")
    
    aruco_detector = ArucoDetector()

    while True:

        for device_info in devices:
            device = device_info['device']
            capture = device.update()
            ret_color, color_image = capture.get_color_image()
            ret_depth, transformed_depth_image = capture.get_transformed_depth_image()
            _, depth_image = capture.get_colored_depth_image()

            if not ret_color:
                continue

            device_info['rgb_image'] = color_image
            device_info['transformed_depth_image'] = transformed_depth_image
            device_info['depth_image'] = depth_image

		
        for i in range(num_devices):          
            # Plot the image
            device_info = devices[i]
            image, _, _, _, _ = aruco_detector.detect(device_info['device'].calibration, device_info['rgb_image'], device_info['transformed_depth_image'])

            scale = 0.25
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            cv2.imshow(f"Color Image_{i}", image)
            cv2.imshow(f"Depth Image_{i}", device_info['depth_image'])
		
		# Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break
        
    close_devices(devices)
