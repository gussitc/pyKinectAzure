# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:29:09 2024

@author: 28731
"""

import pykinect_azure as pykinect
import pykinect_azure.k4a._k4a as k4a
import time
import cv2

# Start single camera
def start_camera(device_info):
    
    device = device_info['device']
    device.start_cameras(device_info['config'])
    print(
        f"Successfully started camera for device {device_info['index']} ({device_info['type']})")
    
# Close all the devices
def close_devices(devices):
        
    for device_info in devices:
        device_info['device'].close()
        
        
if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)
    
    # A list to store the devices
    devices = []
    
    # The number of your devices
    num_devices = pykinect.k4a_device_get_installed_count() 
    
    # Modify camera configuration and start devices
    for i in range(num_devices):
        device = pykinect.Device(i)
        device_config, device_type = device.device_configinit()
        device_config.depth_mode = k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.color_resolution = k4a.K4A_COLOR_RESOLUTION_720P
        device_config.camera_fps = k4a.K4A_FRAMES_PER_SECOND_5
        bodyTracker = None
        devices.append({
            'device': device,
            'bodyTracker': bodyTracker,
            'type': device_type,
            'config': device_config,
            'index': i,
            'rgb_image': None})
        
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
    
    for i in range(num_devices):
        devices[i]['bodyTracker'] = pykinect.start_body_tracker(devices[i]['device'])

    video_writers = []
    frame_width = 512
    frame_height = 512
    fps = 5

    for i in range(num_devices):
        video_writer = cv2.VideoWriter(
            f'output_device_{i}.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (frame_width, frame_height)
        )
        video_writers.append(video_writer)

    frame_count = 0
    start_time = time.time()

    while True:

        for device_info in devices:
            device = device_info['device']
            bodyTracker = device_info['bodyTracker']
            capture = device.update()
            body_frame = bodyTracker.update(device)
            ret_color, color_image = capture.get_color_image()
            if not ret_color:
                continue

            ret_depth, depth_image = capture.get_colored_depth_image()
            ret_body, body_image_color = body_frame.get_segmentation_image()

            device_info['rgb_image'] = color_image
            device_info['depth_image'] = depth_image
            device_info['body_image_color'] = body_image_color
            device_info['body_frame'] = body_frame
        
        for i in range(num_devices):          
            # Combine both images
            depth_color_image = devices[i]['depth_image']
            body_image_color = devices[i]['body_image_color']
            body_frame = devices[i]['body_frame']
            combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

            # Draw the skeletons
            combined_image = body_frame.draw_bodies(combined_image)

            # Write the combined image to the video file
            video_writers[i].write(combined_image)

            # Display the images
            cv2.imshow(f"Depth Image_{i}", combined_image)
            # cv2.imshow(f"Body Image_{i}", body_image_color)

        frame_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            actual_fps = frame_count / elapsed_time
            print(f"Actual FPS: {actual_fps:.2f}")
            frame_count = 0
            start_time = time.time()
        
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release video writers
    for writer in video_writers:
        writer.release()

    close_devices(devices)
