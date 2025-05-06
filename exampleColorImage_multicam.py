# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:29:09 2024

@author: 28731
"""

import pykinect_azure as pykinect
import pykinect_azure.k4a._k4a as k4a
import pykinect_azure.k4abt._k4abt as k4abt
import time
import cv2
import threading
import json

running = True

def start_camera(device_info):
    device = device_info['device']
    device.start_cameras(device_info['config'])
    print(
        f"Successfully started camera for device {device_info['index']} ({device_info['type']})")

def close_devices(devices):
    for device_info in devices:
        device_info['device'].close()

def process_camera(device_info, video_writer):
    frame_count = 0
    total_frame_count = 0
    start_time = time.time()
    global running

    json_file_path = f"track_data_cam{device_info['index']}.json"
    json_file = open(json_file_path, "w")
    json_file.write("[")
    while running:
        device = device_info['device']
        bodyTracker = device_info['bodyTracker']

        pre_time = time.time_ns()
        capture = device.update()
        utc_timestamp_us = (time.time_ns() + pre_time) // 2000
        device_timestamp_us = k4a.k4a_image_get_device_timestamp_usec(k4a.k4a_capture_get_depth_image(capture.handle()))

        ret_depth, depth_image = capture.get_colored_depth_image()
        if not ret_depth:
            continue

        body_frame = bodyTracker.update(device)
        combined_image = body_frame.draw_bodies(depth_image)

        num_bodies = body_frame.get_num_bodies()
        if num_bodies > 0:
            joints = body_frame.json()[0]['skeleton']['joints']
        else:
            joints = ""

        track_data = {
            "frame": total_frame_count,
            "utc_timestamp_ns": utc_timestamp_us,
            "device_timestamp_us": device_timestamp_us,
            "num_bodies": num_bodies,
            "joints": joints}

        json.dump(track_data, json_file)
        json_file.write("\n,")

        video_writer.write(combined_image)
        cv2.imshow(f"Cam{device_info['index']}", combined_image)

        frame_count += 1
        total_frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            actual_fps = frame_count / elapsed_time
            print(f"Cam{device_info['index']} - Actual FPS: {actual_fps:.2f}")
            frame_count = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            running = False
            break

    # remove the last newline and comma
    json_file.seek(json_file.tell() - 2, 0)
    json_file.truncate()
    json_file.write("]")
    json_file.close()

    video_writer.release()
    device.close()

def main():
    pykinect.initialize_libraries(track_body=True)

    devices = []
    num_devices = pykinect.k4a_device_get_installed_count()

    for i in range(num_devices):
        device = pykinect.Device(i)
        device_config, device_type = device.device_configinit()
        device_config.depth_mode = k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.color_resolution = k4a.K4A_COLOR_RESOLUTION_720P
        device_config.camera_fps = k4a.K4A_FRAMES_PER_SECOND_15
        bodyTracker = None
        devices.append({
            'device': device,
            'bodyTracker': bodyTracker,
            'type': device_type,
            'config': device_config,
            'index': i,
            })

    master_devices = [d for d in devices if d['type'] == 'Master']
    sub_devices = [d for d in devices if d['type'] == 'Sub']
    stan_devices = [d for d in devices if d['type'] == 'Standalone']

    for device_info in stan_devices:
        start_camera(device_info)
    for device_info in sub_devices:
        start_camera(device_info)
    for device_info in master_devices:
        start_camera(device_info)

    if len(master_devices) == 1 and len(sub_devices) == 0:
        close_devices(devices)
        raise Exception(
            "NO Sub device detected but detected Master device, please check the sync cable!")

    elif len(master_devices) > 1:
        close_devices(devices)
        raise Exception(
            "The Master device cannot be more than one, please check the sync cable!")

    elif len(master_devices) == 0 and len(sub_devices) != 0:
        close_devices(devices)
        raise Exception(
            "NO Master device detected but detected Sub device, please check the sync cable!")

    for i in range(num_devices):
        devices[i]['bodyTracker'] = pykinect.start_body_tracker(devices[i]['device'], model_type=k4abt.K4ABT_LITE_MODEL)

    frame_width = 512
    frame_height = 512
    fps = 15

    threads = []
    for i in range(num_devices):
        video_writer = cv2.VideoWriter(
            f'output_device_{i}.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (frame_width, frame_height)
        )
        thread = threading.Thread(target=process_camera, args=(devices[i], video_writer))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    close_devices(devices)


if __name__ == "__main__":
    main()