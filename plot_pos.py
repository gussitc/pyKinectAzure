# %%
import numpy as np
import matplotlib.pyplot as plt
from rotation import closest_rotation_matrix
import json
import glob

calibration_folder = 'data/calibration/cal_0001/'
tracking_folder = 'data/tracking/track_0002/'

# emblobot in room coordinates
c0_room = np.array([[1.846738, 2.1159978, 1.4999355]]).T * 1000

# %%
def reject_outliers(data, m=2.):
    d = np.abs(data - np.mean(data, axis=0))
    d_norm = np.linalg.norm(d, axis=1)
    outlier_idxs = np.where(d_norm > m * np.mean(d_norm))
    return np.delete(data, outlier_idxs, axis=0)

# Load calibration data for all cameras
calibration_files = glob.glob(f"{calibration_folder}/calibration_data_cam*.json")

camera_calibrations = {}
for file in calibration_files:
    with open(file, 'r') as f:
        calibration_data = json.load(f)
    cam_id = file.split('_cam')[-1].split('.json')[0]  # Extract camera ID from filename
    camera_calibrations[cam_id] = {
        "c0": np.array([calibration_data[i]["c0"] for i in range(len(calibration_data))]),
        "x_vec": np.array([calibration_data[i]["x_vec"] for i in range(len(calibration_data))]),
        "y_vec": np.array([calibration_data[i]["y_vec"] for i in range(len(calibration_data))]),
        "z_vec": np.array([calibration_data[i]["z_vec"] for i in range(len(calibration_data))]),
    }

#%%
# Reject outliers for calibration data
for cam_id, calib in camera_calibrations.items():
    calib["c0"] = reject_outliers(calib["c0"])
    calib["x_vec"] = reject_outliers(calib["x_vec"])
    calib["y_vec"] = reject_outliers(calib["y_vec"])
    calib["z_vec"] = reject_outliers(calib["z_vec"])

# %%
# Load tracking data for all cameras
tracking_files = glob.glob(f"{tracking_folder}/track_data_cam*.json")

camera_data = {}
camera_ids = [file.split('_cam')[-1].split('.json')[0] for file in tracking_files]  # Extract camera IDs

for file in tracking_files:
    with open(file, 'r') as f:
        cam_id = file.split('_cam')[-1].split('.json')[0]
        camera_data[cam_id] = json.load(f)

# Helper functions
def get_joint_positions(json_data, joint_id):
    return np.array(
        [
            (
                json_data[i]["joints"][joint_id]["position"]["v"]
                if json_data[i]["num_bodies"] > 0
                else [0, 0, 0]
            )
            for i in range(len(json_data))
        ]
    )

def get_joint_confidences(json_data, joint_id):
    return np.array(
        [
            (
                json_data[i]["joints"][joint_id]["confidence_level"]
                if json_data[i]["num_bodies"] > 0
                else 0
            )
            for i in range(len(json_data))
        ]
    )

def get_timestamps(json_data):
    return np.array(
        [
            json_data[i]["utc_timestamp_ns"]
            for i in range(len(json_data))
        ]
    )

# Process joint data for all cameras
joint_id = 26  # Example: head_joint_id
camera_positions = {}
camera_confidences = {}
camera_timestamps = {}
camera_tracks = {}

for cam_id, json_data in camera_data.items():
    camera_positions[cam_id] = get_joint_positions(json_data, joint_id)
    camera_confidences[cam_id] = get_joint_confidences(json_data, joint_id)
    camera_timestamps[cam_id] = get_timestamps(json_data)
    camera_tracks[cam_id] = np.array([json_data[i]["num_bodies"] > 0 for i in range(len(json_data))])

# %%
# Compute mean calibration vectors and rotation matrices for all cameras
camera_means = {}
camera_rotations = {}

for cam_id, calib in camera_calibrations.items():
    c_mean = np.mean(calib["c0"], axis=0)
    x_vec_mean = np.mean(calib["x_vec"], axis=0)
    y_vec_mean = np.mean(calib["y_vec"], axis=0)
    z_vec_mean = np.mean(calib["z_vec"], axis=0)
    R = np.array([x_vec_mean, y_vec_mean, z_vec_mean]).T
    R_SVD = closest_rotation_matrix(R)
    camera_means[cam_id] = c_mean
    camera_rotations[cam_id] = R_SVD

# Transform positions to room coordinates
camera_room_positions = {}

for cam_id, positions in camera_positions.items():
    R_SVD = camera_rotations[cam_id]
    c_mean = camera_means[cam_id]
    room_positions = R_SVD.T @ (positions - c_mean).T
    room_positions = room_positions + c0_room
    room_positions /= 1000  # Convert to meters
    camera_room_positions[cam_id] = room_positions.T

# %%
# Handle missing tracks by replacing with closest timestamps from other cameras
def get_closest_timestamp_index(timestamp, target_timestamp):
    closest_index = (np.abs(timestamp - target_timestamp)).argmin()
    return closest_index

# TODO: Implement the logic to fill in missing camera positions
# for i in range(len(camera_ids)):
#     cam_id = camera_ids[i]
#     for j in range(len(camera_positions[cam_id])):
#         if not camera_tracks[cam_id][j]:
#             for other_cam_id in camera_ids:
#                 if other_cam_id != cam_id and camera_tracks[other_cam_id][j]:
#                     closest_idx = get_closest_timestamp_index(
#                         camera_timestamps[other_cam_id], camera_timestamps[cam_id][j]
#                     )
#                     camera_positions[cam_id][j] = camera_positions[other_cam_id][closest_idx]
#                     break

# %%
# Plot data for all cameras
fig, axs = plt.subplots(4, 1, figsize=(10, 10))

reference_timestamp = camera_timestamps[camera_ids[0]][0]
for cam_id in camera_ids:
    timestamps = (camera_timestamps[cam_id] - reference_timestamp) / 1e6
    positions = camera_room_positions[cam_id]
    axs[0].plot(timestamps, positions[:, 0], label=f'cam{cam_id} x')
    axs[1].plot(timestamps, positions[:, 1], label=f'cam{cam_id} y')
    axs[2].plot(timestamps, positions[:, 2], label=f'cam{cam_id} z')
    axs[3].plot(timestamps, camera_tracks[cam_id], label=f'cam{cam_id} track')

axs[0].set_ylabel('X pos [m]')
axs[1].set_ylabel('Y pos [m]')
axs[2].set_ylabel('Z pos [m]')
axs[3].set_ylabel('Has track')
axs[3].set_xlabel('Time [s]')

for ax in axs:
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()
