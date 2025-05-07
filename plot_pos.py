# %%
import numpy as np
import matplotlib.pyplot as plt
from rotation import closest_rotation_matrix
import json

# %%
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.mean(data, axis=0))
    d_norm = np.linalg.norm(d, axis=1)
    outlier_idxs = np.where(d_norm > m * np.mean(d_norm))
    return np.delete(data, outlier_idxs, axis=0)

calibration = np.load('kinect_calibration.npz')
x_vec = calibration['x_vec']
y_vec = calibration['y_vec']
z_vec = calibration['z_vec']
x_vec1 = calibration['x_vec1']
y_vec1 = calibration['y_vec1']
z_vec1 = calibration['z_vec1']

c0 = calibration['c0']
c1 = calibration['c1']

x_vec = reject_outliers(x_vec)
y_vec = reject_outliers(y_vec)
z_vec = reject_outliers(z_vec)

x_vec1 = reject_outliers(x_vec1)
y_vec1 = reject_outliers(y_vec1)
z_vec1 = reject_outliers(z_vec1)

c0 = reject_outliers(c0)
c1 = reject_outliers(c1)

# %%

json_file0 = open('track_data_cam0.json', "r")
json_file1 = open('track_data_cam1.json', "r")
json_data0 = json.load(json_file0)
json_data1 = json.load(json_file1)
print(len(json_data0))
print(len(json_data1))

# %%

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

naval_joint_id = 1
head_joint_id = 26
joint_id = head_joint_id
cam0 = get_joint_positions(json_data0, joint_id)
cam1 = get_joint_positions(json_data1, joint_id)

cam0_confidence = get_joint_confidences(json_data0, joint_id)
cam1_confidence = get_joint_confidences(json_data1, joint_id)
timestamp0 = get_timestamps(json_data0)
timestamp1 = get_timestamps(json_data1)

is_track0 = np.array([json_data0[i]["num_bodies"] > 0 for i in range(len(json_data0))])
is_track1 = np.array([json_data1[i]["num_bodies"] > 0 for i in range(len(json_data1))])

print(cam0.shape)
print(cam1.shape)
# %%

# cam0 = data['cam0']
# cam1 = data['cam1']
# cam0_confidence = data['cam0_confidence']
# cam1_confidence = data['cam1_confidence']
# timestamp0 = data['timestamp0']
# timestamp1 = data['timestamp1']
# is_track0 = data['is_track0']
# is_track1 = data['is_track1']

# first_timestamp0 = last_timestamp // 1000 - (timestamp0[-1] - timestamp0[0])
# timestamp0 = first_timestamp0 + timestamp0

# first_timestamp1 = last_timestamp // 1000 - (timestamp1[-1] - timestamp1[0])
# timestamp1 = first_timestamp1 + timestamp1

# %%

c0_mean = np.mean(c0, axis=0)
c1_mean = np.mean(c1, axis=0)

x_vec_mean = np.mean(x_vec, axis=0)
x_vec1_mean = np.mean(x_vec1, axis=0)

y_vec_mean = np.mean(y_vec, axis=0)
y_vec1_mean = np.mean(y_vec1, axis=0)

z_vec_mean = np.mean(z_vec, axis=0)
z_vec1_mean = np.mean(z_vec1, axis=0)

# %%

R = np.array([x_vec_mean, y_vec_mean, z_vec_mean]).T
R_SVD = closest_rotation_matrix(R)

R1 = np.array([x_vec1_mean, y_vec1_mean, z_vec1_mean]).T
R1_SVD = closest_rotation_matrix(R1)

# %%

# emblo coordinates seen by R10 sensor
c0_room = np.array([[1.846738, 2.1159978, 1.4999355]]).T * 1000

cam0_room = R_SVD.T @ (cam0 - c0_mean).T
cam1_room = R1_SVD.T @ (cam1 - c1_mean).T

cam0_room = cam0_room + c0_room
cam1_room = cam1_room + c0_room

cam0_room /= 1000
cam1_room /= 1000

# %%
cam0 = cam0_room.T
cam1 = cam1_room.T

#%%

height_offset = 0.3
cam0[:, 2] += height_offset
cam1[:, 2] += height_offset

# %%

def get_closest_timestamp_index(timestamp, target_timestamp):
    # Find the index of the closest timestamp in the array
    closest_index = (np.abs(timestamp - target_timestamp)).argmin()
    return closest_index

# if is_track0 or is_track1 is False, then replace with the data of the other camera
for i in range(min(len(cam0), len(cam1))):
    if is_track0[i] == False and is_track1[i] == True:
        cam0[i] = cam1[get_closest_timestamp_index(timestamp1, timestamp0[i])]
    elif is_track1[i] == False and is_track0[i] == True:
        cam1[i] = cam0[get_closest_timestamp_index(timestamp0, timestamp1[i])]
    else:
        # TODO: handle the case when neither camera has track
        pass


# %%
radar_data = np.load("processed_radar_data.npz")
trajectory = radar_data['trajectory']
radar_timestamps = radar_data['timestamps']

radar_timestamps /= 1000

# TODO: why does this happen?
radar_timestamps_offset = 500 * 1000
radar_timestamps += radar_timestamps_offset

start_idx = get_closest_timestamp_index(radar_timestamps, timestamp0[0])
stop_idx = get_closest_timestamp_index(radar_timestamps, timestamp0[-1])
trajectory = trajectory[start_idx:stop_idx]
radar_timestamps = radar_timestamps[start_idx:stop_idx]

# %%
radar_timestamps = (radar_timestamps - radar_timestamps[0]) / 1e6
ref_timestamp = timestamp0[0]
timestamp0 = (timestamp0 - ref_timestamp) / 1e6
timestamp1 = (timestamp1 - ref_timestamp) / 1e6

# %%

# plot x y and z in three subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[0].plot(timestamp0, cam0[:, 0], label='cam0 x')
axs[0].plot(timestamp1, cam1[:, 0], label='cam1 x')
axs[0].plot(radar_timestamps, trajectory[:, 0], label='radar x')
# axs[0].set_title('X axis')
# axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('X pos [m]')
axs[0].legend()
axs[0].grid()

axs[1].plot(timestamp0, cam0[:, 1], label='cam0 y')
axs[1].plot(timestamp1, cam1[:, 1], label='cam1 y')
axs[1].plot(radar_timestamps, trajectory[:, 1], label='radar y')
# axs[1].set_title('Y axis')
# axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Y pos [m]')
axs[1].legend()
axs[1].grid()

axs[2].plot(timestamp0, cam0[:, 2], label='cam0 z')
axs[2].plot(timestamp1, cam1[:, 2], label='cam1 z')
axs[2].plot(radar_timestamps, trajectory[:, 2], label='radar z')
# axs[2].set_title('Z axis')
# axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Z pos [m]')
axs[2].legend()
axs[2].grid()

# axs[3].plot(cam0_confidence, label='cam0 confidence')
# axs[3].plot(cam1_confidence, label='cam1 confidence')
# axs[3].set_title('Confidence')
# axs[3].set_xlabel('Frame')
# axs[3].set_ylabel('Confidence')
# axs[3].legend()
# axs[3].grid()

axs[3].plot(timestamp0, is_track0, label='cam0 track')
axs[3].plot(timestamp1, is_track1, label='cam1 track')
# axs[3].set_title('Has track')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Has track')
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

tracks = np.loadtxt('trajectory_radar.txt')

plt.scatter(tracks[:, 0], tracks[:, 1], c='r', s=1, label="R10")
plt.scatter(cam0[:, 0], cam0[:, 1], c='b', s=1, label="Kinect Cam0")
plt.scatter(cam1[:, 0], cam1[:, 1], c='g', s=1, label="Kinect Cam1")
plt.xlim(0, 4.77)
plt.ylim(0, 4.48)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Position Trajectory")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.legend()
plt.show()
