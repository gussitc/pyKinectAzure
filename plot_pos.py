#%%
import numpy as np
import matplotlib.pyplot as plt
from rotation import closest_rotation_matrix

#%%
calibration = np.load('kinect_calibration.npz')
x_vec = calibration['x_vec']
y_vec = calibration['y_vec']
z_vec = calibration['z_vec']
x_vec1 = calibration['x_vec1']
y_vec1 = calibration['y_vec1']
z_vec1 = calibration['z_vec1']

c0 = calibration['c0']
c1 = calibration['c1']

#%%

data = np.load('kinect_position.npz')
cam0 = data['cam0']
cam1 = data['cam1']
cam0_confidence = data['cam0_confidence']
cam1_confidence = data['cam1_confidence']
timestamp0 = data['timestamp0']
timestamp1 = data['timestamp1']
is_track0 = data['is_track0']
is_track1 = data['is_track1']

#%%

c0_mean = np.mean(c0, axis=0)
c1_mean = np.mean(c1, axis=0)

x_vec_mean = np.mean(x_vec, axis=0)
x_vec1_mean = np.mean(x_vec1, axis=0)

y_vec_mean = np.mean(y_vec, axis=0)
y_vec1_mean = np.mean(y_vec1, axis=0)

z_vec_mean = np.mean(z_vec, axis=0)
z_vec1_mean = np.mean(z_vec1, axis=0)

#%%

R = np.array([x_vec_mean, y_vec_mean, z_vec_mean]).T
R_SVD = closest_rotation_matrix(R)

R1 = np.array([x_vec1_mean, y_vec1_mean, z_vec1_mean]).T
R1_SVD = closest_rotation_matrix(R1)

#%%

cam0_room = R_SVD.T @ (cam0 - c0_mean).T
cam1_room = R1_SVD.T @ (cam1 - c1_mean).T

#%%
cam0 = cam0_room.T
cam1 = cam1_room.T

#%%

def get_closest_timestamp_index(timestamp, target_timestamp):
    # Find the index of the closest timestamp in the array
    closest_index = (np.abs(timestamp - target_timestamp)).argmin()
    return closest_index

# if is_track0 or is_track1 is False, then replace with the data of the other camera
for i in range(len(cam0)):
    if is_track0[i] == False and is_track1[i] == True:
        cam0[i] = cam1[get_closest_timestamp_index(timestamp1, timestamp0[i])]
    elif is_track1[i] == False and is_track0[i] == True:
        cam1[i] = cam0[get_closest_timestamp_index(timestamp0, timestamp1[i])]
    else:
        # TODO: handle the case when neither camera has track
        pass

#%%

# plot x y and z in three subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[0].plot(timestamp0, cam0[:, 0], label='cam0 x')
axs[0].plot(timestamp1, cam1[:, 0], label='cam1 x')
axs[0].set_title('X axis')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('X position')
axs[0].legend()
axs[0].grid()

axs[1].plot(timestamp0, cam0[:, 1], label='cam0 y')
axs[1].plot(timestamp1, cam1[:, 1], label='cam1 y')
axs[1].set_title('Y axis')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Y position')
axs[1].legend()
axs[1].grid()

axs[2].plot(timestamp0, cam0[:, 2], label='cam0 z')
axs[2].plot(timestamp1, cam1[:, 2], label='cam1 z')
axs[2].set_title('Z axis')
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Z position')
axs[2].legend()
axs[2].grid()

# axs[3].plot(cam0_confidence, label='cam0 confidence')
# axs[3].plot(cam1_confidence, label='cam1 confidence')
# axs[3].set_title('Confidence')
# axs[3].set_xlabel('Frame')
# axs[3].set_ylabel('Confidence')
# axs[3].legend()
# axs[3].grid()

axs[3].plot(timestamp0, is_track0, label='cam0 is_track')
axs[3].plot(timestamp1, is_track1, label='cam1 is_track')
axs[3].set_title('is_track')
axs[3].set_xlabel('Frame')
axs[3].set_ylabel('is_track')
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()


