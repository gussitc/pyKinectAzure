#%%
import numpy as np
import matplotlib.pyplot as plt

data = np.load('kinect_position.npz')
cam0_raw = data['cam0']
cam1_raw = data['cam1']
cam0_confidence = data['cam0_confidence']
cam1_confidence = data['cam1_confidence']
timestamp0 = data['timestamp0']
timestamp1 = data['timestamp1']

#%%

START_FRAME = 0

cam0 = cam0_raw[START_FRAME:]
cam1 = cam1_raw[START_FRAME:]

#%%

# plot x y and z in three subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
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

plt.tight_layout()
plt.show()


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

c0_mean = np.mean(c0, axis=0)
c1_mean = np.mean(c1, axis=0)

x_vec_mean = np.mean(x_vec, axis=0)
x_vec1_mean = np.mean(x_vec1, axis=0)

y_vec_mean = np.mean(y_vec, axis=0)
y_vec1_mean = np.mean(y_vec1, axis=0)

z_vec_mean = np.mean(z_vec, axis=0)
z_vec1_mean = np.mean(z_vec1, axis=0)

#%%
from rotation import closest_rotation_matrix

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
# plot x y and z coordinates of c0 and c1 in three subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[0].plot(c0[:, 0], label='cam0 x')
# axs[0].plot(c1[:, 0], label='cam1 x')
axs[0].set_title('X axis')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('X position')
axs[0].legend()
axs[0].grid()

axs[1].plot(c0[:, 1], label='cam0 y')
# axs[1].plot(c1[:, 1], label='cam1 y')
axs[1].set_title('Y axis')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Y position')
axs[1].legend()
axs[1].grid()

axs[2].plot(c0[:, 2], label='cam0 z')
# axs[2].plot(c1[:, 2], label='cam1 z')
axs[2].set_title('Z axis')
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Z position')
axs[2].legend()
axs[2].grid()




