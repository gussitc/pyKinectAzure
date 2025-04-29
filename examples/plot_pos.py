#%%
import numpy as np
import matplotlib.pyplot as plt

kinect_pos = np.load('kinect_pos.npy')

#%%

radar_pos = np.loadtxt('trajectory_square.txt')
print(radar_pos.shape)

#%%
# Color fades with time
color = np.repeat(['r'], kinect_pos.shape[0])
color[0:100] = 'g'
# plt.scatter(kinect_pos[:, 0], kinect_pos[:, 1], s=2, label="Kinect", c=range(len(kinect_pos), 0, -1), cmap=plt.get_cmap('Blues'))
# plt.scatter(radar_pos[:, 0], radar_pos[:, 1], s=2, label="R10", c=range(len(radar_pos), 0, -1), cmap=plt.get_cmap('Reds'))
plt.scatter(kinect_pos[:, 0], kinect_pos[:, 1], s=2, label="Kinect", c='b')
plt.scatter(radar_pos[:, 0], radar_pos[:, 1], s=2, label="R10", c='r')
plt.xlim(0, 4.77)
plt.ylim(0, 4.48)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajectory")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.legend()
plt.show()

#%%