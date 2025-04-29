#%%
import numpy as np
import matplotlib.pyplot as plt

kinect_pos = np.load('kinect_pos.npy')

#%%
# Color fades with time
color = np.repeat(['r'], kinect_pos.shape[0])
color[0:100] = 'g'
plt.scatter(kinect_pos[:, 0], kinect_pos[:, 1], s=1, label="Kinect", c=range(len(kinect_pos), 0, -1), cmap=plt.get_cmap('Blues'))
plt.xlim(0, 4.77)
plt.ylim(0, 4.48)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Position Trajectory")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.legend()
plt.show()

