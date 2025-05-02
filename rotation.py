#%%
import numpy as np

def closest_rotation_matrix(Q):
    U,s,VT = np.linalg.svd(Q)
    R = U@VT
    return R

if __name__ == '__main__':
    data = np.load('kinect_calibration.npz')

    x = data['x_vec']
    y = data['y_vec']
    z = data['z_vec']

    print("x: ", x)
    print("y: ", y)
    print("z: ", z)

    R_approx = np.array([x, y, z]).T
    print("\nR_approx:")
    print(R_approx)
    print("\nIdentity:")
    print(R_approx @ R_approx.T)

    R = closest_rotation_matrix(R_approx)
    print("\nR:")
    print(R)
    print("\nIdentity:")
    print(R @ R.T)

    print(data['emblo_pos'])