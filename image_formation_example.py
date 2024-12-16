import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt

# Euler angle (degree, counterclockwise is positive) to rotation matrix
def euler_to_rotmat(x_degree, y_degree, z_degree):
    x_angle = math.radians(x_degree)
    y_angle = math.radians(y_degree)
    z_angle = math.radians(z_degree)

    # Rotations around X, Y, Z axis, respectively
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(x_angle), -math.sin(x_angle)],
                   [0, math.sin(x_angle), math.cos(x_angle)]])

    Ry = np.array([[math.cos(y_angle), 0, math.sin(y_angle)],
                   [0, 1, 0],
                   [-math.sin(y_angle), 0, math.cos(y_angle)]])

    Rz = np.array([[math.cos(z_angle), -math.sin(z_angle), 0],
                   [math.sin(z_angle), math.cos(z_angle), 0],
                   [0, 0, 1]])

    # compute a general rotation matrix
    R = Rz @ Ry @ Rx

    return R


# %% create a cube in world coordinate system (in meter)
v_o = np.empty((3, 8))
i = 0
for x in [-1, 1]:
    for y in [-1, 1]:
        for z in [-1, 1]:
            v_o[:, i] = np.array([x, y, z])
            i += 1

# v_o = np.array([(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]).transpose()
#
v_o *= 1000  # convert m to mm

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(v_o[0], v_o[1], v_o[2], color='r', s=80)
ax.plot(v_o[:,0], v_o[:,1])
plt.show()



# %% object coordinate system to world
R_ow = euler_to_rotmat(30, 0, 0)  # object to world rotation
tx_ow, ty_ow, tz_ow = (500, 0, 0)  # object to world translation in x, y, z
T_ow = np.array([[tx_ow], [ty_ow], [tz_ow]])  # object to world translation vector

# rotate and translate the cube vetices using R
v_w = R_ow @ v_o + T_ow  # or [R_ow | T_ow] @ v_o (homogenious coordinate system)


# %% transform the cube from world coordinate system to camera coordinate system

# world to camera coordinate system (extrinsics)
R_cw = euler_to_rotmat(0, 0, 0)  # here the angles are camera's rotation relative to world
R_wc = R_cw.transpose()  # world's rotation relative to camera (not camera to world)

cam_pos = np.array([0, 0, -8000])  # camera optical center position in world coordinate system
T_cw = cam_pos[:, None]  # to translate a point in camera coordinate system to world (use origin to explain)
T_wc = -R_wc @ T_cw  # to translate a point in world coordinate system to camera

# transform the cube vetices to camera coordinate system
v_c = R_wc @ v_w + T_wc

# %% camera coordinate system to image coordinate system (intrinsics)
# intrinsics
cam_h, cam_w = (360, 540)  # camera image size (pixel)
sensor_h, sensor_w = (24, 36)  # camera sensor size (mm)
focal_length = 50  # camera lens focal length (mm)

fx, fy = (focal_length / sensor_w * cam_w, focal_length / sensor_h * cam_h)
cx, cy = (cam_w / 2, cam_h / 2)
cam_k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # camera intrinsics
# inv_cam_k = np.linalg.inv(cam_k)  # precompute inverse of camera intrinsics

# project 3D point to 2D image
v_i = cam_k @ v_c  # projection (intrinsic) matrix
v_i = v_i / v_i[-1, :]  # divide by Z and drop Z
v_i = v_i[:-1]

# %% draw vertices
im = np.ones((cam_h, cam_w, 3))
# plt.scatter(v_i[0], v_i[1])
# plt.show()

# draw cube lines
im = cv.line(im, v_i[:,0].astype(np.int),  v_i[:,1].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,0].astype(np.int),  v_i[:,2].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,1].astype(np.int),  v_i[:,3].astype(np.int), (255, 0, 0), 5)
im = cv.line(im, v_i[:,2].astype(np.int),  v_i[:,3].astype(np.int), (255, 0, 0), 5)
im = cv.line(im, v_i[:,4].astype(np.int),  v_i[:,5].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,4].astype(np.int),  v_i[:,6].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,5].astype(np.int),  v_i[:,7].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,6].astype(np.int),  v_i[:,7].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,0].astype(np.int),  v_i[:,4].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,1].astype(np.int),  v_i[:,5].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,2].astype(np.int),  v_i[:,6].astype(np.int), (0, 255, 0), 5)
im = cv.line(im, v_i[:,3].astype(np.int),  v_i[:,7].astype(np.int), (255, 0, 0), 5)

# draw vertex indices
for i in range(v_i.shape[1]):
    im = cv.putText(im, str(i), v_i[:, i].astype(np.int), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)

plt.imshow(im)
plt.scatter(v_i[0], v_i[1])
# plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.show()
