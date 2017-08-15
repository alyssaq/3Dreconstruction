import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import camera
import processor
import structure
import transformers


def plot_projections(points):
    num_images = len(points)

    plt.figure()
    plt.suptitle('3D to 2D Projections', fontsize=16)
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.plot(points[i][0], points[i][1], 'r.')


def plot_cube(points3d, title=''):
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.plot(points3d[0], points3d[1], points3d[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    return ax

def extrinsic_from_camera_pose(m_c1_wrt_c2):
    # Inverse to get extrinsic matrix from camera to world view
    # http://ksimek.github.io/2012/08/22/extrinsic/
    # Returns homogenous 4x4 extrinsic camera matrix
    # Alternatively, R = R^t, T = -RC
    # Rct = m_c1_wrt_c2[:3, :3].T
    # t = -np.dot(Rct, m_c1_wrt_c2[:3, 3])
    H_m = np.vstack([m_c1_wrt_c2, [0, 0, 0, 1]])
    ext = np.linalg.inv(H_m)
    return ext


def camera_corners(camera, dist=0.25):
    d = dist
    x, y, z = np.ravel(camera.t)
    corners = np.array([
        [x-d, y+d, z],
        [x+d, y+d, z],
        [x+d, y-d, z],
        [x-d, y-d, z],
        [x-d, y+d, z]
    ]).T

    return np.asarray(np.dot(camera.R, corners))


size = 300  # size of image in pixels
center = size / 2
intrinsic = np.array([
    [size, 0, center],
    [0, size, center],
    [0, 0, 1]
])

# Points of on the surface of the cube
points3d = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2],
    [2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, -1, -1, -2, -2, -2, 0, 0, -1, -1, -2, -2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Define pose of cube with respect to camera1 in world view
rotation_mat = transformers.rotation_3d_from_angles(120, 0, 60)
translation_mat = np.matrix([0, 0, 5]).T
c = camera.Camera(K=intrinsic, R=rotation_mat, t=translation_mat)

# Project 3d points to camera1 on the left
points1 = c.project(points3d)
points1 = processor.cart2hom(points1)

# Get 4x4 homogenous extrinsic parameters of camera 1
H_c1 = np.vstack([c.extrinsic, [0, 0, 0, 1]])

# Define rotation of camera1 wrt camera2 and
# translation of camera2 wrt camera1
rotation_mat_wrt_c1 = transformers.rotation_3d_from_angles(0, -25, 0)
translation_mat_wrt_c1 = np.matrix([3, 0, 1]).T
H_c2_c1 = np.hstack([rotation_mat_wrt_c1, translation_mat_wrt_c1])
H_c1_c2 = extrinsic_from_camera_pose(H_c2_c1)

# Calculate pose of model wrt to camera2 in world view
H_c2 = np.dot(H_c1_c2, H_c1)

# Project 3d points to camera 2 on the right
c2 = camera.Camera(K=intrinsic, R=H_c2[:3, :3], t=H_c2[:3, 3])
points2 = c2.project(points3d)
points2 = processor.cart2hom(points2[:2])

# True essential matrix E = [t]R
true_E = np.dot(structure.skew(translation_mat_wrt_c1), rotation_mat_wrt_c1)
print('True essential matrix:', true_E)

# Calculate essential matrix with 2d points.
# Result will be up to a scale
# First, normalize points
points1n = np.dot(np.linalg.inv(intrinsic), points1)
points2n = np.dot(np.linalg.inv(intrinsic), points2)
E = structure.compute_essential_normalized(points1n, points2n)
print('Computed essential matrix:', (-E / E[0][1]))

# True fundamental matrix F = K^-t E K^-1
true_F = np.dot(np.dot(np.linalg.inv(intrinsic).T, true_E), np.linalg.inv(intrinsic))
F = structure.compute_fundamental_normalized(points1, points2)
print('True fundamental matrix:', true_F)
print('Computed fundamental matrix:', (F * true_F[2][2]))

# Given we are at camera 1, calculate the parameters for camera 2
# Using the essential matrix returns 4 possible camera paramters
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2s = structure.compute_P_from_essential(E)

ind = -1
for i, P2 in enumerate(P2s):
    # Find the correct camera parameters
    d1 = structure.reconstruct_one_point(points1n[:, 0], points2n[:, 0], P1, P2)
    P2_homogenous = extrinsic_from_camera_pose(P2)
    d2 = np.dot(P2_homogenous[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

print('True pose of c2 wrt c1: ', H_c1_c2)
P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
P2f = structure.compute_P_from_fundamental(F)
print('Calculated camera 2 parameters:', P2, P2f)

tripoints3d = structure.reconstruct_points(points1n, points2n, P1, P2)
tripoints3d = structure.linear_triangulation(points1n, points2n, P1, P2)

structure.plot_epipolar_lines(points1n, points2n, E)
plot_projections([points1, points2])

ax = plot_cube(points3d, 'Original')
cam_corners1 = camera_corners(c)
cam_corners2 = camera_corners(c2)
ax.plot(cam_corners1[0], cam_corners1[1], cam_corners1[2], 'g-')
ax.plot(cam_corners2[0], cam_corners2[1], cam_corners2[2], 'r-')

plot_cube(tripoints3d, '3D reconstructed')
plt.show()
