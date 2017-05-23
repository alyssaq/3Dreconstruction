import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import camera
import processor
import structure

def plot_projection(p1, p2):
    plt.figure()
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.plot(p1[0], p1[1], 'r.')

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.plot(p2[0], p2[1], 'r.')

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

size = 300 # size of image in pixels
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
  [0, 0, 0, -1,-1, -1, -2, -2, -2, 0, 0, -1, -1, -2, -2],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Project 3d points to camera 1 on the left
rotation_mat = camera.rotation_mat_from_angles(75, 0, 60)
translation_mat = np.matrix([0, 0, 5]).T
c = camera.Camera(K=intrinsic, R=rotation_mat, t=translation_mat)
points1 = c.project(points3d)
points1 = processor.cart2hom(points1)
#np.savetxt('points2d_left.txt', points2d[:2].T, fmt='%.10f')

# Get 4x4 homogenous extrinsic parameters of camera 1
H_c1 = np.vstack([c.extrinsic, [0, 0, 0, 1]])

# Calculate pose of the model wrt camera 2
# Create camera 2 extrinsic matrix wrt to camera 1
rotation_mat_wrt_c1 = camera.rotation_mat_from_angles(0, -25, 0)
translation_mat_wrt_c1 = np.matrix([3, 0, 1]).T
c2_extrinsic = np.hstack([rotation_mat_wrt_c1, translation_mat_wrt_c1])
# Get 4x4 homogenous extrinsic parameters of camera 2 wrt camera 1
H_c2_c1 = np.vstack([c2_extrinsic, [0, 0, 0, 1]])

# Get extrinsic parameters of camera 2
H_c1_c2 = np.linalg.inv(H_c2_c1)
H_c2 = np.dot(H_c1_c2, H_c1)

# Project 3d points to camera 2 on the right
c2 = camera.Camera(K=intrinsic, R=H_c2[:3, :3], t=H_c2[:3, 3])
points2 = c2.project(points3d)
points2 = processor.cart2hom(points2[:2])

# True essential matrix E = [t]R
E = np.dot(structure.skew(translation_mat_wrt_c1), rotation_mat_wrt_c1)
print('Original essential matrix:', E)

# Calculate essential matrix with 2d points.
# Result will be up to a scale
points1n = np.dot(np.linalg.inv(intrinsic), points1)
points2n = np.dot(np.linalg.inv(intrinsic), points2)
E = structure.compute_essential_normalized(points1n, points2n)
print('Computed essential matrix:', E)

# Given we are at camera 1, calculate the parameters for camera 2
# Using the essential matrix returns 4 possible camera paramters
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2s = structure.compute_P_from_essential(E)

# pick the solution with most points in front of camera
depth = 0
tripoints3d = None
ind = -1
for i, P2 in enumerate(P2s):
    # triangulate inliers and compute depth for each camera
    X = structure.linear_triangulation(points1n, points2n, P1, P2)
    d1 = np.dot(P1, X)[2]
    d2 = np.dot(P2, X)[2]

    if sum(d1 > 0) + sum(d2 > 0) > depth:
      ind = i
      tripoints3d = X
      depth = sum(d1>0) + sum(d2>0)
      infront = (d1>0) & (d2>0)

print('ind', ind, depth)
print('Num points triangulated', X.shape[1], tripoints3d.shape[1])

plt.figure()
structure.plot_epipolar_lines(points1n, points2n, E)
plot_projection(points1, points2)
plot_cube(points3d, 'Original')
plot_cube(tripoints3d[:, infront], '3D reconstructed')
plt.show()
