import numpy as np

"""
Helper functions to create various matrices
"""

def rotation_3d_from_angles(x_angle, y_angle=0, z_angle=0):
    """ Creates a 3D rotation matrix given angles in degrees.
        Positive angles rotates anti-clockwise.
    :params x_angle, y_angle, z_angle: x, y, z angles between 0 to 360
    :returns: 3x3 rotation matrix """
    ax = np.deg2rad(x_angle)
    ay = np.deg2rad(y_angle)
    az = np.deg2rad(z_angle)

    # Rotation matrix around x-axis
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)]
    ])
    # Rotation matrix around y-axis
    ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ])
    # Rotation matrix around z-axis
    rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1]
    ])

    return np.dot(np.dot(rx, ry), rz)
