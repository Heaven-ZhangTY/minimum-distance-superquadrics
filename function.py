from math import *
import numpy as np

def DegToRad(value):
    return value * pi / 180.0

def ConvertRPYToMat(psi, theta, phi):
    "Convert Roll, Pitch, Yaw angles to a rotation matrix"
    # psi, theta, phi: input angles
    # Convert angles from degrees to radians
    psi = DegToRad(psi)
    theta = DegToRad(theta)
    phi = DegToRad(phi)

    Matrix = np.array([
        [cos(phi) * cos(theta), cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi), cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)],
        [sin(phi) * cos(theta), sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)],
        [-sin(theta), cos(theta) * sin(psi), cos(theta) * cos(psi)]
    ])
    return Matrix

def RotX(theta):
    # Convert angle from degrees to radians
    theta = np.radians(theta)
    # Create a rotation matrix for rotation around the X-axis by angle theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])
    return R_x

def RotY(theta):
    # Convert angle from degrees to radians
    theta = np.radians(theta)
    # Create a rotation matrix for rotation around the Y-axis by angle theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_y = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    return R_y

def RotZ(theta):
    # Convert angle from degrees to radians
    theta = np.radians(theta)
    # Create a rotation matrix for rotation around the Z-axis by angle theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_z = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return R_z

def axes_add(x, y, z, axes, length):
    "Add coordinate axes"
    # Set origin marker
    origin = [x, y, z]
    axes.scatter(origin[0], origin[1], origin[2], c='r', s=200, marker='o')
    # The quiver() function is used to plot 2D and 3D vector fields
    axes.quiver(x, y, z, length, 0, 0, color='red', length=1, arrow_length_ratio=0.1, linewidths=1)
    axes.quiver(x, y, z, 0, length, 0, color='green', length=1, arrow_length_ratio=0.1)
    axes.quiver(x, y, z, 0, 0, length, color='blue', length=1, arrow_length_ratio=0.1)
