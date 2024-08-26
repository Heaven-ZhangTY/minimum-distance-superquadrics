"""
Demo: Minimum distance between superquadric surfaces
"""
from matplotlib import colors
from minkowski_sum_definition import *
import numpy as np
from mpl_toolkits import mplot3d

# Use Matplotlib for visualization
figure = plt.figure()
axes = mplot3d.Axes3D(figure)

# Superquadric surfaces
n = 20
n1_01, n2_01, a_01, b_01, c_01 = 0.6, 0.4, 200, 100, 100
n1_02, n2_02, a_02, b_02, c_02 = 1.2, 1.2, 172, 250, 160
RotX_1, RotY_1, RotZ_1 = 0, 0, 0
TranX_1, TranY_1, TranZ_1 = 0, 0, 0
RotX_2, RotY_2, RotZ_2 = 0, 0, 45
TranX_2, TranY_2, TranZ_2 = 500, 500, 500

# Superquadric model - 01
super_qudric_01 = super_qudric(n1_01, n2_01, a_01, b_01, c_01)
x_01, y_01, z_01 = super_qudric_01.point_super_qudric()

# Calculate the gradient of the surface points of superquadric - 01
grad_x_01, grad_y_01, grad_z_01 = super_qudric_01.point_gradient()
mu_x_M, mu_y_M, mu_z_M = super_qudric_01.mu_M(grad_x_01, grad_y_01, grad_z_01, n1_02, n2_02, a_02, b_02, c_02, RotX_1, RotY_1, RotZ_1, RotX_2, RotY_2, RotZ_2)
g_x_M, g_y_M, g_z_M = super_qudric_01.g(mu_x_M, mu_y_M, mu_z_M, n1_02, n2_02, a_02, b_02, c_02)
x_M, y_M, z_M = super_qudric_01.MinSum_M(g_x_M, g_y_M, g_z_M, grad_x_01, grad_y_01, grad_z_01, RotX_1, RotY_1, RotZ_1, RotX_2, RotY_2, RotZ_2)

# Plot superquadric model - 01
x_01_new, y_01_new, z_01_new = super_qudric_01.RotationXYZ(x_01, y_01, z_01, RotX_1, RotY_1, RotZ_1, TranX_1, TranY_1, TranZ_1)
surf_01_new = axes.plot_surface(x_01_new, y_01_new, z_01_new, alpha=0.8, color='darkgray')

# Plot superquadric model - 02
super_qudric_02 = super_qudric(n1_02, n2_02, a_02, b_02, c_02)
x_02, y_02, z_02 = super_qudric_02.point_super_qudric()
x_02_new, y_02_new, z_02_new = super_qudric_02.RotationXYZ(x_02, y_02, z_02, RotX_2, RotY_2, RotZ_2, TranX_2, TranY_2, TranZ_2)
surf_02_new = axes.plot_surface(x_02_new, y_02_new, z_02_new, alpha=0.6, color='gray')

# Plot Minkowski sum
surf_sum_M = axes.plot_surface(x_M, y_M, z_M, alpha=0.4, color='lightgray')

# Construct matrix b
# Create an initial 3D array
p2 = np.array([TranX_2, TranY_2, TranZ_2]).reshape((1, 3))
# Define the initial array
b = np.array([]).reshape((0, 3))
# Loop to add arrays
for i in range(n):
    for j in range(n):
        b_x = TranX_2 - x_M[i, j]
        b_y = TranY_2 - y_M[i, j]
        b_z = TranZ_2 - z_M[i, j]
        new_arr = np.array([b_x, b_y, b_z]).reshape((1, 3))
        b = np.append(b, new_arr, axis=0)

# Construct matrix a
# Define the initial array
a = np.array([]).reshape((0, 3))
# Loop to add arrays
for i in range(n):
    for j in range(n):
        a_x = grad_x_01[i, j]
        a_y = grad_y_01[i, j]
        a_z = grad_z_01[i, j]
        new_arr = np.array([a_x, a_y, a_z]).reshape((1, 3))
        a = np.append(a, new_arr, axis=0)

# Calculate the length of the cross product of the vectors
lengths = np.linalg.norm(np.cross(a, b), axis=1)

# Find the minimum value and its corresponding position
min_length = np.min(lengths)
min_indices = np.where(lengths == min_length)[0]

# Find the minimum vector pair
min_vec1 = a[min_indices[0]]
min_vec2 = b[min_indices[0]]

# Minimum distance point on the boundary of the Minkowski sum
x_min = p2 - min_vec2
print('Minimum distance point: ', x_min)

# Minimum distance
dis_min = np.linalg.norm(min_vec2)
print('Minimum distance: ', dis_min)

# Plot the point
axes.scatter(x_min[0][0], x_min[0][1], x_min[0][2], c='red', s=100)

# Draw a line between two points
# Define the coordinates of the two points
point1 = (x_min[0][0], x_min[0][1], x_min[0][2])
point2 = (TranX_2, TranY_2, TranZ_2)
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
z_values = [point1[2], point2[2]]
# Plot the line
plt.plot(x_values, y_values, z_values, color='black', marker='o')  # Draw line and points

# Add labels
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_zlabel('Z')

# Add coordinate axes
axes_add(0, 0, 0, axes, 5)
axes_add(TranX_2, TranY_2, TranZ_2, axes, 2)

# Disable the coordinate grid
axes.grid(b=False)
# Set grid line style and color
axes.grid(linestyle='--', linewidth=5, color='gray', alpha=1)

# Disable the coordinate system
axes.axis('off')

# Show the plot
plt.show()
