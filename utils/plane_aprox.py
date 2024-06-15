import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA

# Sample input data
# np.random.seed(0)
# dem_points = np.random.rand(1_000, 3)  # Replace with actual DEM data
# dem_points[:,2] = 1

dem_points = np.genfromtxt("dem.txt", delimiter=' ')
# dem_points = dem_points[:10_000,:]
# dem_points[:,2] = 1

ray_origin = np.array([599_400, 5_287_400, 660])  # Example ray origin
ray_direction = np.array([599_400, 5_287_400, 740])  # Example ray direction

# Step 1: Fit a plane to the DEM points
# We use PCA to find the normal of the plane
pca = PCA(n_components=3)
pca.fit(dem_points)
normal = pca.components_[-1]

# Plane equation coefficients: ax + by + cz + d = 0
a, b, c = normal
d = -np.dot(normal, pca.mean_)

# Step 2: Ray-Plane Intersection
# Ray equation: ray_origin + t * ray_direction
# Plane equation: ax + by + cz + d = 0

x0, y0, z0 = ray_origin
vx, vy, vz = ray_direction

# Substitute the ray equation into the plane equation and solve for t
t = -(a*x0 + b*y0 + c*z0 + d) / (a*vx + b*vy + c*vz)

# Find the intersection point
intersection_point = ray_origin + t * ray_direction

# Step 3: Identify the closest point on the DEM to the intersection point
distances = np.linalg.norm(dem_points - intersection_point, axis=1)
closest_index = np.argmin(distances)
closest_point = dem_points[closest_index]

print("Intersection Point:", intersection_point)
print("Closest Point on DEM:", closest_point)

# Step 4: Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot only a subset of DEM points
subset_indices = np.random.choice(len(dem_points), size=10000, replace=False)
ax.scatter(dem_points[subset_indices, 0], dem_points[subset_indices, 1], dem_points[subset_indices, 2],
           s=1, c='b', alpha=0.5, label='DEM Points (Subset)')

# Plot plane
xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
zz = (-a * xx - b * yy - d) / c
ax.plot_surface(xx, yy, zz, color='g', alpha=0.2, label='Fitted Plane')

# Plot ray
ax.plot([ray_origin[0], ray_origin[0] + ray_direction[0]],
        [ray_origin[1], ray_origin[1] + ray_direction[1]],
        [ray_origin[2], ray_origin[2] + ray_direction[2]], 'r-', label='Ray')

# Plot intersection point
ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='r', label='Intersection Point')

# Plot closest DEM point
ax.scatter(closest_point[0], closest_point[1], closest_point[2], color='k', label='Closest DEM Point')

# Set axis limits based on data range
x_min, x_max = np.min(dem_points[:, 0]), np.max(dem_points[:, 0])
y_min, y_max = np.min(dem_points[:, 1]), np.max(dem_points[:, 1])
z_min, z_max = np.min(dem_points[:, 2]), np.max(dem_points[:, 2])
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.legend()

plt.title('Ray-Plane Intersection with DEM Visualization')
plt.show()