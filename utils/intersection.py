import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

DEM_PATH = "dem.txt"

class IntersectionChecker:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, path2dem: str) -> None:
        self.origin = origin
        self.direction = direction
        self.path = path2dem
        self.pointcloud: np.ndarray = self.read_dem()

        if not self.is_ray_valid():
            sys.exit(0)

    def read_dem(self) -> np.ndarray:
        """Use numpy to read DEM file efficiently"""
        print("Reading txt file ...")
        data = np.genfromtxt(self.path, delimiter=' ')
        return data

    def is_ray_valid(self) -> bool:
        if self.is_ray_parallel():
            print("ERROR: Ray is parallel to the plane approximation")
            return False
        
        if not self.is_origin_in_range():
            print("ERROR: Origin is out of range")
            return False

        return True

    def is_ray_parallel(self) -> bool:
        """Checks if the given ray is parallel to the plane approximation"""
        a, b, c, _ = self.__fit_plane_in_pointcloud()
        normal = np.array([a, b, c])
        
        # Check if the dot product of the normal and the ray direction is close to zero
        dot_product = np.dot(normal, self.direction)
        return np.isclose(dot_product, 0)

    def is_origin_in_range(self) -> bool:
        return True

    def __fit_plane_in_pointcloud(self):
        # We use PCA to find the normal of the plane
        pca = PCA(n_components=3)
        pca.fit(self.pointcloud)
        normal = pca.components_[-1]

        # Plane equation coefficients: ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, pca.mean_)
        return a, b, c, d

    def find_closest_point(self):
        # Normalize the direction vector
        self.direction = self.direction / np.linalg.norm(self.direction)

        # Build a k-d tree for the DEM points
        tree = KDTree(self.pointcloud[:, :3])
        
        # Generate a set of points along the ray direction
        t_values = np.linspace(0, 1000, 10000)  # Adjust range and number of points if necessary
        ray_points = self.origin + np.outer(t_values, self.direction)
        
        # Find the closest DEM point to each ray point
        distances, indices = tree.query(ray_points)
        
        # Get the index of the minimum distance
        closest_index = indices[np.argmin(distances)]
        closest_point = self.pointcloud[closest_index]
        
        return closest_point


def parse_args(args) -> np.ndarray:
    """Parse input values to origin and direction of a three-dimensional ray"""
    x = float(args[1])
    y = float(args[2])
    z = float(args[3])
    vx = float(args[4])
    vy = float(args[5])
    vz = float(args[6])
    return np.array([x, y, z]), np.array([vx, vy, vz])


def main():
    # Check if the correct number of arguments are provided
    # if len(sys.argv) != 7:
    #     print("Usage: python ray_intersection.py <x> <y> <z> <vx> <vy> <vz>")
    #     sys.exit(1)
    # origin, direction = parse_args(sys.argv)

    # Harcoded values
    origin = np.array([599_400, 5_287_400, 660])
    direction = np.array([400, 200, 70])

    # Compute closest DEM point
    inter = IntersectionChecker(origin, direction, DEM_PATH)
    closest_point = inter.find_closest_point()

    print("Closest point on the DEM to the ray:", closest_point)

    # Step 4: Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot only a subset of DEM points for clarity
    subset_indices = np.random.choice(len(inter.pointcloud), size=min(len(inter.pointcloud), 10000), replace=False)
    # subset_indices = np.arange(len(inter.pointcloud))  # This will select all indices

    
    ax.scatter(inter.pointcloud[subset_indices, 0], inter.pointcloud[subset_indices, 1], inter.pointcloud[subset_indices, 2],
               s=1, c='b', alpha=0.5, label='DEM Points (Subset)')
    
    # Plot ray
    ax.plot([inter.origin[0], inter.origin[0] + direction[0]],
            [inter.origin[1], inter.origin[1] + direction[1]],
            [inter.origin[2], inter.origin[2] + direction[2]], 'r-', label='Ray')
    
    # Plot closest DEM point if found
    if closest_point is not None:
        ax.scatter(closest_point[0], closest_point[1], closest_point[2], color='k',
                   label='Closest DEM Point')
    
    # Set axis limits based on data range
    x_min, x_max = np.min(inter.pointcloud[:, 0]), np.max(inter.pointcloud[:, 0])
    y_min, y_max = np.min(inter.pointcloud[:, 1]), np.max(inter.pointcloud[:, 1])
    z_min, z_max = np.min(inter.pointcloud[:, 2]), np.max(inter.pointcloud[:, 2])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title('Ray-Plane Intersection with DEM Visualization')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
