import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from typing import Tuple

DEM_PATH = "dem.txt"
VISUALIZATION = True

class IntersectionChecker:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, path2dem: str) -> None:
        self.origin = origin
        self.direction = direction
        self.path = path2dem
        self.pointcloud: np.ndarray = self.read_dem()
        # Pointcloud bounds
        self.min_x, self.min_y, self.min_z = None, None, None
        self.max_x, self.max_y, self.max_z = None, None, None
        self.__get_pointcloud_bounds()

        # Plane approximation coefficients
        self.a, self.b, self.c, self.d = None, None, None, None
        self.__fit_plane_in_pointcloud()
        self.closest_point = None

        if not self.is_ray_valid():
            if VISUALIZATION:
                self.plot_results()
            sys.exit(0)

    def read_dem(self) -> np.ndarray:
        """Use numpy to read DEM file efficiently"""
        print("Reading txt file ...")
        data = np.genfromtxt(self.path, delimiter=' ')
        return data

    def is_ray_valid(self) -> bool:
        """Finds the intersection point of the ray with the approximation plane and checks if it is within the bounds."""
        if not self.is_origin_near_bounds(5000):
            print("ERROR: Origin have to be near to the pointcloud bounds")
            self.__print_pointcloud_bounds()
            return False

        x0, y0, z0 = self.origin
        vx, vy, vz = self.direction

        # Calculate t for the ray-plane intersection
        denominator = self.a * vx + self.b * vy + self.c * vz
        if np.isclose(denominator, 0):
            print("ERROR: The ray is parallel to the plane")
            return False

        t = -(self.a * x0 + self.b * y0 + self.c * z0 + self.d) / denominator
        intersection_point = self.origin + t * self.direction

        # Check if the intersection point is within the plane bounds
        x_int, y_int, z_int = intersection_point
        if self.min_x <= x_int <= self.max_x and self.min_y <= y_int <= self.max_y:
            return True
        else:
            print("ERROR: Ray outside bounds of pointcloud")
            print(f"INFO: Intersection with approx. plane: (x:{x_int:.3f}, y:{y_int:.3f}, z:{z_int:.3f})")
            self.__print_pointcloud_bounds()
            return False

    def is_origin_near_bounds(self, threshold: float) -> bool:
        """
        Check if the given origin point is near the specified bounds within a threshold distance.
        - threshold: Maximum distance threshold to consider "nearby".
        """
        x, y, z = self.origin
        
        # Calculate distance to the nearest boundary
        distance_x = min(abs(x - self.min_x), abs(x - self.max_x))
        distance_y = min(abs(y - self.min_y), abs(y - self.max_y))
        distance_z = min(abs(z - self.min_z), abs(z - self.max_z))
        
        # Calculate Euclidean distance to the closest point on the boundary
        closest_distance = np.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
        
        # Check if the closest distance is within the threshold
        if closest_distance <= threshold:
            return True
        else:
            return False

    def __get_pointcloud_bounds(self) -> None:
        min_values = np.min(self.pointcloud, axis=0)
        max_values = np.max(self.pointcloud, axis=0)
        self.min_x, self.min_y, self.min_z = min_values
        self.max_x, self.max_y, self.max_z = max_values

    def __fit_plane_in_pointcloud(self) -> Tuple[float, float, float, float]:
        # We use PCA to find the normal of the plane
        pca = PCA(n_components=3)
        pca.fit(self.pointcloud)
        normal = pca.components_[-1]

        # Plane equation coefficients: ax + by + cz + d = 0
        self.a, self.b, self.c = normal
        self.d = -np.dot(normal, pca.mean_)
        return self.a, self.b, self.c, self.d

    def __print_pointcloud_bounds(self) -> None:
        print(f"INFO: Pointcloud bounds: (min_x: {self.min_x:.3f}, max_x: {self.max_x:.3f}), (min_y: {self.min_y:.3f}, max_y: {self.max_y:.3f}), (min_z:{self.min_z:.3f}, max_z:{self.max_z:.3f})")

    def find_closest_point(self) -> np.ndarray:
        # Normalize the direction vector
        norm_direction = self.direction / np.linalg.norm(self.direction)

        # Build a k-d tree for the DEM points
        tree = KDTree(self.pointcloud[:, :3])
        
        # Generate a set of points along the ray direction
        t_values = np.linspace(0, 1000, 10000)  # Adjust range and number of points if necessary
        ray_points = self.origin + np.outer(t_values, norm_direction)
        
        # Find the closest DEM point to each ray point
        distances, indices = tree.query(ray_points)
        
        # Get the index of the minimum distance
        closest_index = indices[np.argmin(distances)]
        self.closest_point = self.pointcloud[closest_index]
        
        return self.closest_point

    def plot_results(self, subset=True, ray=True, plane=True) -> None:
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if subset:
            # Plot only a subset of DEM points
            subset_indices = np.random.choice(len(self.pointcloud), size=min(len(self.pointcloud), 10000), replace=False)
        else:
            # Plot all DEM points
            subset_indices = np.arange(len(self.pointcloud))

        # Ploting pointcloud
        ax.scatter(self.pointcloud[subset_indices, 0], self.pointcloud[subset_indices, 1], self.pointcloud[subset_indices, 2],
                s=1, c='b', alpha=0.5, label='DEM Points (Subset)')
        
        if ray:
            # Plot ray
            ax.plot([self.origin[0], self.origin[0] + self.direction[0]],
                    [self.origin[1], self.origin[1] + self.direction[1]],
                    [self.origin[2], self.origin[2] + self.direction[2]], 'r-', label='Ray')
            
        if plane:
            # Plot plane
            xx, yy = np.meshgrid(np.linspace(self.min_x, self.max_x, 1000), np.linspace(self.min_y, self.max_y, 1000))
            zz = (-self.a * xx - self.b * yy - self.d) / self.c
            ax.plot_surface(xx, yy, zz, color='g', alpha=0.2)
            
        # Plot closest DEM point if found
        if self.closest_point is not None:
            ax.scatter(self.closest_point[0], self.closest_point[1], self.closest_point[2], color='k',
                    label='Closest DEM Point')
            
        # Set axis limits based on pointcloud bounds
        ax.set_xlim(self.min_x, self.max_x)
        ax.set_ylim(self.min_y, self.max_y)
        ax.set_zlim3d(self.min_z, self.max_z)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.title('Ray Intersection with DEM')
        plt.legend()
        plt.show()

def parse_args(args) -> Tuple[np.ndarray, np.ndarray]:
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
    if len(sys.argv) != 7:
        print("Usage: python3 intersection.py <x> <y> <z> <vx> <vy> <vz>")
        sys.exit(1)
    origin, direction = parse_args(sys.argv)

    # Compute closest DEM point
    inter = IntersectionChecker(origin, direction, DEM_PATH)
    inter.find_closest_point()

    print("Closest point on the DEM to the ray:", inter.closest_point)

    if VISUALIZATION:
        inter.plot_results()


if __name__ == "__main__":
    main()
