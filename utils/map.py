import numpy as np

class Map:
    """Class that stores a Digital Elevation Model as a discrete map"""
    def __init__(self, path2dem: str) -> None:
        self.path = path2dem
        self.map: np.ndarray = self.read_dem()
    

    def read_dem(self) -> np.ndarray:
        """Converts txt file into numpy array of shape nx3"""
        data = np.genfromtxt(self.path, delimiter=' ')
        return data