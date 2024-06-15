from utils.map import Map

DEM_PATH = "dem.txt"

def main():
    dem = Map(DEM_PATH)
    print(dem.map.shape)


if __name__ == "__main__":
    main()