import pandas as pd
from pyproj import CRS, Transformer
from pathlib import Path

class ConvertLandData:
    def __init__(self, destination_path: Path, crs_from: str = "EPSG:3035", crs_to: str = "EPSG:4326", columns: list = ["easting", "northing", "height", "latitude", "longitude"]):
        """Class that automatically converts the coordinates from the LAEA to the WGS84 system.

        Args:
            destination_path (Path): Path where the converted files will be saved.
            crs_from (str, optional): Coordinate system of the land data files. Defaults to "EPSG:3035".
            crs_to (str, optional): Coordinate system to convert the land data files. Defaults to "EPSG:4326".
            columns (list, optional): Columns of the file to keep for the transformation. Defaults to ["easting", "northing", "height", "latitude", "longitude"].
        """
        self.destination_path = destination_path
        self.easting_col: str = columns[0]
        self.northing_col: str = columns[1]
        self.height_col: str = columns[2]
        self.latitude_col: str = columns[3]
        self.longitude_col: str = columns[4]
        self.crs_from = crs_from
        self.crs_to = crs_to
    
    def _defineTransformation(self):
        """Define the transformation from initial to final coordinate system.
        """
        coord_init = CRS(self.crs_from)
        coord_final = CRS(self.crs_to)

        # Create a transformer object for the conversion from LAEA to WGS84
        self.transformer = Transformer.from_crs(coord_init, coord_final, always_xy=True)
    
    def _getCoordinatesBoundaries(self, df: pd.DataFrame):
        """Get the maximum and minimum values of the easting and northing columns.

        Args:
            df (pd.DataFrame): file containing the land data, with WGS84 coordinates.

        Returns:
            list: coordinates boundaries
        """
        max_latitude = df[self.latitude_col].max()
        min_latitude = df[self.latitude_col].min()
        max_longitude = df[self.longitude_col].max()
        min_longitude = df[self.longitude_col].min()
        return max_latitude, min_latitude, max_longitude, min_longitude
    
    def transform(self, file_path: Path):
        """Convert the land data from LAEA to WGS84.
        
        Args:
            file_path (Path): path of the land data file.
        """
        self._defineTransformation()
        
        # check if the file exists
        if not file_path.exists():
            print(f"File {file_path} does not exist.")
            return
        df = pd.read_csv(file_path)
        file_name = file_path.stem
        longitude, latitude = self.transformer.transform(df[self.easting_col], df[self.northing_col])
        df1 = pd.DataFrame({self.latitude_col: latitude, self.longitude_col: longitude, self.height_col: df[self.height_col]})
        df1.to_csv(self.destination_path / (file_name + ".csv"), index=False)
        print(f"File {file_name} converted.")
        
    def getCoordinateBoundaries(self, file_paths: list):
        """Get the latitude and longitude boundaries of each file

        Args:
            file_paths (list): list of path to the files.
        """
        coordinates_boundaries = pd.DataFrame(columns= ["file_name", "max_latitude", "min_latitude", "max_longitude", "min_longitude"])
        for i, file in enumerate(file_paths):
            df = pd.read_csv(file)
            max_latitude, min_latitude, max_longitude, min_longitude = self._getCoordinatesBoundaries(df)
            coordinates_boundaries.loc[i] = [file, max_latitude, min_latitude, max_longitude, min_longitude]
        coordinates_boundaries.to_csv(self.destination_path + "coordinates_boundaries.csv", index=False)
            
    