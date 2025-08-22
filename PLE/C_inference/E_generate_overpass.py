import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import stack_binary_extents_georeferenced


if __name__ == "__main__":

    # Generate overpass files to later make statistics and build the boat density raster layer

    area_ids = [19]
    for area_id in area_ids:
        stack_binary_extents_georeferenced(
            dir_rasters="PLE/C_inference/images", 
            output_path=f"PLE/C_inference/overpass/{area_id}_overpass.tif",
            compression=100, # 1 = no compression
            area_id=area_id
        )