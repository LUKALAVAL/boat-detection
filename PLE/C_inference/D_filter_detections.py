import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import filter_by_cloudmask, filter_by_clusters, filter_by_confidence, filter_by_geometry, filter_by_landmask
import geopandas as gpd
import pandas as pd




if __name__ == "__main__":

    # Filter the detections using land mask, cloud mask, clusters, geometry and confidence

    area_ids = [19]
    dir_landmasks = "geodata/landmask"
    dir_cloudmasks = "PLE/C_inference/masks"
    dir_detections = "PLE/C_inference/detections"
    dir_output = "PLE/C_inference/detections_filtered"

    for area_id in area_ids:
        file_csv = os.path.join(dir_detections, f"grouped_{area_id}.csv")
        file_landmask = os.path.join(dir_landmasks, f"landmask_{area_id}.gpkg")
        file_landmask_buffered = os.path.join(dir_landmasks, f"landmask_buffer_{area_id}.gpkg")

        # Load detections
        gdf_detections = gpd.read_file(file_csv)
        gdf_detections = gpd.GeoDataFrame(gdf_detections, geometry=gpd.points_from_xy(gdf_detections['longitude'], gdf_detections['latitude']))
        gdf_detections.crs = "EPSG:4326"
        gdf_detections['class_id'] = pd.to_numeric(gdf_detections['class_id'], errors='coerce')
        gdf_detections['latitude'] = pd.to_numeric(gdf_detections['latitude'], errors='coerce')
        gdf_detections['longitude'] = pd.to_numeric(gdf_detections['longitude'], errors='coerce')
        gdf_detections['length'] = pd.to_numeric(gdf_detections['length'], errors='coerce')
        gdf_detections['breadth'] = pd.to_numeric(gdf_detections['breadth'], errors='coerce')
        gdf_detections['heading'] = pd.to_numeric(gdf_detections['heading'], errors='coerce')
        gdf_detections['confidence'] = pd.to_numeric(gdf_detections['confidence'], errors='coerce')
        gdf_detections['aspect_ratio'] = gdf_detections['breadth'] / gdf_detections['length'] # add aspect ratio column
        print("Detections", len(gdf_detections))
        
        # Filter by landmask
        gdf_landmask = gpd.read_file(file_landmask)
        gdf_landmask.to_crs(epsg=gdf_detections.crs.to_epsg(), inplace=True)
        gdf_detections = filter_by_landmask(gdf_detections, gdf_landmask, min_length=0)
        print("Landmask", len(gdf_detections))

        # Filter by landmask with buffer
        gdf_landmask_buffered = gpd.read_file(file_landmask_buffered)
        gdf_landmask_buffered.to_crs(epsg=gdf_detections.crs.to_epsg(), inplace=True)
        gdf_detections = filter_by_landmask(gdf_detections, gdf_landmask_buffered, min_length=500)
        print("Landmask buffer", len(gdf_detections))

        # Remove weird clusters
        gdf_detections = filter_by_clusters(gdf_detections)
        print("Clusters", len(gdf_detections))

        # Filter by cloudmask
        gdf_detections = filter_by_cloudmask(gdf_detections, dir_cloudmasks)
        print("Cloudmask", len(gdf_detections))

        # Filter by geometry
        gdf_detections = filter_by_geometry(gdf_detections)
        print("Geometry", len(gdf_detections))

        # Filter by confidence
        gdf_detections = filter_by_confidence(gdf_detections)
        print("Confidence", len(gdf_detections))

        # Save filtered detections
        output_file = os.path.join(dir_output, f"filtered_{area_id}.csv")
        gdf_detections = gdf_detections.drop(columns=['geometry', 'aspect_ratio'])
        gdf_detections.to_csv(output_file, index=False)



