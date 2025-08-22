import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import inference_planet, txt_to_geojson, txt_to_csv, group_csv
from sahi import AutoDetectionModel




if __name__ == "__main__":

    # Detect boats on a directory of planet images

    ordered_bands = [3, 2, 1]  # order bands in RGB
    tile_size = 512
    overlap_ratio = 0.1

    dir_rasters = 'PLE/C_inference/images'
    dir_output_predictions = 'PLE/C_inference/detections'
    os.makedirs(dir_output_predictions, exist_ok=True)

    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="PLE/weights.pt",
        confidence_threshold=0.001,
        device="cuda:0",
    )

    list_rasters = os.listdir(dir_rasters)
    list_rasters = [os.path.join(dir_rasters, f) for f in list_rasters if f.endswith('.tif')]

    for path_raster in list_rasters:
        print(f"Processing {path_raster}")
        path_prediction = inference_planet(path_raster, dir_output_predictions, detection_model, ordered_bands, tile_size, overlap_ratio)
        path_prediction = os.path.join(dir_output_predictions, os.path.basename(path_raster).replace('.tif', '.txt'))
        txt_to_geojson(
            input_txt=path_prediction,
            input_tif=path_raster,
            output_geojson=path_prediction.replace('.txt', '.geojson')
        )
        txt_to_csv(
            input_txt=path_prediction,
            input_tif=path_raster,
            output_csv=path_prediction.replace('.txt', '.csv'),
            dict_class = {0:0, 1:1},
            pixel_resolution=3,  # meters per pixel
            adjust=-12,  # adjust length and breadth by this amount
        )

    aoi_ids = [19]  # List of AOI IDs to process
    for aoi_id in aoi_ids:
        csv_files = os.listdir(dir_output_predictions)
        csv_files = [f for f in csv_files if f.endswith(".csv") and f.startswith(f"{aoi_id}_")]
        csv_files = [os.path.join(dir_output_predictions, f) for f in csv_files]
        group_csv(csv_files, os.path.join(dir_output_predictions, f"grouped_{aoi_id}.csv"))
