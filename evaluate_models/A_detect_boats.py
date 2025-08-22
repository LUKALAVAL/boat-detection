import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import make_predictions, txt_to_csv

from sahi import AutoDetectionModel



# def get_distance(coord1, coord2):
#     # Calculate the geographic distance between two coordinates
#     return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)
    




if __name__ == "__main__":

    input_image = "evaluate_models/planet_rendered.tif"
    output_dir = "evaluate_models/detections"
    overlap_ratio = 0.1
    confidence_threshold = 0.001
    pixel_resolution = 3 # meters per pixel


    #---------------------------------- DOT MODEL ----------------------------------#
    # Parameters
    output_txt = os.path.join(output_dir, "dot_detections.txt")
    output_csv = output_txt.replace('.txt','.csv')
    model = "DOT/weights.pt"
    tile_size = 1024
    dict_class = {
        1: 0, # class id 1 corresponds to boats/ships in the DOTAv1 dataset
    }

    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
        confidence_threshold=confidence_threshold,
        device="cuda:0",
    )

    # Make predictions
    make_predictions(
        input_image,
        output_txt,
        detection_model,
        tile_size,
        overlap_ratio
    )

    # Convert to CSV
    txt_to_csv(output_txt, input_image, output_csv, dict_class, pixel_resolution)



    #---------------------------------- PLA MODEL ----------------------------------#
    # Parameters
    output_txt = os.path.join(output_dir, "pla_detections.txt")
    output_csv = output_txt.replace('.txt','.csv')
    model = "PLA/weights.pt"
    tile_size = 512
    overlap_ratio = 0.1
    confidence_threshold = 0.001
    dict_class = {
        0: 0, # class id 0 corresponds to (small) boats
        1: 1, # class id 1 corresponds to boat wakes
        2: 0, # class id 2 corresponds to (large) ships
    }

    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
        confidence_threshold=confidence_threshold,
        device="cuda:0",
    )

    # Make predictions
    make_predictions(
        input_image,
        output_txt,
        detection_model,
        tile_size,
        overlap_ratio
    )

    # Convert to CSV
    txt_to_csv(output_txt, input_image, output_csv, dict_class, pixel_resolution)



    #---------------------------------- PLE MODEL ----------------------------------#
    # Parameters
    output_txt = os.path.join(output_dir, "ple_detections.txt")
    output_csv = output_txt.replace('.txt','.csv')
    model = "PLE/weights.pt"
    tile_size = 512
    overlap_ratio = 0.1
    confidence_threshold = 0.001
    dict_class = {
        0: 0, # class id 0 corresponds to boats
        1: 1, # class id 1 corresponds to boat wakes
    }
    adjust = -12 # Adjust size as the boxes were enlarged by 2*2*3=12 meters during training

    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
        confidence_threshold=confidence_threshold,
        device="cuda:0",
    )

    # Make predictions
    make_predictions(
        input_image,
        output_txt,
        detection_model,
        tile_size,
        overlap_ratio
    )

    # Convert to CSV
    txt_to_csv(output_txt, input_image, output_csv, dict_class, pixel_resolution, adjust)