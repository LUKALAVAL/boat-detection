import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import make_predictions
from sahi import AutoDetectionModel

if __name__ == "__main__":

    # The images are annotated using a general model to detect objects on satellite images
    # Then the annotations are manually verified and modified using qgis

    tile_size = 1024
    overlap_ratio = 0.1

    images_dir = "PLE/A_dataset/images"
    output_dir = "PLE/A_dataset/images_labels"
    os.makedirs(output_dir, exist_ok=True)

    list_files = os.listdir(images_dir)
    list_files = [f for f in list_files if f.lower().endswith('.tif')]
    list_files.sort()

    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11x-obb.pt", # largest model trained on DOTAv1 dataset (will be automatically downloaded)
        confidence_threshold=0.3,
        device="cuda:0",
    )

    for i, image_file in enumerate(list_files):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(output_dir, image_file.replace('.tif', '.txt'))

        print(f"\nProcessing {i+1}/{len(list_files)}: {image_path}")
        make_predictions(image_path, label_path, detection_model, tile_size, overlap_ratio)
