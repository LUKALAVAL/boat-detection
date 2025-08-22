import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import pleiades_to_planet


if __name__ == "__main__":

    # The Pléiades images are transformed to simulate PlanetScope style
    # First downsampling and then blurring using the function pleiades_to_planet

    target_resolution = 3 # in m/px
    blur_sigma = 1.0

    input_dir = "PLE/A_dataset/images"
    output_dir = "PLE/A_dataset/images_transformed"
    os.makedirs(output_dir, exist_ok=True)

    list_files = os.listdir(input_dir)
    list_files = [f for f in list_files if f.endswith('.tif')]
    list_files.sort()

    for i, filename in enumerate(list_files):

        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        print(f"\nProcessing {i+1}/{len(list_files)}: {input_file} -> {output_file}")
        pleiades_to_planet(input_file, output_file, target_resolution=target_resolution, blur_sigma=blur_sigma)

