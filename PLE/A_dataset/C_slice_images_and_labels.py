import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import slice_tif, slice_label    


if __name__ == "__main__":

    # The transformed images are sliced into tiles of 512 by 512 pixels
    # The labels as well so that they fit the image tiles

    #### SLICE IMAGES ####

    tile_size = 512
    remove_empty_tiles = True

    dir_input = "PLE/A_dataset/images_transformed"
    dir_output_tiles = "PLE/A_dataset/tiles"
    os.makedirs(dir_output_tiles, exist_ok=True)

    list_files = os.listdir(dir_input)
    list_files = [f for f in list_files if f.endswith(('.tif'))]
    list_files.sort()

    for i, filename in enumerate(list_files[2:]):

        input_tif = os.path.join(dir_input, filename)

        print(f"Processing {i+1}/{len(list_files)}: {input_tif}")
        slice_tif(input_tif, dir_output_tiles, tile_size=tile_size, remove_empty_tiles=remove_empty_tiles)

    #### SLICE LABELS ####

    tile_size = 512
    margin = 2 # in pixels (1 pixel = 3 meters)

    dir_labels = "PLE/A_dataset/images_labels"
    dir_rasters = "PLE/A_dataset/images_transformed"
    dir_output_labels = "PLE/A_dataset/tiles_labels"
    os.makedirs(dir_output_labels, exist_ok=True)

    list_files = os.listdir(dir_labels)
    list_files = [f for f in list_files if f.endswith(('.txt'))]
    list_files.sort() 

    for i, filename in enumerate(list_files):

        input_label = os.path.join(dir_labels, filename)
        input_raster = os.path.join(dir_rasters, filename.replace('.txt', '.tif'))

        print(f"Processing {i+1}/{len(list_files)}: {input_label} with raster {input_raster}")
        slice_label(input_label, input_raster, dir_output_labels, tile_size=tile_size, margin=margin)

    # Only keep labels if it has a tile associated to it
    list_files = os.listdir(dir_output_labels)
    list_files = [f for f in list_files if f.endswith(('.txt'))]
    list_files.sort()

    for filename in list_files:
        input_label = os.path.join(dir_output_labels, filename)
        input_raster = os.path.join(dir_output_tiles, filename.replace('.txt', '.tif'))

        if not os.path.exists(input_raster):
            print(f"Warning: No corresponding raster for {input_label}, removing label file.")
            os.remove(input_label)

        else:
            print(f"Keeping label file: {input_label}")