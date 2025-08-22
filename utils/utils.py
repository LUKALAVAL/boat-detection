import os
import subprocess
import random
import json
import csv
import math
import gc
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
# from shapely import Polygon
from shapely.geometry import Point, Polygon, mapping
from sklearn.cluster import DBSCAN
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.ndimage import gaussian_filter
from osgeo import gdal
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.enums import Resampling as ResampleEnum, ColorInterp
from sahi.predict import get_sliced_prediction






def read_large_tiff_image(image_path):

    # Read a large TIFF image and return it as a numpy array

    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))  # (bands, h, w) → (h, w, bands)
        image = image[:, :, :3]
    return image

def convert_sahi_predictions_to_yolo_obb(predictions, image_shape):

    # Convert SAHI predictions to YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4 confidence

    height, width = image_shape[:2]
    yolo_obb_lines = []

    for pred in predictions:

        # Use rotated box
        class_id = pred.category.id
        confidence = pred.score.value
        x1,y1,x2,y2,x3,y3,x4,y4 = pred.mask.segmentation[0] 

        # Normalize to [0,1]
        x1, y1 = x1 / width, y1 / height
        x2, y2 = x2 / width, y2 / height
        x3, y3 = x3 / width, y3 / height
        x4, y4 = x4 / width, y4 / height

        # Format as YOLO OBB line
        yolo_obb_lines.append(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f} {confidence:.6f}")

    return yolo_obb_lines

def make_predictions(image_file, labels_file, detection_model, tile_size, overlap_ratio):

    # Make predictions on a large TIFF image using sahi to slice the image

    print(f"Making predictions for {image_file}...")

    # Get sliced prediction
    image = read_large_tiff_image(image_file)
    result = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=False,
    )

    # Convert to YOLO-OBB format
    yolo_obb_lines = convert_sahi_predictions_to_yolo_obb(result.object_prediction_list, image.shape)

    # Save to labels file
    with open(labels_file, "w") as f:
        for line in yolo_obb_lines:
            f.write(line + "\n")

    # Cleanup to prevent memory leaks
    del image
    del result
    torch.cuda.empty_cache()
    gc.collect()











def downsample(input_file, output_file, target_resolution=3):

    # downsamples the input raster to a target resolution

    # Open input dataset to read geotransform
    input_ds = gdal.Open(input_file)
    if input_ds is None:
        raise RuntimeError(f"Failed to open input file: {input_file}")
    
    gt = input_ds.GetGeoTransform()

    # Compute appropriate resolution
    x_res = target_resolution * 10 ** (math.floor(math.log10(abs(gt[1]))) + 1)
    y_res = target_resolution * 10 ** (math.floor(math.log10(abs(gt[5]))) + 1)

    # Construct gdalwarp command
    cmd = [
        "gdalwarp",
        "-tr", str(x_res), str(y_res),
        "-r", "bilinear",
        "-of", "GTiff",
        input_file,
        output_file
    ]

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Downsampled {input_file} to {output_file} with target resolution approx {target_resolution} meters")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdalwarp failed: {e}")
    
def blur(input_file, output_file, sigma=1.0):

    # Blurs the input raster using gaussian blur of sigma

    with rasterio.open(input_file) as src:
        profile = src.profile
        bands = src.read()  # read all bands as (bands, rows, cols) numpy array

    # Prepare output array of the same shape
    filtered = np.empty_like(bands, dtype=np.float32)

    for i in range(bands.shape[0]):  # loop over bands
        band = bands[i]
        filtered_band = gaussian_filter(
            band,
            sigma=sigma,
            order=0,
            mode='constant',
            cval=0.0,
            truncate=4.0
        )

        # Cast filtered band to original dtype if integer
        original_dtype = profile['dtype']
        if np.issubdtype(np.dtype(original_dtype), np.integer):
            filtered_band = np.clip(filtered_band, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
            filtered_band = filtered_band.astype(original_dtype)
        else:
            filtered_band = filtered_band.astype(np.float32)

        filtered[i] = filtered_band

    # Update dtype in profile for floats if needed
    if not np.issubdtype(np.dtype(profile['dtype']), np.integer):
        profile.update(dtype=rasterio.float32)

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(filtered)

    print(f"Blurred {input_file} to {output_file} with sigma={sigma}")


def pleiades_to_planet(input_file, output_file, target_resolution=3, blur_sigma=1):

    # Converts a pleaides image to a planet-look-alike image

    temp_file = f"{output_file}.tmp.tif"

    downsample(input_file, temp_file, target_resolution=target_resolution)
    blur(temp_file, output_file, sigma=blur_sigma)
    os.remove(temp_file)  # remove temporary file to save space










def pad_array(arr, target_height, target_width):

    # Pads the input array to the target height and width

    pad_height = target_height - arr.shape[1]
    pad_width = target_width - arr.shape[2]
    if pad_height == 0 and pad_width == 0:
        return arr
    pad = ((0, 0), (0, pad_height), (0, pad_width))
    return np.pad(arr, pad, mode='constant', constant_values=0)

def slice_tif(input_path, output_dir, tile_size, remove_empty_tiles=False):

    # Slices the raster input into tiles and optionnaly remove the empty ones

    print(f"Slicing TIFF file: {input_path}")
    
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        width = src.width
        height = src.height
        n_cols = math.ceil(width / tile_size)
        n_rows = math.ceil(height / tile_size)

        for row in range(n_rows):
            for col in range(n_cols):
                x_off = col * tile_size
                y_off = row * tile_size
                win_width = min(tile_size, width - x_off)
                win_height = min(tile_size, height - y_off)

                window = Window(x_off, y_off, win_width, win_height)
                data = src.read(window=window)

                # Check if the tile is empty
                if remove_empty_tiles and np.all(data == 0):
                    continue # Skip empty tiles

                # Pad if needed
                if win_width < tile_size or win_height < tile_size:
                    data = pad_array(data, tile_size, tile_size)

                # Update transform
                transform = src.window_transform(window)

                out_meta = meta.copy()
                out_meta.update({
                    "height": tile_size,
                    "width": tile_size,
                    "transform": transform
                })

                out_path = os.path.join(
                    output_dir,
                    f"{os.path.basename(input_path)[:-4]}_{x_off}_{y_off}.tif"
                )

                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(data)


def denormalize_coords(coords, image_width, image_height):

    # Denormalizes coordinates from [0, 1] range to pixel values

    denormalized = []
    for i, coord in enumerate(coords):
        if i % 2 == 0:  # x coordinate
            denormalized.append(coord * image_width)
        else:  # y coordinate
            denormalized.append(coord * image_height)
    return denormalized

def normalize_coords(coords, tile):

    # Normalizes coordinates from pixel values to [0, 1] range

    tile_origin_x, tile_origin_y, tile_width, tile_height = tile

    normalized = []
    for i, coord in enumerate(coords):
        if i % 2 == 0:  # x coordinate
            n_coord = (coord - tile_origin_x) / tile_width
        else:  # y coordinate
            n_coord = (coord - tile_origin_y) / tile_height
        # normalized.append(max(0, min(1, n_coord))) # Ensure normalized value is between 0 and 1
        normalized.append(n_coord)  # Allow values outside [0, 1] for YOLO format
    return normalized

def overlap_area(rect1, rect2):

    # Computes the overlap area between two rectangles

    poly1 = Polygon(rect1)
    poly2 = Polygon(rect2)

    # Compute intersection
    intersection = poly1.intersection(poly2)

    if intersection.is_empty:
        return 0.0

    return intersection.area / min(poly1.area, poly2.area)

def add_margin_to_oriented_rectangle(coords, margin):

    # Adds margin to a bounding box

    # Convert to 4x2 array of points
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
    
    # Compute center of rectangle
    center = np.mean(pts, axis=0)

    # Compute edge directions
    edge1 = pts[1] - pts[0]
    edge2 = pts[3] - pts[0]

    # Normalize directions
    dir1 = edge1 / np.linalg.norm(edge1)
    dir2 = edge2 / np.linalg.norm(edge2)

    # Expand each point along dir1 and dir2 from center
    new_pts = []
    for pt in pts:
        offset = pt - center
        proj_dir1 = np.dot(offset, dir1)
        proj_dir2 = np.dot(offset, dir2)
        new_offset = (proj_dir1 + np.sign(proj_dir1) * margin) * dir1 + \
                     (proj_dir2 + np.sign(proj_dir2) * margin) * dir2
        new_pts.append(center + new_offset)

    return np.array(new_pts).flatten()


def slice_label(file_label, file_raster, output_dir, tile_size=512, margin=2):

    # Slices the labels to fit the tiles optionnaly with a margin in pixels

    print(f"Slicing label file: {file_label} with raster {file_raster}")

    with rasterio.open(file_raster) as src:
        width = src.width
        height = src.height

    with open(file_label, 'r') as f:
        lines = f.readlines()
    
    n_cols = math.ceil(width / tile_size)
    n_rows = math.ceil(height / tile_size)
    tiles = [
        (col * tile_size, row * tile_size, tile_size, tile_size)
        for row in range(n_rows)
        for col in range(n_cols)
    ]

    for tile in tiles:

        tile_origin_x, tile_origin_y, tile_width, tile_height = tile

        rect_tile = [
            (tile_origin_x, tile_origin_y),
            (tile_origin_x + tile_width, tile_origin_y),
            (tile_origin_x + tile_width, tile_origin_y + tile_height),
            (tile_origin_x, tile_origin_y + tile_height)
        ]

        # Create output path for the tile
        out_path = os.path.join(output_dir, f"{os.path.basename(file_label)[:-4]}_{tile_origin_x}_{tile_origin_y}.txt")

        with open(out_path, 'w') as out_f:
            for line in lines:
                parts = line.strip().split(' ')
                cls = parts[0]
                coords = list(map(float, parts[1:]))

                # Denormalize coordinates
                coords = denormalize_coords(coords, image_width=width, image_height=height)

                # Calculate the intersection area with the tile in orientation
                rect_label = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                area = overlap_area(rect_tile, rect_label)

                if area > 0.3:  # If more than 30% overlap
                    coords = normalize_coords(coords, tile=tile)
                    coords_with_margins = add_margin_to_oriented_rectangle(coords, margin=margin)  # Add margin (2 pixels = 6 meters)
                    out_f.write(f"{cls} {' '.join(map(str, coords_with_margins))}\n")








def create_dataset(dir_images, dir_labels, dir_output, train_val_test_split):

    # Construct the dataset directory structure with the repartition below

    train,val,test = train_val_test_split
    assert train + val + test == 1, "Train, validation, and test splits must sum to 1."

    list_images = os.listdir(dir_images)
    list_images = [f for f in list_images if f.endswith(('.tif'))]
    list_images.sort()
    list_labels = os.listdir(dir_labels)
    list_labels = [f for f in list_labels if f.endswith(('.txt'))]
    list_labels.sort()
    assert len(list_images) == len(list_labels), "Number of images and labels must match."

    os.makedirs(os.path.join(dir_output, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "images/test"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "labels/val"), exist_ok=True)
    os.makedirs(os.path.join(dir_output, "labels/test"), exist_ok=True)

    num_files = len(list_images)
    indices = list(range(num_files))
    random.seed(42) # Fixed seed
    random.shuffle(indices)

    train_end = int(train * num_files)
    val_end = int((train + val) * num_files)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Copy the files to the dataset not only move
    for i in train_indices:
        os.system(f"cp {os.path.join(dir_images, list_images[i])} {os.path.join(dir_output, 'images/train')}")
        os.system(f"cp {os.path.join(dir_labels, list_labels[i])} {os.path.join(dir_output, 'labels/train')}")

    for i in val_indices:
        os.system(f"cp {os.path.join(dir_images, list_images[i])} {os.path.join(dir_output, 'images/val')}")
        os.system(f"cp {os.path.join(dir_labels, list_labels[i])} {os.path.join(dir_output, 'labels/val')}")

    for i in test_indices:
        os.system(f"cp {os.path.join(dir_images, list_images[i])} {os.path.join(dir_output, 'images/test')}")
        os.system(f"cp {os.path.join(dir_labels, list_labels[i])} {os.path.join(dir_output, 'labels/test')}")


def create_yaml_file(dir_output):

    # Create a YAML file for dataset splits
    # The function must be called after creating the dataset structure
    #
    # Add more flexibility by allowing custom classes

    yaml_content = f"""train: images/train
val: images/val
test: images/test

names: 
  0: boat
  1: wake
"""

    with open(os.path.join(dir_output, "dataset.yaml"), "w") as f:
        f.write(yaml_content)







def render_planet(input_tif, output_tif, ordered_bands):

    # render the planet image to be used for inference

    with rasterio.open(input_tif) as src:
        profile = src.profile
        bands_data = []

        data = src.read(ordered_bands)  
        print(data.shape)  # Debugging: print shape of data


        for i in range(len(ordered_bands)):
            band = data[i].astype(np.float32)
            band_min = np.percentile(band, 0.01)
            band_max = np.percentile(band, 99.99)

            print(f"Band {i}: min = {band_min}, max = {band_max}")

            # Avoid division by zero
            if band_max - band_min == 0:
                stretched = np.zeros(band.shape, dtype=np.uint8)
            else:
                # Stretch to 0–255
                stretched = ((band - band_min) / (band_max - band_min)) * 255.0
                stretched = np.clip(stretched, 1, 254)
                stretched = stretched.astype(np.uint8)
            
            new_band_max = stretched.max()
            new_band_min = stretched.min()
            print(f"Stretched Band {i}: new min = {new_band_min}, new max = {new_band_max}")

            bands_data.append(stretched)


        # Update profile for 8-bit output
        profile.update(
            dtype=rasterio.uint8,
            count=len(bands_data)
        )

        with rasterio.open(output_tif, 'w', **profile) as dst:
            for i, stretched_band in enumerate(bands_data, start=1):
                dst.write(stretched_band, i)

    print(f"Stretched image written to {output_tif}")

def parse_obb_line(line, width, height):
    # Parse and denormalize YOLO OBB line: class x1 y1 x2 y2 x3 y3 x4 y4
    parts = line.strip().split()
    class_id = int(parts[0])
    # Denormalize coordinates
    coords = [float(parts[i]) * (width if i % 2 == 1 else height) for i in range(1, 9)]
    confidence = float(parts[9]) if len(parts) > 9 else 1.0
    return class_id, coords, confidence

def georeference_obb(coords, transform):
    # Convert all 4 points of the OBB to geographic coordinates.
    return [rasterio.transform.xy(transform, coords[i], coords[i + 1]) for i in range(0, 8, 2)]

def txt_to_geojson(input_txt, input_tif, output_geojson):

    # Converts yolo txt predictions to georeferenced geojson format

    with open(input_txt, 'r') as f:
        lines = f.readlines()

    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        features = []
        for line in lines:
            class_id, coords, confidence = parse_obb_line(line, width, height)
            geo_coords = georeference_obb(coords, transform)
            polygon = Polygon(geo_coords)

            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {"class_id": class_id,
                               "confidence": confidence,}
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "crs": {
                "type": "name",
                "properties": {
                    "name": crs.to_string()
                }
            }
        }

        with open(output_geojson, 'w') as out:
            json.dump(geojson, out, indent=2)



def georeference_point(transform, height, width, x, y):

    # Convert normalized coordinates (0-1) to geographic coordinate
    # using the transform, width and height parameters of the raster tif

    x, y = x * height, y * width
    geo_x, geo_y = rasterio.transform.xy(transform, x, y)
    return (geo_x, geo_y)

def get_distance(coord1, coord2):

    # Returns the distance between two georeferenced points in meters

    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    distance = math.sqrt(dx*dx + dy*dy)
    # distance is in feet, convert to meters
    return distance * 0.3048

def get_heading(lat1, lon1, lat2, lon2):

    # Calculates the heading of the two sets of coordinates
    # Returns heading in degrees

    return math.degrees(math.atan2(lat2 - lat1, lon2 - lon1))


def get_length_breadth_heading(transform, height, width, pixel_resolution, coords):

    # Returns the length, breadth and heading of the bounding box from the georeferenced coordinates

    max_edge_length = 0
    min_edge_length = float('inf')
    for i in range(4):
        p1 = coords[i]
        p2 = coords[(i + 1) % 4]
        gp1 = georeference_point(transform, height, width, p1[1], p1[0])
        gp2 = georeference_point(transform, height, width, p2[1], p2[0])

        edge_length = get_distance(gp1, gp2)
        if edge_length > max_edge_length:
            max_edge_length = edge_length
            heading = get_heading(gp1[1], gp1[0], gp2[1], gp2[0]) % 180
        if edge_length < min_edge_length:
            min_edge_length = edge_length

    length = max_edge_length * pixel_resolution
    breadth = min_edge_length * pixel_resolution

    return length, breadth, heading

def get_center_point(coords):

    # Returns the center point of the bounding box from the georeferenced coordinates

    x = [coord[1] for coord in coords]
    y = [coord[0] for coord in coords]
    center_x = np.mean(x)
    center_y = np.mean(y)
    return center_x, center_y


def txt_to_csv(input_txt, input_tif, output_csv, dict_class, pixel_resolution, adjust=0):

    # Constructs the csv file from the input txt (yolo annotations with confidence) and the raster tif file
    # The dict_class parameter is used to filter detections based on class id and rename them
    # eg. {0: 0, 1: 0} means class id 0 and 1 are kept and renamed to 0 and 0 respectively
    # the other classes are removed

    with rasterio.open(input_tif) as src:
        tif_width, tif_height = src.width, src.height
        tif_transform = src.transform
        tif_crs = src.crs

    with open(input_txt, 'r') as f:
        lines = f.readlines()

    with open(output_csv, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class_id', 'latitude', 'longitude', 'confidence', 'length', 'breadth', 'heading'])

        for line in lines:
            
            parts = line.strip().split()

            class_id = int(parts[0])
            if class_id not in dict_class:
                continue # removes detection if class not in dict_class

            # Map old class id to new class id
            new_class_id = dict_class[class_id]

            # Get coordinates and geometry
            coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, 8, 2)]
            center_x, center_y = get_center_point(coords)
            longitude, latitude = georeference_point(tif_transform, tif_height, tif_width, center_x, center_y)
            length, breadth, heading = get_length_breadth_heading(tif_transform, tif_height, tif_width, pixel_resolution, coords)
            length += adjust
            breadth += adjust
            confidence = float(parts[-1])

            # Round values
            confidence = round(confidence, 3)
            length = round(length, 1)
            breadth = round(breadth, 1)
            heading = round(heading, 0)

            # Write to CSV
            writer.writerow([new_class_id, latitude, longitude, confidence, length, breadth, heading])
    csvfile.close()

    # Change projection to EPSG:4326 to ensure data compatibility
    gdf = gpd.read_file(output_csv)
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude))
    gdf.crs = tif_crs
    gdf.to_crs("EPSG:4326", inplace=True)
    gdf['latitude'] = gdf.geometry.y.round(8)
    gdf['longitude'] = gdf.geometry.x.round(8)
    gdf.drop(columns=['geometry'], inplace=True)
    gdf.to_csv(output_csv, index=False)



def inference_planet(image_file, prediction_dir, detection_model, ordered_bands, tile_size, overlap_ratio):

    # Function to predict boat locations on a raw raster image
    # and save the predictions to a text file in the yolo obb format.

    rendered_image_path = image_file.replace('.tif', '_rendered.tif')
    prediction_file = os.path.join(prediction_dir, f"{os.path.basename(image_file).replace('.tif', '.txt')}")

    if os.path.exists(prediction_file):
        print(f"Prediction file already exists: {prediction_file}")
    else:
        print(f"Rendering image: {image_file} to {rendered_image_path}")
        render_planet(image_file, rendered_image_path, ordered_bands)
        make_predictions(rendered_image_path, prediction_file, detection_model, tile_size, overlap_ratio)
        os.remove(rendered_image_path)
    
    return prediction_file

def group_csv(list_csv, output_csv):

    # group csv files to make one

    all_rows = []

    for csv_file in list_csv:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            aoi_id = int(os.path.basename(csv_file).split('_')[0])
            date = int(os.path.basename(csv_file).split('_')[1])
            source = os.path.basename(csv_file).replace('.csv', '')
            for row in reader:
                row.append(date)
                row.append(aoi_id)
                row.append(source)
                all_rows.append(row)
                

    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'latitude', 'longitude', 'confidence', 'length', 'breadth', 'heading', 'date', 'aoi_id', 'source'])
        writer.writerows(all_rows)











def filter_by_landmask(gdf_detections, gdf_landmask, min_length=0):
    
    # filters the detections (geopandas dataframe) using a landmask polygon

    gdf_bigger = gdf_detections[gdf_detections['length'] > min_length]
    gdf_bigger = gpd.sjoin(gdf_bigger, gdf_landmask, how='left', predicate='intersects')
    gdf_bigger = gdf_bigger[gdf_bigger['index_right'].isnull()].drop(columns='index_right')

    gdf_smaller = gdf_detections[gdf_detections['length'] <= min_length]

    return pd.concat([gdf_bigger, gdf_smaller], ignore_index=True)



def filter_by_confidence(df):

    # Filters the detections (dataframe) by confidence score

    df_boat = df[df['class_id'] == 0]
    df_wake = df[df['class_id'] == 1]

    df_wake = df_wake[(df_wake['confidence'] > 0.29)]
    df_boat = df_boat[(df_boat['confidence'] > 0.14)]

    return pd.concat([df_wake, df_boat], ignore_index=True)



def remove_clusters(df, params, eps=1.5, min_samples=5):

    # Removes clustered detections based on spatial features

    keys = list(params.keys())
    values = list(params.values())
    features = df[keys].copy()
    for key, value in zip(keys, values):
        features[key] /= value

    X = features.values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Keep only unclustered rows (label == -1)
    return df[labels == -1]



def filter_by_clusters(df):

    # Apply remove_clusters function using the specified params

    df_boat = df[df['class_id'] == 0]
    df_wake = df[df['class_id'] == 1]

    # Remove artefacts (repetitions in space with same geometry and orientation)
    df_big_boat = df_boat[(df_boat['length'] > 45)]
    df_small_boat = df_boat[(df_boat['length'] <= 45)]
    params = {
        "breadth": 1,
        "heading": 1,
        "aspect_ratio": 0.005,
    }
    df_big_boat = remove_clusters(df_big_boat, params, eps=1.5, min_samples=5)
    df_boat = pd.concat([df_big_boat, df_small_boat], ignore_index=True)

    # Remove obstacles (repetitions in time at the same location)
    df_big_boat = df_boat[(df_boat['length'] > 60)]
    df_small_boat = df_boat[(df_boat['length'] <= 60)]
    params = {
        "latitude": 1,
        "longitude": 1,
        "length": 1,
        "breadth": 1,
        "heading": 1,
    }
    df_big_boat = remove_clusters(df_big_boat, params, eps=1.5, min_samples=5)
    df_boat = pd.concat([df_big_boat, df_small_boat], ignore_index=True)

    dff = pd.concat([df_wake, df_boat], ignore_index=True)
    return dff
        


def filter_by_geometry(df):
    
    # Filter the detection dataframe by the length and aspect_ratio

    df_boat = df[df['class_id'] == 0]
    df_wake = df[df['class_id'] == 1]

    polygon = Polygon([(0.05,450), (0.05,100), (0.1,0), (1,0), (1,30), (0.7,50), (0.4,200), (0.3,450), (0.05,450)])
    mask = df_boat.apply(lambda row: polygon.contains(Point(row['aspect_ratio'], row['length'])), axis=1)
    df_boat = df_boat[mask]

    dff = pd.concat([df_wake, df_boat], ignore_index=True)
    return dff


def filter_by_cloudmask(gdf, dir_cloudmasks):

    # Removes the detections inside the cloud masks

    grouped = gdf.groupby('source')
    filtered_groups = []

    for source, group in grouped:
        cloudmask_file = os.path.join(dir_cloudmasks, f"{source}_udm2.tif")

        with rasterio.open(cloudmask_file) as src:
            cloudmask = src.read(6).astype(bool)
            cloudmask = ~cloudmask  # invert if needed

            group = group.to_crs(src.crs)

            def is_clear(geom):
                x, y = geom.x, geom.y
                row, col = src.index(x, y)
                return cloudmask[row, col]

            group = group[group.geometry.apply(is_clear)]

            # After filtering, reproject back to EPSG:4326
            if not group.empty:
                group = group.to_crs(epsg=4326)
                filtered_groups.append(group)

    # Combine results
    if filtered_groups:
        return gpd.GeoDataFrame(pd.concat(filtered_groups, ignore_index=True), crs="EPSG:4326")
    else:
        return gpd.GeoDataFrame(columns=gdf.columns, crs="EPSG:4326")
    







def stack_binary_extents_georeferenced(dir_rasters, output_path, compression, area_id=None):

    # Stack all the images extent as a binary matrix for statistics later

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect bounds, resolution, and CRS
    crs = None
    bounds = []
    resolutions = []

    list_files = os.listdir(dir_rasters)
    if area_id is not None:
        print(f"Applying to area {area_id}")
        list_files = [f for f in list_files if f.startswith(f"{area_id}_")]
    list_files = [os.path.join(dir_rasters, f) for f in list_files if f.endswith(".tif")]
    list_files.sort()

    print("\nChecking CRS and max bounds.")
    posid = 0
    for i, tif in enumerate(list_files):
        print(f"{i+1}/{len(list_files)} {os.path.basename(tif)}")

        with rasterio.open(tif) as src:

            if crs is None:
                crs = src.crs

            posid += 1
            bounds.append(src.bounds)
            res = src.res
            resolutions.append(res)
        src.close()


    # Use minimum resolution and apply reduction
    xres = compression
    yres = compression

    # Compute union bounds
    minx = min(b.left for b in bounds)
    maxx = max(b.right for b in bounds)
    miny = min(b.bottom for b in bounds)
    maxy = max(b.top for b in bounds)

    width = int(np.ceil((maxx - minx) / xres))
    height = int(np.ceil((maxy - miny) / yres))

    # New transform covering all rasters at reduced resolution
    transform = from_origin(minx, maxy, xres, yres)

    # Prepare output file
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': posid,
        'width': width,
        'height': height,
        'crs': crs,
        'transform': transform,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'deflate'
    }

    print("\nMerging the visit extents.")
    with rasterio.open(output_path, 'w', **profile) as dst:
        posid = 1
        for i, tif in enumerate(list_files, start=1):
            print(f"{i}/{len(list_files)} {os.path.basename(tif)}")

            with rasterio.open(tif) as src:

                # Create binary mask of valid data
                data = src.read(1, masked=True)
                if src.crs != crs:
                    continue
                binary = (~data.mask).astype(np.uint8)

                # Reproject to common grid and resolution
                dest = np.zeros((height, width), dtype=np.uint8)
                reproject(
                    source=binary,
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=ResampleEnum.nearest
                )

                # Write band to output file
                dst.write(dest, posid)

                # Extract date from filename
                date = tif.split('_')[1]
                dst.set_band_description(posid, date)
                posid += 1

        dst.colorinterp = [ColorInterp.gray] * (posid - 1)

    print(f"Saved stacked binary extent to: {output_path}")