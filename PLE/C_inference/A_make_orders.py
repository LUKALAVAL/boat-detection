from planet import Planet
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from shapely.geometry import Polygon
from datetime import timedelta
import json
import os



PLANET_API_KEY = "YOUR_PLANET_API_KEY"
os.environ['PL_API_KEY'] = PLANET_API_KEY
pl = Planet()



def search_images(start_date, end_date, coordinates_aoi):

    # Search for images in the given date range and area of interest (AOI)

    # Convert start and end dates to the required format
    formatted_start_date = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    formatted_end_date = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Define the search request payload
    request = {
        "item_types": [
            "PSScene"
        ],
        "filter": {
            "type": "AndFilter",
            "config": [
                {
                    "field_name": "geometry",
                    "type": "GeometryFilter",
                    "config": {
                        "coordinates": [coordinates_aoi],
                        "type": "Polygon"
                    }
                },
                {
                    "type": "PermissionFilter",
                    "config": [
                        "assets:download"
                    ]
                },
                {
                    "field_name": "acquired",
                    "type": "DateRangeFilter",
                    "config": {
                        "gte": formatted_start_date,
                        "lte": formatted_end_date
                    }
                },
                {
                    "field_name": "cloud_cover",
                    "type": "RangeFilter",
                    "config": {
                        "lte": 1.0
                    }
                },
                {
                    "type": "OrFilter",
                    "config": [
                        {
                            "config": [
                                {
                                    "config": [
                                        "basic_analytic_4b",
                                        "basic_analytic_8b",
                                        "ortho_analytic_4b",
                                        "ortho_analytic_4b_sr",
                                        "ortho_analytic_8b",
                                        "ortho_analytic_8b_sr"
                                    ],
                                    "type": "AssetFilter"
                                },
                                {
                                    "config": [
                                        "PSScene"
                                    ],
                                    "type": "StringInFilter",
                                    "field_name": "item_type"
                                },
                                {
                                    "field_name": "publishing_stage",
                                    "type": "StringInFilter",
                                    "config": [
                                        "standard",
                                        "finalized"
                                    ]
                                }
                            ],
                            "type": "AndFilter"
                        }
                    ]
                }
            ]
        }
    }

    # Submit the search request
    print(request)
    search_result = requests.post("https://api.planet.com/data/v1/quick-search",
                                  auth=HTTPBasicAuth(PLANET_API_KEY, ''),
                                  json=request)
    print(search_result)
    print(f"Search result status code: {search_result.status_code}")

    return search_result.json()

def place_order(file_orders, item_ids, coordinates_aoi, name="order_product"):

    # Place an order for the given item IDs and area of interest (AOI)
    # Items must be in the inverse order of importance
    # Best to use items from the same date

    # Define order payload
    order_payload = {
        "name": name,
        "products": [{
            "item_ids": item_ids,
            "item_type": 'PSScene',
            "product_bundle": 'analytic_sr_udm2'
        }],
        "tools": [
            {
                "composite": {},
            },
            {
                "clip": {
                    "aoi": {
                        "type": "Polygon",
                        "coordinates": [coordinates_aoi]
                    }
                }
            }
        ]
    }

    # Submit order
    order_response = requests.post("https://api.planet.com/compute/ops/orders/v2", 
                                   auth=HTTPBasicAuth(PLANET_API_KEY, ''), 
                                   json=order_payload)
    print(f"Order response status code: {order_response.status_code}")

    aoi = name.split('_')[1]
    date_from = name.split('_')[2]
    date_to = name.split('_')[3]
    data = order_response.json()
    try:
        order_id = data['id']
        date = data['products'][0]['item_ids'][0][:8]
        print(order_id, aoi, date, date_from, date_to)
    except:
        print(f"Error processing file: {name}")
        order_id = None
        date = None
        print(order_id, aoi, date, date_from, date_to)

    with open(file_orders, 'a') as f:
        if os.stat(file_orders).st_size == 0:
            f.write("order_id,aoi,date,date_from,date_to\n")
        f.write(f"{order_id},{aoi},{date},{date_from},{date_to}\n")

    return data

def get_aoi_coordinates(file_path):
    # Open a geojson file and return the coordinates of the polygon inside
    with open(file_path, 'r') as f:
        data = json.load(f)
    coordinates = data['features'][0]['geometry']['coordinates'][0][0]
    return coordinates

def split_items_by_day(items):
    # Split the items by day
    items_by_day = {}
    for item in items:
        date = item['properties']['acquired'][:10]
        if date not in items_by_day:
            items_by_day[date] = []
        items_by_day[date].append(item)
    items_by_day = dict(sorted(items_by_day.items()))
    return items_by_day

def order_items_by_cloud_cover(items):
    # Sort the items by cloud cover
    ordered_items = sorted(items, key=lambda x: x['properties']['cloud_cover'])
    ordered_items.reverse() # Sort in descending order for planet API
    return ordered_items

def get_ids(items):
    # Extract the IDs from the items
    ids = [item['id'] for item in items]
    return ids

def get_overlap(items, coordinates_aoi):
    # Calculate the overlap of each item with the area of interest
    poly_aoi = Polygon(coordinates_aoi)
    poly_overlap = None
    for item in items:
        poly_item = Polygon(item['geometry']['coordinates'][0])
        intersection = poly_item.intersection(poly_aoi) # item INT aoi
        poly_overlap = poly_overlap.union(intersection) if poly_overlap else intersection # overlap UNI (item INT aoi)
    return round(poly_overlap.area / poly_aoi.area, 2)

def get_cloud_cover(ordered_items, coordinates_aoi):
    # Calculate the cloud cover 
    items = ordered_items.copy()
    items.reverse()
    poly_aoi = Polygon(coordinates_aoi)
    poly_overlap = None
    contribution_list = []
    for item in items:
        poly_item = Polygon(item['geometry']['coordinates'][0])
        intersection = poly_item.intersection(poly_aoi) # item INT aoi
        contribution = intersection.difference(poly_overlap) if poly_overlap else intersection # (item INT aoi) - overlap
        contribution_list.append(contribution)
        poly_overlap = poly_overlap.union(intersection) if poly_overlap else intersection # overlap UNI (item INT aoi)
    cloud_cover_list = [item['properties']['cloud_cover'] for item in items]
    contribution_list = [contribution.area / poly_overlap.area for contribution in contribution_list]
    cloud_cover = np.dot(cloud_cover_list, contribution_list) # cloud cover = sum(cloud cover * contribution)
    return round(cloud_cover,2)

def score_items(items, coordinates_aoi):
    # Calculate the score for the items based on cloud cover and overlap area
    cloud_cover = get_cloud_cover(items, coordinates_aoi)
    overlap_area = get_overlap(items, coordinates_aoi)
    alpha = 0.7
    score = alpha * overlap_area + (1-alpha) * (1-cloud_cover)
    score = round(score, 2)
    return score

def get_best_day(items_by_day, coordinates_aoi):
    # Find the best day based on the score of the items
    best_day = None
    best_score = 0
    for date, items in items_by_day.items():
        ordered_items = order_items_by_cloud_cover(items)
        score = score_items(ordered_items, coordinates_aoi)
        if score > best_score:
            best_score = score
            best_day = date
    return best_day, best_score

def make_orders(file_aoi, file_orders, start_date='2018-01-01', end_date='2023-12-31'):

    # check if the start date is a Monday
    if datetime.strptime(start_date, "%Y-%m-%d").weekday() != 0:
        raise ValueError("Start date must be a Monday")

    # Get the area of interest (AOI) coordinates
    aoi_name = os.path.splitext(os.path.basename(file_aoi))[0]
    coordinates_aoi = get_aoi_coordinates(file_aoi)

    # Set the start and end dates for the first week
    d1 = datetime.strptime(start_date, "%Y-%m-%d") # Monday at 00:00
    d6 = d1 + timedelta(days=5) # Saturday at 00:00 ~ Friday
    d8 = d1 + timedelta(days=7) # Next Monday at 00:00 ~ Sunday


    while d1 < d8:

        ### d1 -> d6
        name = f"{aoi_name}_{d1.strftime('%Y%m%d')}_{d6.strftime('%Y%m%d')}"
        search_result = search_images(d1, d6, coordinates_aoi)
        # Find the best day
        all_items = search_result['features']
        items_by_day = split_items_by_day(all_items)
        best_day, best_score = get_best_day(items_by_day, coordinates_aoi)
        if best_day is None:
            aoi = name.split('_')[1]
            date_from = name.split('_')[2]
            date_to = name.split('_')[3]
            with open(file_orders, 'a') as f:
                if os.stat(file_orders).st_size == 0:
                    f.write("order_id,aoi,date,date_from,date_to\n")
                f.write(f"{None},{aoi},{None},{date_from},{date_to}\n")
        else:
            best_items = items_by_day[best_day]
            # Order items by cloud cover
            ordered_items = order_items_by_cloud_cover(best_items)
            # Order the items in the right order
            ordered_ids = get_ids(ordered_items)
            order_response = place_order(file_orders, ordered_ids, coordinates_aoi, name=name)

        ### d6 -> d8
        name = f"{aoi_name}_{d6.strftime('%Y%m%d')}_{d8.strftime('%Y%m%d')}"
        search_result = search_images(d6, d8, coordinates_aoi)
        # Find the best day
        all_items = search_result['features']
        items_by_day = split_items_by_day(all_items)
        best_day, best_score = get_best_day(items_by_day, coordinates_aoi)
        if best_day is None:
            aoi = name.split('_')[1]
            date_from = name.split('_')[2]
            date_to = name.split('_')[3]
            with open(file_orders, 'a') as f:
                if os.stat(file_orders).st_size == 0:
                    f.write("order_id,aoi,date,date_from,date_to\n")
                f.write(f"{None},{aoi},{None},{date_from},{date_to}\n")
        else:
            best_items = items_by_day[best_day]
            # Order items by cloud cover
            ordered_items = order_items_by_cloud_cover(best_items)
            # Order the items in the right order
            ordered_ids = get_ids(ordered_items)
            order_response = place_order(file_orders, ordered_ids, coordinates_aoi, name=name)
            
        print("")

        ### Update the dates
        d1 = d8 # New Monday at 00:00
        d6 = min(d1 + timedelta(days=5), datetime.strptime(end_date, "%Y-%m-%d")) # New Saturday at 00:00 ~ Friday ; Limit to the end date
        d8 = min(d1 + timedelta(days=7), datetime.strptime(end_date, "%Y-%m-%d")) # New next Monday at 00:00 ~ Sunday ; Limit to the end date




if __name__ == "__main__":

    # Make orders to later download the Planet images

    list_id = list(range(1,22))

    for aoi_id in list_id:
        print(f"Processing AOI {aoi_id}...")
        file_aoi = f"geodata/aoi/aoi_{aoi_id}.gpkg"
        file_orders = f"download-planet-data/orders/orders_{aoi_id}.csv"
        make_orders(file_aoi, file_orders, start_date='2018-01-01', end_date='2023-12-31')

