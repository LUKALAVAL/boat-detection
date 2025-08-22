from planet import Planet
import os
import csv
import time
import httpx



PLANET_API_KEY = "YOUR_PLANET_API_KEY"
os.environ['PL_API_KEY'] = PLANET_API_KEY
pl = Planet()




def list_orders():
    orders = pl.orders.list_orders()
    return orders # dictionnary of orders

def get_object_url(order_id, name):
    try:
        order = pl.orders.get_order(order_id)
        results = order.get('_links', {}).get('results', [])
        for result in results:
            if name in result['name'] and result['delivery'] == 'success':
                return result['location']
    except Exception as e:
        print(f"Error getting image URL for order {order_id}: {e}")
    return None


def download_asset(url, filename, dir_destination, retries=5, delay=5):
    for attempt in range(1, retries + 1):
        try:
            path = pl.orders.download_asset(url, directory=dir_destination, filename=filename)
            return path
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status in (500, 504):  # Server errors
                print(f"Server error ({status}) on attempt {attempt} for {url}. Retrying in {delay} seconds...")
                time.sleep(delay * attempt)
                continue
            else:
                print(f"HTTP error {status} on attempt {attempt} for {url}: {e}")
                break
        except httpx.TimeoutException as e:
            print(f"TimeoutException on attempt {attempt} for {url}: {e}")
            time.sleep(delay * attempt)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt} downloading {url}: {e}")
            time.sleep(delay * attempt)

    print(f"Failed to download after {retries} attempts: {url}")
    return None


def download_orders(file_orders, dir_destination, name):

    os.makedirs(dir_destination, exist_ok=True)

    with open(file_orders, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            order_id = row[0]
            aoi = row[1]
            date = row[2]
            date_from = row[3]
            date_to = row[4]

            if order_id == "None":
                continue

            filename = f"{aoi}_{date}_{order_id}.tif"
            url = get_object_url(order_id, name)
            if url is None:
                print(f"Order {order_id} does not have a valid URL.")
                continue

            print(f"Downloading {url} as {filename}...")
            path = download_asset(url, filename, dir_destination)
            if path:
                print(f"Downloaded to {path}")
            else:
                print(f"Failed to download {filename}")




if __name__ == "__main__":

    # Download the planet images as well as the associated masks

    list_id = list(range(1, 22))

    for aoi_id in list_id:
        print(f"Processing AOI {aoi_id}...")
        file_orders = f"orders_{aoi_id}.csv"
        download_orders(file_orders, "PLE/C_inference/images", "composite.tif")
        download_orders(file_orders, "PLE/C_inference/masks", "composite_udm2.tif")
    