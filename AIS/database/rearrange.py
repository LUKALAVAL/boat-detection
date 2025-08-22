import geopandas as gpd
import pandas as pd
import os

if __name__ == "__main__":

    dir_input = "AIS/data"
    dir_output = "AIS/data_rearranged"

    # keep only the data within the time range
    time_range = ["09:00:00", "11:00:00"]

    list_files = os.listdir(dir_input)
    list_files = [os.path.join(dir_input, f) for f in list_files if f.endswith(".csv")]
    list_files.sort()

    for file in list_files:
        df = gpd.read_file(file)
        
        df = df.rename(columns={"lat": "latitude", "lon": "longitude"})

        df["class_id"] = 0 # set all boats to class 0
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
        df.loc[df["speed"] > 1, "class_id"] = 1 # set all moving boats to class 1

        df["heading"] = pd.to_numeric(df["heading"], errors="coerce")
        df["heading"] = df["heading"].mod(180) # normalize heading to [0, 180)

        df["last_pos_utc"] = df["last_pos_utc"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S", errors='coerce'))
        df["date"] = df["last_pos_utc"].dt.strftime("%Y%m%d")
        df["time"] = df["last_pos_utc"].dt.time
        df = df[df["time"].between(pd.to_datetime(time_range[0]).time(), pd.to_datetime(time_range[1]).time())] # keep only the data within the time range

        df = df[~df.duplicated(subset=["mmsi", "date"], keep="first")] # remove duplicates

        df = df[[
            "class_id",
            "latitude", 
            "longitude",
            "length",
            "breadth",
            "heading",
            "date",
            ]]
        output_file = os.path.join(dir_output, os.path.basename(file))
        df.to_csv(output_file, index=False) # save to csv