import geopandas as gpd
import pandas as pd
import os

if __name__ == "__main__":

    dir_input = "GFW/data"
    dir_output = "GFW/data_rearranged"

    list_files = os.listdir(dir_input)
    list_files = [os.path.join(dir_input, f) for f in list_files if f.endswith(".csv")]
    list_files.sort()

    for file in list_files:
        df = gpd.read_file(file)

        df = df.rename(columns={"lat": "latitude", "lon": "longitude", "length_m_inferred": "length"})

        df["class_id"] = 0 # set all boats to class 0
        df["speed_kn_inferred"] = pd.to_numeric(df["speed_kn_inferred"], errors="coerce")
        df.loc[df["speed_kn_inferred"] > 1, "class_id"] = 1 # set all moving boats to class 1

        df["heading"] = pd.to_numeric(df["heading_deg_inferred"], errors="coerce")
        df["heading"] = df["heading"].mod(180) # normalize heading to [0, 180)

        df["detect_timestamp"] = pd.to_datetime(df['detect_timestamp'], utc=True, format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
        df["date"] = pd.to_datetime(df["detect_timestamp"]).dt.strftime("%Y%m%d")
        df["time"] = pd.to_datetime(df["detect_timestamp"]).dt.time

        df = df[[
            "class_id",
            "latitude", 
            "longitude",
            "length",
            "heading",
            "date",
            "time",
            ]]
        output_file = os.path.join(dir_output, os.path.basename(file))
        df.to_csv(output_file, index=False) # save to csv