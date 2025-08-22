import geopandas as gpd
import os

if __name__ == "__main__":

    # Rearrange the data to group the predictions per months in file for better comparision

    dir_input = "PLE/database/data"
    dir_output = "PLE/database/data_rearranged"

    list_files = os.listdir(dir_input)
    list_files = [os.path.join(dir_input, f) for f in list_files if f.endswith(".csv")]
    list_files.sort()

    for file in list_files:
        aoi_id = os.path.basename(file).split("_")[1].split(".")[0]

        df = gpd.read_file(file)

        df["yearmon"] = df["date"].str[:6]
        df_grouped = df.groupby("yearmon")
        
        for yearmon, group in df_grouped:
            output_file = os.path.join(dir_output, f"ple_{aoi_id}_{yearmon}.csv")
            group = group[[
                "class_id",
                "latitude", 
                "longitude",
                "length",
                "breadth",
                "heading",
                "date",
                ]]
            group.to_csv(output_file, index=False)
