# boat-detection

Monitoring small vessel activity is critical for understanding human pressure on marine and coastal ecosystems, yet such vessels often lack AIS (Automatic Identification System) transponders and typically require costly, very high-resolution satellite imagery for reliable detection that may be sparse in time and space. We developed a boat detection model that identifies vessels under 6 meters in length from PlanetScope 3m/px satellite imagery by training on degraded Pléiades high-resolution images. This enables scalable mapping of small-boat activity across broad spatial and temporal extents, providing valuable proxies for human activity in ecological studies, such as species distribution modeling. For more context, read `report.pdf`.

The repository is organized to facilitate both model experimentation and comparative evaluation. Each model subfolder (`DOT`, `PLA`, and `PLE`) contains the respective model weights, and training datasets (when available). `PLE` also contains the preprocessing scripts, and functions for model training and inference. The databases subfolders (`AIS`, `GFW`, and `PLE`) are structured in a similar way containing the evaluation datasets and utilities required for benchmarking. The evaluations scripts for models and databases are in `evaluation_models` and `evaluation_databases` folders. Finally, `boat-detections-med-2018-2023` correspond to the filtered boat detection database derived using the PLE model on PlanetScope images over 6 years close to the french coasts.


```bash
├── AIS
│   └── database
│       ├── data
│       │   └── ais_5_2022**.csv
│       ├── data_rearranged
│       │   └── ais_5_2022**.csv
│       └── rearrange.py
├── boat-detections-med-2018-2023
│   └── ple_*.csv
├── DOT
│   ├── context.md
│   └── weights.pt
├── evaluate_databases
│   ├── evaluate.ipynb
│   ├── match_ais_gfw.csv
│   └── match_ais_ple.csv
├── evaluate_models
│   ├── A_detect_boats.py
│   ├── B_evaluate.ipynb
│   ├── detections
│   │   ├── dot_detections.csv
│   │   ├── dot_detections.txt
│   │   ├── match_gt_dot.csv
│   │   ├── match_gt_pla.csv
│   │   ├── match_gt_ple.csv
│   │   ├── pla_detections.csv
│   │   ├── pla_detections.txt
│   │   ├── ple_detections.csv
│   │   └── ple_detections.txt
│   └── groundtruth.csv
├── geodata
│   ├── aoi
│   │   ├── aoi_*.gpkg
│   │   └── aoi.gpkg
│   └── landmask
│       ├── landmask_*.gpkg
│       ├── landmask_buffer_*.gpkg
│       └── landmask.gpkg
├── GFW
│   └── database
│       ├── data
│       │   └── gfw_5_2022**.csv
│       ├── data_rearranged
│       │   └── gfw_5_2022**.csv
│       └── rearrange.py
├── PLA
│   ├── A_dataset
│   │   └── dataset
│   ├── B_model
│   │   ├── A_train.py
│   │   ├── B_val.py
│   │   ├── C_test.py
│   │   └── runs
│   │       └── obb
│   │           ├── test
│   │           ├── train
│   │           └── val
│   └── weights.pt
├── PLE
│   ├── A_dataset
│   │   ├── A_annotate_images.py
│   │   ├── B_transform_images.py
│   │   ├── C_slice_images_and_labels.py
│   │   ├── dataset
│   │   └── D_build_dataset.py
│   ├── B_model
│   │   ├── A_train.py
│   │   ├── B_predict.py
│   │   ├── performances.ipynb
│   │   ├── predict
│   │   └── runs
│   │       └── obb
│   │           └── train
│   ├── C_inference
│   │   ├── A_make_orders.py
│   │   ├── B_download_data.py
│   │   ├── C_detect_boats.py
│   │   ├── detections
│   │   │   ├── 19_20190718_3e7839dc-2ca6-4cc6-8f38-8457242b7342.csv
│   │   │   ├── 19_20190718_3e7839dc-2ca6-4cc6-8f38-8457242b7342.geojson
│   │   │   ├── 19_20190718_3e7839dc-2ca6-4cc6-8f38-8457242b7342.txt
│   │   │   └── grouped_19.csv
│   │   ├── detections_filtered
│   │   │   └── filtered_19.csv
│   │   ├── D_filter_detections.py
│   │   ├── E_generate_overpass.py
│   │   ├── images
│   │   │   └── 19_20190718_3e7839dc-2ca6-4cc6-8f38-8457242b7342.tif
│   │   ├── masks
│   │   │   └── 19_20190718_3e7839dc-2ca6-4cc6-8f38-8457242b7342.tif
│   │   ├── orders
│   │   └── overpass
│   │       └── 19_overpass.tif
│   ├── database
│   │   ├── data
│   │   │   └── filtered_5.csv
│   │   ├── data_rearranged
│   │   │   └── ple_5_20****.csv
│   │   └── rearrange.py
│   └── weights.pt
├── README.md
├── report.pdf
└── utils
    ├── box_evaluation.py
    ├── check_dataset.py
    ├── point_evaluation.py
    └── utils.py
```