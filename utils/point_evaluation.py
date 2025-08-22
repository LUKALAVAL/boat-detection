import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import geopandas as gpd





def haversine_distance(lat1, lon1, lat2, lon2):

    # Calculate the Haversine distance between coordinates
    # Returns distance in meters

    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c * 1000  # return distance in meters

def match_dt_gt(gdf_detections, gdf_groundtruth, max_distances, with_conf=False):
    
    # Matches detections to groundtruth using Hungarian algorithm based on Haversine distance
    # Returns a gdf of matched indices

    gdf_detections = gdf_detections.reset_index(drop=True)
    gdf_groundtruth = gdf_groundtruth.reset_index(drop=True)

    n_dts = len(gdf_detections)
    n_gts = len(gdf_groundtruth)

    if n_dts == 0 or n_gts == 0:
        return pd.DataFrame(columns=["matched", "gt_class", "gt_lat", "gt_lon", "dt_class", "dt_lat", "dt_lon"] + ["dt_conf"] if with_conf else [])

    # Build cost matrix using Haversine distance
    cost = np.zeros((n_dts, n_gts))
    for i in range(n_dts):
        for j in range(n_gts):
            cost[i, j] = haversine_distance(
                gdf_detections.loc[i, "latitude"], gdf_detections.loc[i, "longitude"],
                gdf_groundtruth.loc[j, "latitude"], gdf_groundtruth.loc[j, "longitude"]
            )
            if cost[i, j] > max_distances[gdf_groundtruth.loc[j, "class_id"]]:
                cost[i, j] = 9e9

    # Optimal matching
    row_ind, col_ind = linear_sum_assignment(cost)

    # Add matches to GeoDataFrame
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 9e9:
            match = {
                "matched": True,
                "gt_class": gdf_groundtruth.loc[c, "class_id"],
                "gt_lat": gdf_groundtruth.loc[c, "latitude"],
                "gt_lon": gdf_groundtruth.loc[c, "longitude"],
                "dt_class": gdf_detections.loc[r, "class_id"],
                "dt_lat": gdf_detections.loc[r, "latitude"],
                "dt_lon": gdf_detections.loc[r, "longitude"],
            }
            if with_conf:
                match["dt_conf"] = gdf_detections.loc[r, "confidence"]
            matches.append(match)
        else:
            row_ind = np.delete(row_ind, np.where(row_ind == r))
            col_ind = np.delete(col_ind, np.where(col_ind == c))

    # Add unmatched groundtruths (do not forget max_distance)
    unmatched_groundtruths = set(range(n_gts)) - set(col_ind)
    for j in unmatched_groundtruths:
        match = {
            "matched": False,
            "gt_class": gdf_groundtruth.loc[j, "class_id"],
            "gt_lat": gdf_groundtruth.loc[j, "latitude"],
            "gt_lon": gdf_groundtruth.loc[j, "longitude"],
            "dt_class": None,
            "dt_lat": None,
            "dt_lon": None,
        }
        if with_conf:
            match["dt_conf"] = None
        matches.append(match)

    # Add unmatched detections do not forget max_distance
    unmatched_detections = set(range(n_dts)) - set(row_ind)
    for i in unmatched_detections:
        match = {
            "matched": False,
            "gt_class": None,
            "gt_lat": None,
            "gt_lon": None,
            "dt_class": gdf_detections.loc[i, "class_id"],
            "dt_lat": gdf_detections.loc[i, "latitude"],
            "dt_lon": gdf_detections.loc[i, "longitude"],
        }
        if with_conf:
            match["dt_conf"] = gdf_detections.loc[i, "confidence"]
        matches.append(match)

    return pd.DataFrame(matches)


def evaluate(gdf_match):
    
    # Extracts metrics from the matched GeoDataFrame
    # Returns a dictionary with True Positives (TP), False Positives (FP), False Negatives (FN)

    tp = len(gdf_match[gdf_match["matched"] == True])
    fp = len(gdf_match[gdf_match["gt_class"].isnull()])
    fn = len(gdf_match[gdf_match["dt_class"].isnull()])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def get_confmat(gdf_match, nb_classes=2):
    # create empty matrix of nb_classes + 1 columns and rows
    confusion_matrix = np.zeros((nb_classes + 1, nb_classes + 1))

    for i, row in gdf_match.iterrows():
        if row["matched"]:
            gt_class = int(row["gt_class"])
            dt_class = int(row["dt_class"])
            confusion_matrix[gt_class, dt_class] += 1
        elif pd.isnull(row["dt_class"]):
            confusion_matrix[gt_class, -1] += 1
        elif pd.isnull(row["gt_class"]):
            confusion_matrix[-1, dt_class] += 1

    return confusion_matrix

def multi_evaluate_conf(gdf_detections, gdf_groundtruth, min_confidence_list, max_distances):

    # Applies the evaluate function for multiple minimum confidence thresholds
    # Returns a dictionary with lists of results for each threshold

    aggregated_result = {
        "tp": [],
        "fp": [],
        "fn": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "accuracy": []
    }

    for min_confidence in min_confidence_list:
        print(f"Evaluating with min_confidence={min_confidence:.2f}")
        gdf_dt = gdf_detections[gdf_detections['confidence'] >= min_confidence].copy()
        gdf_match = match_dt_gt(gdf_dt, gdf_groundtruth, max_distances=max_distances)
        result = evaluate(gdf_match)
        aggregated_result["tp"].append(result["tp"])
        aggregated_result["fp"].append(result["fp"])
        aggregated_result["fn"].append(result["fn"])
        aggregated_result["precision"].append(result["precision"])
        aggregated_result["recall"].append(result["recall"])
        aggregated_result["f1"].append(result["f1"])
        aggregated_result["accuracy"].append(result["accuracy"])

    return aggregated_result


def multi_evaluate_files(list_files_dt, list_files_gt, max_distance):

    # matches the ground truth and detections on multiple files
    # returns the aggragated results

    aggregated_result = {
        "tp": [],
        "fp": [],
        "fn": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "accuracy": []
    }

    for file_gt, file_dt in zip(list_files_gt, list_files_dt):

        # Load groundtruth and preprocess the data
        gdf_gt = gpd.read_file(file_gt)
        gdf_gt['class_id'] = gdf_gt['type'].apply(lambda x: 0 if x == 'boat' else 1)
        gdf_gt = gdf_gt.drop(columns=['type'])
        gdf_gt = gdf_gt[gdf_gt['class_id'] == 0]
        gdf_gt = gdf_gt.apply(pd.to_numeric, errors='coerce')
        gdf_gt = gdf_gt.dropna(subset=['latitude', 'longitude'])

        # Load detections and preprocess the data
        gdf_dt = gpd.read_file(file_dt)
        gdf_dt['class_id'] = gdf_dt['type'].apply(lambda x: 0 if x == 'boat' else 1)
        gdf_dt = gdf_dt.drop(columns=['type'])
        gdf_dt = gdf_dt[gdf_dt['class_id'] == 0]
        gdf_dt = gdf_dt.apply(pd.to_numeric, errors='coerce')
        gdf_dt = gdf_dt.dropna(subset=['latitude', 'longitude'])

        # Match detections to groundtruth and save the results
        df_match = match_dt_gt(gdf_dt, gdf_gt, max_distance)
        result = evaluate(df_match)
        aggregated_result["tp"].append(result["tp"])
        aggregated_result["fp"].append(result["fp"])
        aggregated_result["fn"].append(result["fn"])
        aggregated_result["precision"].append(result["precision"])
        aggregated_result["recall"].append(result["recall"])
        aggregated_result["f1"].append(result["f1"])
        aggregated_result["accuracy"].append(result["accuracy"])

    return aggregated_result



def plot_confmat(results, title='Confusion Matrix'):

    # Plots a confusion matrix based on the results of the evaluation

    conf_matrix = np.array([[results["tp"], results["fn"]], [results["fp"], 0]])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Boat', 'Background'])

    fig, ax = plt.subplots(figsize=(15, 6), nrows=1, ncols=2)
    fig.suptitle(title)
    disp.plot(ax=ax[0], cmap="Blues", values_format='d', xticks_rotation=45)

    # normalize the confusion matrix
    total = results["tp"] + results["fn"]
    if total > 0:
        conf_matrix_normalized = np.array([[results["tp"]/total, results["fn"]/total], [1.0, 0]])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized, display_labels=['Boat', 'Background'])
        disp.plot(ax=ax[1], cmap="Blues", values_format='.2f', xticks_rotation=45)


def plot_prf1_conf(results, min_confidence_list, title='Precision, Recall and F1 Score vs Confidence'):

    # Plots Precision, Recall, and F1 Score curves against minimum confidence thresholds

    plt.figure(figsize=(10, 6))
    plt.plot(min_confidence_list, results["precision"], label='Precision', marker='.')
    plt.plot(min_confidence_list, results["recall"], label='Recall', marker='.')
    plt.plot(min_confidence_list, results["f1"], label='F1 Score', marker='.')
    plt.title(title)
    plt.xlabel('Confidence')
    plt.ylabel('Score')
    plt.xticks(min_confidence_list)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()