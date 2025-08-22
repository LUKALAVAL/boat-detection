import os
from shapely.geometry import Polygon
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import hypot

def load_yolo_obb(file_path, has_confidence=False):

    # Load the yolo prediction file and return the labels

    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:9]))
            polygon = Polygon([(coords[0], coords[1]), (coords[2], coords[3]),
                               (coords[4], coords[5]), (coords[6], coords[7])])
            confidence = float(parts[9]) if has_confidence else None
            annotations.append((class_id, polygon, confidence))
    return annotations

def iou(poly1, poly2):

    # intersection over union of two polygons

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area != 0 else 0.0

def compute_confusion_matrix(gt_folder, pred_folder, iou_threshold, confidence_thresholds):

    # Computes and returns the confusion matrix from the ground truth labels and the predicted labels
    
    all_gt_classes = []
    all_pred_classes = []

    for filename in os.listdir(gt_folder):
        if not filename.endswith(".txt"):
            continue

        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        if not os.path.exists(pred_path):
            continue

        gt_boxes = load_yolo_obb(gt_path, has_confidence=False)
        pred_boxes = load_yolo_obb(pred_path, has_confidence=True)
        # apply confidence_thresholds ({cls:conf, cls:conf})
        tmp = []
        for cls, conf in confidence_thresholds.items():
            tmp += [p for p in pred_boxes if p[2] is None or (p[2] >= conf and p[0] == cls)]
        pred_boxes = tmp

        matched_pred = set()
        for gt_class, gt_poly, _ in gt_boxes:
            best_iou = 0
            best_pred_idx = -1
            for idx, (pred_class, pred_poly, _) in enumerate(pred_boxes):
                if idx in matched_pred:
                    continue
                iou_val = iou(gt_poly, pred_poly)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred_idx = idx

            if best_iou >= iou_threshold and best_pred_idx != -1:
                all_gt_classes.append(gt_class)
                all_pred_classes.append(pred_boxes[best_pred_idx][0])
                matched_pred.add(best_pred_idx)
            else:
                # Missed detection
                all_gt_classes.append(gt_class)
                all_pred_classes.append(-1)

        # Remaining predictions
        for idx, (pred_class, _, _) in enumerate(pred_boxes):
            if idx not in matched_pred:
                all_gt_classes.append(-1)
                all_pred_classes.append(pred_class)

    cm = confusion_matrix(all_gt_classes, all_pred_classes, labels=list(confidence_thresholds.keys()) + [-1])

    return cm

def compute_precision_recall_f1(cm):

    # Compute and returns the precision recall and f1 scores from the confusion matrix

    precision = {}
    recall = {}
    f1_score = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = round(tp / (tp + fp) if (tp + fp) > 0 else 1, 3)
        recall[i] = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 3)
        f1_score[i] = round(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0, 3)
    return precision, recall, f1_score

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):

    # plots the confusion matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if cm.dtype == np.float64 else 'd', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

def get_length(poly):

    # returns the longest side of a polygon with size adjustment
    #
    # This function is kind of a duplicate of another function. It must be cleaned later

    p0, p1, p2 = poly.exterior.coords[:3]
    length = max(hypot(p1[0]-p0[0], p1[1]-p0[1]),
                  hypot(p2[0]-p1[0], p2[1]-p1[1]))
    return (length * 512 * 3) - 12

def match(gt_folder, pred_folder, iou_threshold, confidence_threshold, classes):
    
    # This is the same function as compute_confusion_matrix except the lengths of the corresponding
    # matched objects are returned
    #
    # has to be generalized later

    all_gt_lengths = []
    all_pred_lengths = []

    for filename in os.listdir(gt_folder):
        if not filename.endswith(".txt"):
            continue

        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        if not os.path.exists(pred_path):
            continue

        gt_boxes = load_yolo_obb(gt_path, has_confidence=False)
        pred_boxes = load_yolo_obb(pred_path, has_confidence=True)
        pred_boxes = [p for p in pred_boxes if p[2] is None or p[2] >= confidence_threshold]

        matched_pred = set()
        for gt_class, gt_poly, _ in gt_boxes:
            if gt_class not in classes:
                continue
            best_iou = 0
            best_pred_idx = -1
            for idx, (pred_class, pred_poly, _) in enumerate(pred_boxes):
                if idx in matched_pred:
                    continue
                iou_val = iou(gt_poly, pred_poly)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred_idx = idx

            if best_iou >= iou_threshold and best_pred_idx != -1:
                gt_length = get_length(gt_poly)
                pred_length = get_length(pred_boxes[best_pred_idx][1])
                all_gt_lengths.append(gt_length)
                all_pred_lengths.append(pred_length)
                matched_pred.add(best_pred_idx)
            else:
                # Missed detection → map to background
                all_gt_lengths.append(gt_length)
                all_pred_lengths.append(-1)

        for idx, (pred_class, _, _) in enumerate(pred_boxes):
            if pred_class not in classes:
                continue
            if idx not in matched_pred:
                pred_length = get_length(pred_boxes[idx][1])
                all_gt_lengths.append(-1)
                all_pred_lengths.append(pred_length)

    return all_gt_lengths, all_pred_lengths