import os
from turtle import width
import matplotlib.pyplot as plt
import pandas as pd


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def show_annotations(dir_images, dir_labels):
    img_size = 512

    list_images = sorted(os.listdir(dir_images))

    for i, image_file in enumerate(list_images):

        

        label_file = os.path.join(dir_labels, os.path.splitext(image_file)[0] + ".txt")
        if not os.path.exists(label_file):
            continue

        with open(label_file, "r") as f:
            annotations = f.readlines()

        if len(annotations) == 0:
            continue

        img = plt.imread(os.path.join(dir_images, image_file))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image {i + 1} / {len(list_images)}")
        
        # display the yolo obb annotations cls x1 y1 x2 y2 x3 y3 x4 y4
        for annotation in annotations:
            line = annotation.strip().split()
            if len(line) == 10:
                cls, x1, y1, x2, y2, x3, y3, x4, y4, conf = map(float, line)
                if conf < 0.165: #treshold just for some prediction visualization
                    continue
            elif len(line) == 9:
                cls, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line)

            # denormalize coordinates
            x1 *= img_size
            y1 *= img_size
            x2 *= img_size
            y2 *= img_size
            x3 *= img_size
            y3 *= img_size
            x4 *= img_size
            y4 *= img_size
            
            # draw oriented rectangle
            plt.gca().add_patch(plt.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=False, color="red" if cls == 0 else "blue"))

        plt.show()
        # plt.savefig(f"utils/images/image_{i + 1}.png", bbox_inches="tight", dpi=200)
        # plt.close()

def display_stats(dir_images, dir_labels):
    img_size = 512
    img_resolution = 3 # m/px

    stats = {
        "total_tiles": 0,
        "tiles_with_annotations": 0,
        "total_annotations": 0,
        "class_distribution": {},
        "length_distribution": {},
        "breadth_distribution": {},
        "objects_per_tile_distribution": []
    }
    
    for label_file in os.listdir(dir_labels):
        with open(os.path.join(dir_labels, label_file), "r") as f:
            annotations = f.readlines()

            stats["total_tiles"] += 1

            if annotations:
                stats["tiles_with_annotations"] += 1
                stats["objects_per_tile_distribution"].append(len(annotations))

                for annotation in annotations:
                    cls = int(annotation.split()[0])
                    stats["class_distribution"][cls] = stats["class_distribution"].get(cls, 0) + 1

                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, annotation.split()[1:9])
                    distances = [
                        distance((x1, y1), (x2, y2)),
                        distance((x2, y2), (x3, y3)),
                        distance((x3, y3), (x4, y4)),
                        distance((x4, y4), (x1, y1))
                    ]

                    length = max(distances)
                    length = round(length * img_size * img_resolution, 1) - 12 # 6 meters padding on both sides
                    stats["length_distribution"][cls] = stats["length_distribution"].get(cls, [])
                    stats["length_distribution"][cls].append(length)

                    breadth = min(distances)
                    breadth = round(breadth * img_size * img_resolution, 1) - 12 # 6 meters padding on both sides
                    stats["breadth_distribution"][cls] = stats["breadth_distribution"].get(cls, [])
                    stats["breadth_distribution"][cls].append(breadth)

    stats["total_annotations"] = sum(stats["objects_per_tile_distribution"])

    print("Dataset statistics:")
    for key, value in stats.items():
        if key in ["length_distribution", "breadth_distribution", "objects_per_tile_distribution"]:
            continue  # Skip
        print(f"  {key}: {value}")


    # plot length by aspect-ratio distribution only for class 0
    plt.figure(figsize=(10, 6))
    cls = 0
    aspect_ratios = [breadth / length if length > 0 else 0 for length, breadth in zip(stats["length_distribution"][cls], stats["breadth_distribution"].get(cls, []))]
    plt.scatter(aspect_ratios, stats["length_distribution"][cls], label=f"Class {cls}", s=20, alpha=0.6)
    plt.xlabel("Aspect Ratio (Breadth/Length)")
    plt.ylabel("Length (m)")
    plt.title("Length by Aspect Ratio Distribution")
    from shapely.geometry import Point, Polygon 
    polygon = Polygon([(0.05,450), (0.05,100), (0.1,0), (1,0), (1,30), (0.7,50), (0.4,200), (0.3,450), (0.05,450)])
    # display the polygon
    plt.plot(*polygon.exterior.xy, color='green', label='Polygon Boundary')

    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    # plot length by breadth distribution
    plt.figure(figsize=(10, 6))
    for cls in stats["length_distribution"]:
        plt.scatter(stats["length_distribution"][cls], stats["breadth_distribution"].get(cls, []), label=f"Class {cls}", s=40, alpha=0.9, marker='+')
    plt.xlabel("Length (m)")
    plt.ylabel("Breadth (m)")
    plt.title("Length vs Breadth Distribution")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    # plot objects per tile distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats["objects_per_tile_distribution"], bins=range(1, max(stats["objects_per_tile_distribution"]) + 2), align="left")
    plt.xlabel("Number of Objects")
    plt.ylabel("Frequency")
    plt.title("Objects per Tile Distribution")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.show()

    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(stats["class_distribution"].keys(), stats["class_distribution"].values())
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.show()

if __name__ == "__main__":

    dir_images = "PLE/A_dataset/dataset/images/train"
    dir_labels = "PLE/A_dataset/dataset/labels/train"
    # dir_labels = "PLA/B_training/predict/labels"

    display_stats(dir_images, dir_labels)
    # show_annotations(dir_images, dir_labels)

