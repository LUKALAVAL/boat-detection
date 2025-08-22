import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_dataset, create_yaml_file

if __name__ == "__main__":

    # Build the dataset to train the model
    # Note that the the classes are written directly in the
    # create_yaml_file function

    train_val_test_split = (0.8, 0.1, 0.1)

    dir_images = "PLE/A_dataset/tiles"
    dir_labels = "PLE/A_dataset/tiles_labels"
    dir_output = "PLE/A_dataset/dataset"

    create_dataset(dir_images, dir_labels, dir_output, train_val_test_split)
    create_yaml_file(dir_output)
