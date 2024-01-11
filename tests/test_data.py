import glob
import os

from tests import _PATH_DATA


def test_raw_data():
    dir = os.path.join(_PATH_DATA, "raw")

    # Checking the number of images
    images = os.listdir(os.path.join(dir, "images"))
    assert len(images) == 1584, "Incorrect number of images detected"
    # Checking if the train.csv file exists in the correct location
    assert os.path.exists(os.path.join(dir, "train.csv")), "Could not find the train.csv file"


def test_data_loaded():
    path = os.path.join(_PATH_DATA, "processed")

    # Checking the labels
    assert os.path.exists(
        os.path.join(path, "labelled_training_data.csv")
    ), "Could not find the labelled training data csv-file"
    assert os.path.exists(
        os.path.join(path, "labelled_test_data.csv")
    ), "Could not find the labelled test data csv-file"

    # Checking the images
    subfolders = [f.name for f in os.scandir(os.path.join(path, "TIMM")) if f.is_dir()]
    assert len(subfolders) == 99, "Incorrect number of classes found."

    print(os.listdir(path))
    images = [images for images in glob.iglob(f"{path}/TIMM/*/*") if images.endswith(".png")]
    N_train = 891
    N_test = 99
    assert len(images) == (N_train + N_test), "Could not find the correct number of processed images"
