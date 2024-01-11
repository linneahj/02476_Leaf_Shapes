import os

import hydra
import matplotlib.pyplot as plt  # skimage.io wants greyscale images in range [0,1], but plt seems to work fine
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from skimage.io import imread
from skimage.transform import resize

# Importing hyperparameters (config)
config = OmegaConf.load("config.yaml")


def pad2square(img):  # Inherited from ealier implementation
    img_shape = img.shape
    r = img_shape[0]
    c = img_shape[1]
    new_side = np.max([r, c])
    padding = (
        (new_side - c) // 2,
        (new_side - r) // 2,
        int(np.ceil((new_side - c) / 2)),
        int(np.ceil((new_side - r) / 2)),
    )
    img_as_img = Image.fromarray(img)
    new_img = ImageOps.expand(img_as_img, padding)
    return np.array(new_img)


class Data_processor:
    def __init__(
        self,
        path_to_base_csv="./data/raw/train.csv",
        image_paths="./data/raw/images/",
        output_path="./data/processed",
        image_shape=config.hyperparameters.image_shape,
    ):
        self.base_csv_path = path_to_base_csv  # TODO: This file should be renamed to data.csv
        self.output_path = output_path
        self.path_to_images = image_paths
        self.image_shape = image_shape

        self.training_csv_path = output_path + "/labelled_training_data.csv"
        self.test_csv_path = output_path + "/labelled_test_data.csv"

        self.train_df = None  # Dataframes holding raw data - set in make _split_into_train_test
        self.test_df = None  # Dataframes holding raw data - set in make _split_into_train_test

    def _split_into_train_test(self):
        """
        Reads all id's and labels from original csv file and splits it into test data and training data.
        For simplicity, the last 10 % of the entries in the original csv file with randomized data are chosen for test data.
        """
        # Load csv containing all labelled data
        df = pd.read_csv(self.base_csv_path)  # names=['id','species']

        # Splitting
        header = df.columns[
            0:2
        ]  # We will use only the images for classifying the plant species, thus all other features can be ignored
        train_df = pd.DataFrame(df[header].values[0:891], columns=header)
        test_df = pd.DataFrame(df[header].values[891:], columns=header)  # Choosing last 99 images to be the test-set

        # Save image paths
        train_df["path_to_image"] = ""
        test_df["path_to_image"] = ""

        for index, (
            id,
            _,
            _,
        ) in train_df.iterrows():  # Images must be stored in the given path using the format {id}.jpg
            train_df["path_to_image"][index] = self.path_to_images + f"{id}" + ".jpg"

        for index, (
            id,
            _,
            _,
        ) in test_df.iterrows():  # Images must be stored in the given path using the format {id}.jpg
            test_df["path_to_image"][index] = self.path_to_images + f"{id}" + ".jpg"

        self.train_df = train_df
        self.test_df = test_df

        # Save to have as an overview
        train_df.to_csv(self.training_csv_path)
        test_df.to_csv(self.test_csv_path)

        return train_df, test_df

    def _create_data(self, df, output_path, image_shape):
        """loads all images in df, processes them and saves in given path"""
        for _, (id, species, path) in df.iterrows():
            image = imread(path, as_gray=True)  # load image

            # Process
            image = pad2square(image)
            image = resize(image, output_shape=image_shape, mode="reflect", anti_aliasing=True)
            # image = np.expand_dims(image, axis=0)

            base_path = output_path + "/TIMM/"
            path = output_path + f"/TIMM/{species}"
            if not os.path.exists(base_path):
                os.mkdir(base_path)

            if not os.path.exists(path):
                os.mkdir(path)

            # plt.imshow(image, cmap="gray")
            plt.imsave(path + f"/{id}.png", image, cmap="gray")

    def process(self):
        self._split_into_train_test()
        self._create_data(self.train_df, self.output_path, self.image_shape)
        # If test also
        # self.__create_data(self.train_df, self.output_path + '/test_images', self.image_shape)


if __name__ == "__main__":
    #  Original .csv files only had labels for training - none for testing, thus we cannot use the original test.csv and all our data is in train.csv
    data_procesor = Data_processor("./data/raw/train.csv", "./data/raw/images/", "./data/processed")
    data_procesor.process()
