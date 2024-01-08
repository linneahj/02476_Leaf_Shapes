# Importing packages
import glob
import os
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import torch
matplotlib.use('Agg')

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[int(row), int(col)] = 1
    return out


def pad2square(img):
    img_shape = img.shape
    r = img_shape[0]
    c = img_shape[1]
    new_side = np.max([r,c])
    padding = ((new_side - c)//2, (new_side - r)//2, int(np.ceil((new_side - c)/2)), int(np.ceil((new_side - r)/2)))
    img_as_img = Image.fromarray(img)
    new_img = ImageOps.expand(img_as_img, padding)
    return np.array(new_img)


class Data_processor():
    # data_train, data_test and le are public
    def __init__(self, csv_path, image_paths, image_shape=(128, 128)):
        df = pd.read_csv(csv_path) # names=['id','species']
        
        # Choosing last 99 images to be the test-set
        # Original .csv files only had labels for training - none for testing, thus we cannot use the original test.csv
        # We will use only the images for classifying the plant species
        header = df.columns[0:2]
        train = pd.DataFrame(df[header].values[0:891],columns=header)
        test = pd.DataFrame(df[header].values[891:],columns=header)
        train.to_csv('./data/processed/train.csv')
        test.to_csv('./data/processed/test.csv')

        image_paths = image_paths
        image_shape = image_shape
        self._load(train, test, image_paths, image_shape)
        # TO DO:
        # Move loader out of the initialization

    def _load(self, train_df, test_df, image_paths, image_shape):
        # load train.csv
        path_dict = self._path_to_dict(image_paths) # numerate image paths and make it a dict
        # merge image paths with data frame
        train_image_df = self._merge_image_df(train_df, path_dict)
        test_image_df = self._merge_image_df(test_df, path_dict)
        # label encoder-decoder (self. because we need it later)
        self.le = LabelEncoder().fit(train_image_df['species'])
        pd.DataFrame(self.le.classes_).to_csv('./data/processed/Class_ids.csv')
        # labels for train
        t_train = self.le.transform(train_image_df['species'])
        # getting data
        print("Loading training data")
        train_data = self._make_dataset(train_image_df, image_shape, t_train)
        print("Loading test data")
        test_data = self._make_dataset(test_image_df, image_shape, t_train)
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data, for_train=True)
        self.test = self._format_dataset(test_data, for_train=False)


    def _path_to_dict(self, image_paths):
        path_dict = dict()
        for image_path in image_paths:
            num_path = int(os.path.basename(image_path[:-4]))
            path_dict[num_path] = image_path
        return path_dict

    def _merge_image_df(self, df, path_dict):
        split_path_dict = dict()
        for index, row in df.iterrows():
            print(f'Row: {row}, index: {index}')
            print(f"Test: {row['id']}")
            split_path_dict[row['id']] = path_dict[row['id']]
        image_frame = pd.DataFrame(list(split_path_dict.values()), columns=['image'])
        df_image =  pd.concat([image_frame, df], axis=1)
        return df_image

    def _make_dataset(self, df, image_shape, targets):
        # make dataset
        data = dict()
        # merge image with 3x64 features
        for i, dat in enumerate(df.iterrows()):
            index, row = dat
            sample = dict()
            sample['t'] = np.asarray(targets[i], dtype='int32')
            image = imread(row['image'], as_gray=True)
            image = pad2square(image)
            image = resize(image, output_shape=image_shape, mode='reflect', anti_aliasing=True)
            image = np.expand_dims(image, axis=2)
            sample['image'] = image
            data[row['id']] = sample
            if i % 100 == 0:
                print("\t%d of %d" % (i, len(df)))
        print(data.values)

        return data

    def _format_dataset(self, df, for_train):
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = list(df.values())[0]
        img_tot_shp = tuple([len(df)] + list(value['image'].shape))
        data['images'] = np.zeros(img_tot_shp, dtype='float32')

        if for_train:
            data['ts'] = np.zeros((len(df),), dtype='int32')
        else:
            data['ids'] = np.zeros((len(df),), dtype='int32')
        for i, pair in enumerate(df.items()):
            key, value = pair
            data['images'][i] = value['image']
            if for_train:
                data['ts'][i] = value['t']
            else:
                data['ids'][i] = key
        return data


if __name__ == '__main__':
    # Get the data and process it
    IMAGE_SHAPE = (50,50,1)  # Add hydra hyperparameter control here
    NUM_CLASSES = 99   # Also hydra-something here

    CSV_PATH = "./data/raw/train.csv"
    # TEST_PATH = "./data/processed/test.csv"
    IMAGE_PATHS = glob.glob("./data/raw/images/*.jpg")


    # TO DO:
    # Optimize this to be built into the make_data thingy.
    data = Data_processor(csv_path=CSV_PATH, image_paths=IMAGE_PATHS,image_shape=IMAGE_SHAPE[:2])
    
    # Starting with the training data: store images and targets separately
    all_image_nps = data.train['images']
    all_target_nps = data.train['ts']

    # concatenate images to be stacked
    images_yir = np.concatenate(all_image_nps,axis=2)
    # Move axis such that the index comes first
    images_yo=np.moveaxis(images_yir,-1,0)

    # Convert to tensors
    train_images = torch.from_numpy(images_yo)
    train_labels = torch.from_numpy(all_target_nps)

    # Unsqueeze for proper dimensions for models
    train_images = train_images.unsqueeze(1)

    # Store training data in a proper dataset
    train_dataset = torch.utils.data.TensorDataset(train_images,train_labels)


    # Do the same for the test data
    all_image_nps_t = data.test['images']
    all_target_nps_t = data.test['ts']
    
    # concatenate images to be stacked
    images_yir_t = np.concatenate(all_image_nps_t,axis=2)
    # Move axis such that the index comes first
    images_yo_t=np.moveaxis(images_yir_t,-1,0)

    # Convert to tensors
    test_images = torch.from_numpy(images_yo_t)
    test_labels = torch.from_numpy(all_target_nps_t)

    # Unsqueeze for proper dimensions for models
    test_images = test_images.unsqueeze(1)

    # Store training data in a proper dataset
    test_dataset = torch.utils.data.TensorDataset(test_images,test_labels)

    # Store datasets for use in models
    torch.save(train_dataset,'./data/processed/train_dataset.pt')
    torch.save(test_dataset,'./data/processed/test_dataset.pt')

    


    