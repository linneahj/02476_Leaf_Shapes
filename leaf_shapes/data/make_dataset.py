# Importing packages
import glob
import os
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
import torch
#matplotlib.use('Agg')


def pad2square(img):  #Inherited from ealier implementation
    img_shape = img.shape
    r = img_shape[0]
    c = img_shape[1]
    new_side = np.max([r,c])
    padding = ((new_side - c)//2, (new_side - r)//2, int(np.ceil((new_side - c)/2)), int(np.ceil((new_side - r)/2)))
    img_as_img = Image.fromarray(img)
    new_img = ImageOps.expand(img_as_img, padding)
    return np.array(new_img)


class Data_processor():
    def __init__(self, base_path, image_paths, image_shape=(128, 128)):

        self.base_path = base_path
        self.base_csv_path = base_path + "raw/train.csv" # TODO: This file should be renamed to data.csv
        self.image_paths = image_paths
        self.image_shape = image_shape

        self.training_csv_path = base_path + "processed/train.csv"
        self.test_csv_path = base_path + "processed/test.csv"
        
        self.encoding = None # Encodes each label to a number, set in load()
        self.train_df = None # Dataframes holding raw data - set in make _split_into_train_test
        self.test_df = None # Dataframes holding raw data - set in make _split_into_train_test

        self.train_set_dict = None
        self.test_set_dict = None

        self.training_set = None # for storing processed datasets
        self.test_set = None # for storing processed datasets


    def process(self):
        self._split_into_train_test()
        self._load_raw_data()
        self._make_and_save_tensor_datasets()


    def _split_into_train_test(self):
        '''
        Reads all id's and labels from original csv file and splits it into test data and training data. 
        For simplicity, the last 10 % of the entries in the original csv file with randomized data are chosen for test data.
        '''
        df = pd.read_csv(self.base_csv_path) # names=['id','species']
        
        # Choosing last 99 images to be the test-set
        # Original .csv files only had labels for training - none for testing, thus we cannot use the original test.csv
        # We will use only the images for classifying the plant species
        header = df.columns[0:2]
        self.train_df = pd.DataFrame(df[header].values[0:891],columns=header)
        self.test_df = pd.DataFrame(df[header].values[891:],columns=header)
        self.train_df.to_csv(self.training_csv_path)
        self.test_df.to_csv(self.test_csv_path)

 
    def _load_raw_data(self):  # From original script, but refactored
        # load train.csv
        path_dict = self._path_to_dict() # numerate image paths and make it a dict
        # merge image paths with data frame
        train_image_df = self._merge_image_df(self.train_df, path_dict)
        test_image_df = self._merge_image_df(self.test_df, path_dict)

        self.encoding  = LabelEncoder().fit(train_image_df['species'])
        pd.DataFrame(self.encoding.classes_).to_csv('./data/processed/Class_ids.csv')

        # labels for train
        t_train = self.encoding.transform(train_image_df['species'])
        t_test = self.encoding.transform(test_image_df['species'])

        # TODO: load images and format data seems unescescarily complicated, try to simplify
        # getting data
        print("Loading training data")
        train_data = self._load_images_into_dict(train_image_df, self.image_shape, t_train)
        print("Loading test data")
        test_data = self._load_images_into_dict(test_image_df, self.image_shape, t_test)

        # need to reformat the train for validation split reasons in the batch_generator
        self.train_set_dict = self._format_dataset(train_data)
        self.test_set_dict = self._format_dataset(test_data)
        


    def _path_to_dict(self): # From original script
        path_dict = dict()
        for image_path in self.image_paths:
            num_path = int(os.path.basename(image_path[:-4]))
            path_dict[num_path] = image_path
        return path_dict

    def _merge_image_df(self, df, path_dict):  # From original script
        split_path_dict = dict()
        for index, row in df.iterrows():
            #print(f'Row: {row}, index: {index}')
            #print(f"Test: {row['id']}")
            split_path_dict[row['id']] = path_dict[row['id']]
        image_frame = pd.DataFrame(list(split_path_dict.values()), columns=['image'])
        df_image =  pd.concat([image_frame, df], axis=1)
        return df_image

    def _load_images_into_dict(self, df, image_shape, targets):  # From original script
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

    def _format_dataset(self, df):  # From original script
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = list(df.values())[0]
        img_tot_shp = tuple([len(df)] + list(value['image'].shape))
        data['images'] = np.zeros(img_tot_shp, dtype='float32')

    
        data['targets'] = np.zeros((len(df),), dtype='int32')
        
        for i, pair in enumerate(df.items()):
            _, value = pair
            data['images'][i] = value['image']
            data['targets'][i] = value['t']
            
        return data
    
    def _make_tensor_dataset(self, dataset_dict):
        ''' Takes the data set produced from _format_dataset() and makes it into a TensorDataset'''

         #Store images and targets separately
        np_images_list = dataset_dict['images']
        np_targets_list = dataset_dict['targets']

        # concatenate images to be stacked
        images_np_temp = np.concatenate(np_images_list,axis=2)
        # Move axis such that the index comes first
        images_np=np.moveaxis(images_np_temp,-1,0)

        # Convert to tensors
        images = torch.from_numpy(images_np)
        labels = torch.from_numpy(np_targets_list)

        # Unsqueeze for proper dimensions for models
        images = images.unsqueeze(1)

        # Store training data in a proper dataset
        return torch.utils.data.TensorDataset(images,labels)
        

    def _make_and_save_tensor_datasets(self):
        self.training_set = self._make_tensor_dataset(self.train_set_dict)
        torch.save(self.training_set, self.base_path + '/processed/train_dataset.pt')

        self.test_set = self._make_tensor_dataset(self.test_set_dict)
        torch.save(self.training_set, self.base_path + '/processed/test_dataset.pt')


if __name__ == '__main__':
    # Get the data and process it
    IMAGE_SHAPE = (50,50,1)  # Add hydra hyperparameter control here
    NUM_CLASSES = 99   # Also hydra-something here

    BASE_PATH = "./data/"
    IMAGE_PATHS = glob.glob("./data/raw/images/*.jpg")


    # TO DO:
    # Optimize this to be built into the make_data thingy.
    data = Data_processor(BASE_PATH, image_paths=IMAGE_PATHS,image_shape=IMAGE_SHAPE[:2])
    data.process()
    
   
    


    