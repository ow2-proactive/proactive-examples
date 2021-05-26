# Copyright Activeeon 2007-2021. All rights reserved.
__file__ = variables.get("PA_TASK_NAME")

import re
import json
import wget
import uuid
import shutil
import zipfile
from os import remove, listdir, makedirs
from os.path import basename, splitext, exists, join
from sklearn.model_selection import train_test_split

# DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/ants_vs_bees.zip'  #CLASSIFICATION DATASET
# DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/pascal_voc.zip'    #DETECTION DATASET
# DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/coco.zip'          #DETECTION DATASET
# DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/oxford.zip'        #SEGMENTATION DATASET

SPLIT_SETS = ['train', 'val', 'test']

DATA_PATH = variables.get("DATA_PATH")
print("DATA_PATH: " + DATA_PATH)

if variables.get("TRAIN_SPLIT") is not None:
    SPLIT_TRAIN = float(str(variables.get("TRAIN_SPLIT")))
if variables.get("VAL_SPLIT") is not None:
    SPLIT_VAL = float(str(variables.get("VAL_SPLIT")))
if variables.get("TEST_SPLIT") is not None:
    SPLIT_TEST = float(str(variables.get("TEST_SPLIT")))

DATASET_TYPE = variables.get("DATASET_TYPE")
DATASET_TYPE = DATASET_TYPE.lower()
print("DATASET_TYPE: ", DATASET_TYPE)

print("Split information:")
print("SPLIT_TRAIN: " + str(SPLIT_TRAIN))
print("SPLIT_VAL:   " + str(SPLIT_VAL))
print("SPLIT_TEST:  " + str(SPLIT_TEST))

assert SPLIT_TRAIN >= 0.0
assert SPLIT_VAL >= 0.0
assert SPLIT_TEST >= 0.0
assert (SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST) == 1.0
if SPLIT_TRAIN == 0.0 and SPLIT_VAL > 0.0:
    raise AssertionError(
        "SPLIT_VAL cannot be defined when SPLIT_TRAIN equals zero")

if DATA_PATH is not None and DATA_PATH.startswith("http"):
    # Get an unique ID
    ID = str(uuid.uuid4())

    # Define localspace
    LOCALSPACE = join('data', ID)
    os.makedirs(LOCALSPACE, exist_ok=True)
    print("LOCALSPACE:  " + LOCALSPACE)

    DATASET_NAME = splitext(DATA_PATH[DATA_PATH.rfind("/") + 1:])[0]
    DATASET_PATH = join(LOCALSPACE, DATASET_NAME)
    os.makedirs(DATASET_PATH, exist_ok=True)

    print("Dataset information: ")
    print("DATASET_NAME: " + DATASET_NAME)
    print("DATASET_PATH: " + DATASET_PATH)

    print("Downloading...")
    filename = wget.download(DATA_PATH, DATASET_PATH)
    print("FILENAME: " + filename)
    print("OK")

    print("Extracting...")
    dataset_zip = zipfile.ZipFile(filename)
    dataset_zip.extractall(DATASET_PATH)
    dataset_zip.close()
    remove(filename)
    print("OK")
else:
    # load global dataset
    DATASET_PATH = variables.get('DATA_PATH')
    DATASET_NAME = DATASET_PATH
    globalspaceapi.connect()
    java_file = gateway.jvm.java.io.File(DATASET_PATH)
    globalspaceapi.pullFile(DATASET_PATH, java_file)


# remove train, test and val folder if exists
remove_folder = [shutil.rmtree(join(DATASET_PATH, SPLIT_NAME)) for SPLIT_NAME in SPLIT_SETS if exists(join(DATASET_PATH, SPLIT_NAME))]
    
# load classification dataset
class LoadClassDatset():

    def __init__(self):
        self.folder_name = []
        self.images_list = []
        self.label_list = []
        self.k = 0

        print('Dataset type:', DATASET_TYPE)

        # create dataset list
        for root in listdir(DATASET_PATH):
            if (not root.startswith('.')):
                self.folder_name.append(root)
                label_dir = join(DATASET_PATH, root)
                print(label_dir)
                files = listdir(label_dir)
                files[:] = [join(label_dir, file) for file in files]
                files_size = len(files)
                files_label = [self.k] * files_size
                self.images_list = self.images_list + files
                self.label_list = self.label_list + files_label
                self.k += 1

        print("Splitting the dataset into train and test")
        self.img_train, self.img_test, self.lab_train, self.lab_test = train_test_split(self.images_list,
                                                                                        self.label_list,
                                                                                        test_size=SPLIT_TEST,
                                                                                        random_state=1)

    # move images to specific folders
    def move_images(self, _path, _images_list, _labels_list):
        os.makedirs(_path, exist_ok=True)
        for _label_name in self.folder_name:
            makedirs(join(_path, _label_name))
        for _image_path, _image_label in zip(_images_list, _labels_list):
            _image_path_dest = join(_path, self.folder_name[_image_label], basename(_image_path))
            print("Moving " + _image_path + " to " + _image_path_dest)
            shutil.move(_image_path, _image_path_dest)

    def move_split_images(self):
        if SPLIT_TRAIN != 0.0 and SPLIT_VAL != 0.0:
            print("Splitting the train into train and val")
            images_train, images_val, labels_train, labels_val = train_test_split(self.img_train, self.lab_train,
                                                                                  test_size=SPLIT_VAL, random_state=1)

        images_split = {SPLIT_SETS[0]: images_train, SPLIT_SETS[1]: images_val, SPLIT_SETS[2]: self.img_test}
        labels_split = {SPLIT_SETS[0]: labels_train, SPLIT_SETS[1]: labels_val, SPLIT_SETS[2]: self.lab_test}

        # move images to train, val and test folders
        for SPLIT_NAME in SPLIT_SETS:
            SPLIT_PATH = join(DATASET_PATH, SPLIT_NAME)
            print("SPLIT_PATH:" + SPLIT_PATH)
            self.move_images(SPLIT_PATH, images_split[SPLIT_NAME], labels_split[SPLIT_NAME])

        dataset_labels = json.dumps(self.folder_name)
        print("DATASET_LABELS: " + dataset_labels)
        return dataset_labels

    
# load detection/segmentation dataset
class LoadSegObjDatset():

    def __init__(self):
        self.folder_name = ['images', 'classes']
        self.folder_number = len(next(os.walk(DATASET_PATH))[1])
        self.image_dir = join(DATASET_PATH, 'images')
        self.label_dir = join(DATASET_PATH, 'classes')
        self.images_list = []
        self.images_list_gt = []

        print('Dataset type:', DATASET_TYPE)

        # create image and class lists
        if os.path.exists(self.image_dir) and os.path.exists(self.label_dir) and self.folder_number == 2:
            # get images
            image_files = listdir(self.image_dir)
            image_files[:] = [join(self.image_dir, file) for file in image_files]
            self.images_list = self.images_list + image_files

            # get classes images
            label_files = listdir(self.label_dir)
            label_files[:] = [join(self.label_dir, file) for file in label_files]
            self.images_list_gt = self.images_list_gt + label_files

            # remove .DS_Store files
            new_list = [self.images_list.remove(''.join(x)) for x in self.images_list if re.search('.DS_Store', x)]
            new_list_gt = [self.images_list_gt.remove(''.join(x)) for x in self.images_list_gt if
                           re.search('.DS_Store', x)]

            # sort the elements of a given iterable in a specific order (either ascending or descending)
            self.images_list = sorted(self.images_list)
            self.images_list_gt = sorted(self.images_list_gt)
        else:
            print(
                "Please check if your dataset is in the correct format! It must contain only 'images' and 'classes' folders")

        print("Splitting the dataset into train and test")
        images_train, images_test, images_train_gt, images_test_gt = train_test_split(self.images_list,
                                                                                      self.images_list_gt,
                                                                                      test_size=SPLIT_TEST,
                                                                                      random_state=1)

        if SPLIT_TRAIN != 0.0 and SPLIT_VAL != 0.0:
            print("Splitting the train into train and val")
            images_train, images_val, images_train_gt, images_val_gt = train_test_split(images_train, images_train_gt,
                                                                                        test_size=SPLIT_VAL,
                                                                                        random_state=1)

            self.images_split = {SPLIT_SETS[0]: images_train, SPLIT_SETS[1]: images_val, SPLIT_SETS[2]: images_test}
            self.images_split_gt = {SPLIT_SETS[0]: images_train_gt, SPLIT_SETS[1]: images_val_gt,
                                    SPLIT_SETS[2]: images_test_gt}

    # moves images to specific folders
    def move_seg_images(self, _path, _images_list):
        os.makedirs(_path, exist_ok=True)
        for _image_path in _images_list:
            _image_path_dest = join(_path, basename(_image_path))
            print("Moving " + _image_path + " to " + _image_path_dest)
            shutil.move(_image_path, _image_path_dest)

    # move images to train, val and test folders
    def move_split_images(self):
        k = 0
        dataset_labels = ''
        for bx in self.folder_name:
            for SPLIT_NAME in SPLIT_SETS:
                SPLIT_PATH = join(DATASET_PATH, SPLIT_NAME)
                if k == 0:
                    print("SPLIT_PATH: " + SPLIT_PATH)
                    self.move_seg_images(SPLIT_PATH + '/' + self.folder_name[0], self.images_split[SPLIT_NAME])
                if k == 1:
                    print("SPLIT_PATH: " + SPLIT_PATH)
                    self.move_seg_images(SPLIT_PATH + '/' + self.folder_name[1], self.images_split_gt[SPLIT_NAME])
            k += 1
        return dataset_labels


if __name__ == '__main__':
    if DATASET_TYPE == 'classification':
        classdataset = LoadClassDatset()
        DATASET_LABELS =  classdataset.move_split_images()
    elif DATASET_TYPE == 'detection' or DATASET_TYPE == 'segmentation':
        segobjdataset = LoadSegObjDatset()
        DATASET_LABELS =   segobjdataset.move_split_images()
    else:
        print('Please, check your dataset type variable!')
        

if 'variables' in locals():
    variables.put("DATASET_NAME", DATASET_NAME)
    variables.put("DATASET_PATH", DATASET_PATH)
    variables.put("DATASET_LABELS", DATASET_LABELS)
    variables.put("DATASET_TYPE", DATASET_TYPE)

print("END " + __file__)