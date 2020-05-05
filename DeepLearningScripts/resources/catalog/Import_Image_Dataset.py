__file__ = variables.get("PA_TASK_NAME")

#print("BEGIN Import_Image_Dataset")

import re
import json
import wget
import uuid 
import shutil
import zipfile
from os import remove, listdir, makedirs
from os.path import basename, splitext, exists, join
from sklearn.model_selection import train_test_split
 
#DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/ants_vs_bees.zip'  #CLASSIFICATION DATASET
#DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/pascal_voc.zip'    #DETECTION DATASET
#DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/coco.zip'          #DETECTION DATASET
#DATASET_URL = 'https://s3.eu-west-2.amazonaws.com/activeeon-public/datasets/oxford.zip'        #SEGMENTATION DATASET

SPLIT_SETS  = ['train','val','test']

DATA_PATH = variables.get("DATA_PATH")
print("DATA_PATH: " + DATA_PATH)

if variables.get("TRAIN_SPLIT") is not None:
    SPLIT_TRAIN = float(str(variables.get("TRAIN_SPLIT")))
if variables.get("VAL_SPLIT") is not None:
    SPLIT_VAL = float(str(variables.get("VAL_SPLIT")))
if variables.get("TEST_SPLIT") is not None:
    SPLIT_TEST = float(str(variables.get("TEST_SPLIT")))
    
DATASET_TYPE = variables.get("DATASET_TYPE")
DATASET_TYPE = DATASET_TYPE.upper()
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
  raise AssertionError("SPLIT_VAL cannot be defined when SPLIT_TRAIN equals zero") 
        
if DATA_PATH is not None and DATA_PATH.startswith("http"):
    # Get an unique ID
    ID = str(uuid.uuid4())

    # Define localspace
    LOCALSPACE = join('data', ID)
    os.makedirs(LOCALSPACE, exist_ok=True)
    print("LOCALSPACE:  " + LOCALSPACE)

    DATASET_NAME = splitext(DATA_PATH[DATA_PATH.rfind("/")+1:])[0]
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
    DATASET_PATH = variables.get('DATA_PATH')
    DATASET_NAME = DATASET_PATH
    globalspaceapi.connect()
    java_file = gateway.jvm.java.io.File(DATASET_PATH)
    globalspaceapi.pullFile(DATASET_PATH, java_file)

#CLASSIFICATION DATASET 
if DATASET_TYPE == 'CLASSIFICATION':  
    k = 0
    images_list = []
    label_list = []
    folder_name = []
     
    # remove train, test and val folder if exists
    for SPLIT_NAME in SPLIT_SETS:
        SPLIT_PATH = join(DATASET_PATH, SPLIT_NAME)
        if exists(SPLIT_PATH):
            shutil.rmtree(SPLIT_PATH)
        
    for root in listdir(DATASET_PATH):
        if (not root.startswith('.')):
            folder_name.append(root)
            label_dir = join(DATASET_PATH, root)
            print(label_dir)
            files = listdir(label_dir)
            files[:] = [join(label_dir,file) for file in files]
            files_size = len(files)
            files_label = [k] * files_size
            images_list = images_list + files
            label_list = label_list  + files_label
            k = k + 1

    print("Splitting the dataset into train and test")
    images_train, images_test, labels_train, labels_test = train_test_split(images_list, label_list, test_size=SPLIT_TEST, random_state=1)

    images_val = []
    labels_val = []
    if SPLIT_TRAIN != 0.0 and SPLIT_VAL != 0.0:
        print("Splitting the train into train and val")
        images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=SPLIT_VAL, random_state=1)

    images_split = {SPLIT_SETS[0] : images_train, SPLIT_SETS[1] : images_val, SPLIT_SETS[2] : images_test}
    labels_split = {SPLIT_SETS[0] : labels_train, SPLIT_SETS[1] : labels_val, SPLIT_SETS[2] : labels_test}

    def move_images(_path, _images_list, _labels_list, _folder_name):
        os.makedirs(_path, exist_ok=True)
        for _label_name in _folder_name:
            makedirs(join(_path, _label_name))
        for _image_path, _image_label in zip(_images_list, _labels_list):
            _image_path_dest = join(_path, _folder_name[_image_label], basename(_image_path))
            print("Moving " + _image_path + " to " + _image_path_dest)
            shutil.move(_image_path, _image_path_dest)

    DATASET_LABELS = json.dumps(folder_name)
    print("DATASET_LABELS: " + DATASET_LABELS)

    for SPLIT_NAME in SPLIT_SETS:
        SPLIT_PATH = join(DATASET_PATH, SPLIT_NAME)
        if exists(SPLIT_PATH):
            shutil.rmtree(SPLIT_PATH)
        print("SPLIT_PATH:" + SPLIT_PATH)
        move_images(SPLIT_PATH, images_split[SPLIT_NAME], labels_split[SPLIT_NAME], folder_name)

    if 'variables' in locals():
        variables.put("DATASET_NAME", DATASET_NAME)
        variables.put("DATASET_PATH", DATASET_PATH)
        variables.put("DATASET_LABELS", DATASET_LABELS)
        variables.put("DATASET_TYPE", DATASET_TYPE)        
    
    print("END Import_Image_Dataset")

#DETECTION / SEGMENTATION DATASET    
elif DATASET_TYPE == 'DETECTION' or DATASET_TYPE == 'SEGMENTATION':
    print(DATASET_TYPE)
    k = 0
    images_list = []
    images_list_gt = []
    folder_name = ['images', 'classes']          
    folder_number = len(next(os.walk(DATASET_PATH))[1])
    
    # remove train, test and val folder if exists
    for SPLIT_NAME in SPLIT_SETS:
        SPLIT_PATH = join(DATASET_PATH, SPLIT_NAME)
        if exists(SPLIT_PATH):
            shutil.rmtree(SPLIT_PATH)
            
    if folder_number == 2:
        image_dir = join(DATASET_PATH, 'images')
        label_dir = join(DATASET_PATH, 'classes')
  
        if os.path.exists(image_dir) and not os.path.isfile(image_dir):
            if os.path.exists(label_dir) and not os.path.isfile(label_dir):
                image_files = listdir(image_dir)
                image_files[:] = [join(image_dir, file) for file in image_files]
                images_list = images_list + image_files

                label_files = listdir(label_dir)
                label_files[:] = [join(label_dir, file) for file in label_files]
                images_list_gt = images_list_gt + label_files
                
    try:
        new_list = [x for x in images_list if re.search('.DS_Store', x)]
        images_list.remove(''.join(new_list))
    except:
        pass            

    try:
        new_list_gt = [x for x in images_list_gt if re.search('.DS_Store', x)]
        images_list_gt.remove(''.join(new_list_gt))    
    except:
        pass    
    
    images_list = sorted(images_list)
    images_list_gt = sorted(images_list_gt)
            
    print("Splitting the dataset into train and test")
    images_train, images_test, images_train_gt, images_test_gt = train_test_split(images_list, images_list_gt, test_size=SPLIT_TEST, random_state=1)
    
    images_val = []

    if SPLIT_TRAIN != 0.0 and SPLIT_VAL != 0.0:
        print("Splitting the train into train and val")
        images_train, images_val, images_train_gt, images_val_gt = train_test_split(images_train, images_train_gt, test_size=SPLIT_VAL, random_state=1)

        images_split = {SPLIT_SETS[0] : images_train, SPLIT_SETS[1] : images_val, SPLIT_SETS[2] : images_test}
        images_split_gt = {SPLIT_SETS[0] : images_train_gt, SPLIT_SETS[1] : images_val_gt, SPLIT_SETS[2] : images_test_gt}

    for SPLIT_NAME in SPLIT_SETS:
        SPLIT_PATH = join(DATASET_PATH, SPLIT_NAME)
        print("SPLIT_PATH: " + SPLIT_PATH)

            
    def move_seg_images(_path, _images_list, _folder_name):
        os.makedirs(_path, exist_ok=True)
        for _image_path in _images_list:
            _image_path_dest = join(_path, basename(_image_path))
            print("Moving " + _image_path + " to " + _image_path_dest)
            shutil.move(_image_path, _image_path_dest)

    k = 0
    for bx in folder_name:
        for SPLIT_NAME in SPLIT_SETS:
            SPLIT_PATH = join(DATASET_PATH, SPLIT_NAME)
            if k == 0:
                print("SPLIT_PATH: " + SPLIT_PATH)
                move_seg_images(SPLIT_PATH + '/' + folder_name[0], images_split[SPLIT_NAME], folder_name)           
            if k == 1:
                print("SPLIT_PATH: " + SPLIT_PATH)
                move_seg_images(SPLIT_PATH + '/' + folder_name[1], images_split_gt[SPLIT_NAME], folder_name)
        k = k + 1            
        
    if 'variables' in locals():
        variables.put("DATASET_NAME", DATASET_NAME)
        variables.put("DATASET_PATH", DATASET_PATH)
        variables.put("DATASET_TYPE", DATASET_TYPE) 
        
    print("END Import_Image_Dataset")

    if folder_number != 2:    
        print('Please, check your dataset!')
        
else: 
    print('Please, check your dataset type variable!')
    
print("END " + __file__)
