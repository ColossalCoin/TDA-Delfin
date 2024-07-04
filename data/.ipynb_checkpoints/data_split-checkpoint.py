from os import getcwd, listdir, mkdir
from random import shuffle
from shutil import copy2

current_path = getcwd()
work_path = current_path + r'\data\brain_tumor_dataset'
yes_path = work_path + r'\yes'
no_path = work_path + r'\no'

train_dir = work_path + r'\\train'
val_dir = work_path + r'\\val'
test_dir = work_path + r'\\test'

try:
    mkdir(train_dir)
    mkdir(train_dir + r'\yes')
    mkdir(train_dir + r'\no')
    
    mkdir(val_dir)
    mkdir(val_dir + r'\yes')
    mkdir(val_dir + r'\no')
    
    mkdir(test_dir)
    mkdir(test_dir + r'\yes')
    mkdir(test_dir + r'\no')
except FileExistsError:
    print(r'Data has been already split. Delete train, val, and test',
          'directories to split the data again.')
    
          
def train_val_test_split(sizes:tuple):
    if sum(sizes) != 100:
        print(f'Tuple sum must be 100 but is {sum(sizes)} instead.')
        return 0
    
    yes_images = listdir(yes_path)
    no_images = listdir(no_path)
    shuffle(yes_images); shuffle(no_images)
    
    train_idx_yes = int(len(yes_images) * (sizes[0] / 100))
    val_idx_yes = int(len(yes_images) * ((sizes[0] + sizes[1]) / 100))
    
    train_idx_no = int(len(no_images) * (sizes[0] / 100))
    val_idx_no = int(len(no_images) * ((sizes[0] + sizes[1]) / 100))
    
    train_images_yes = yes_images[:train_idx_yes]
    val_images_yes = yes_images[train_idx_yes:val_idx_yes]
    test_images_yes = yes_images[val_idx_yes:]
    
    train_images_no = no_images[:train_idx_no]
    val_images_no = no_images[train_idx_no:val_idx_no]
    test_images_no = no_images[val_idx_no:]
    
    for image in train_images_yes:
        try:
            copy2(yes_path + r'\\' + image, train_dir + r'\\yes\\' + image)
        except FileExistsError:
            print(r'Train data has been already set. Delete train, val, and test',
                  'directories to split the data again.')
        except PermissionError:
            pass
    for image in train_images_no:
        try:
            copy2(no_path + r'\\' + image, train_dir + r'\\no\\' + image)
        except FileExistsError:
            print(r'Train data has been already set. Delete train, val, and test',
                  'directories to split the data again.')
        except PermissionError:
            pass
    for image in val_images_yes:
        try:
            copy2(yes_path + r'\\' + image, val_dir + r'\\yes\\' + image)
        except FileExistsError:
            print(r'val data has been already set. Delete val, val, and test',
                  'directories to split the data again.')
        except PermissionError:
            pass
    for image in val_images_no:
        try:
            copy2(no_path + r'\\' + image, val_dir + r'\\no\\' + image)
        except FileExistsError:
            print(r'val data has been already set. Delete val, val, and test',
                  'directories to split the data again.')
        except PermissionError:
            pass
    for image in test_images_yes:
        try:
            copy2(yes_path + r'\\' + image, test_dir + r'\\yes\\' + image)
        except FileExistsError:
            print(r'test data has been already set. Delete test, test, and test',
                  'directories to split the data again.')
        except PermissionError:
            pass
    for image in test_images_no:
        try:
            copy2(no_path + r'\\' + image, test_dir + r'\\no\\' + image)
        except FileExistsError:
            print(r'test data has been already set. Delete test, test, and test',
                  'directories to split the data again.')
        except PermissionError:
            pass
    print('Data splitted correctly.')
    return None

train_val_test_split((70, 20, 10))