from os import getcwd, listdir
from pathlib import Path
from PIL import Image
from numpy import asarray

def image_to_array(path:Path, image_size:tuple, mode:str):
    image = Image.open(path)
    image = image.resize(image_size)
    image = image.convert(mode)
    image_array = asarray(image)
    return image_array
    
def get_images(image_size:tuple, augmented=False, mode='L'):
    yes_arrays = []
    no_arrays = []
    
    if augmented:
        brain_tumor_yes = r'data/brain_tumor_dataset/split_data/train_augmented/yes'
        brain_tumor_no = r'data/brain_tumor_dataset/split_data/train_augmented/no'
    else:
        brain_tumor_yes = r'data/brain_tumor_dataset/yes'
        brain_tumor_no = r'data/brain_tumor_dataset/no'
    
    yes_images = listdir(brain_tumor_yes)
    no_images = listdir(brain_tumor_no)
    
    for image in yes_images:
        if image != '.ipynb_checkpoints':
            image_path = brain_tumor_yes + r'/' + image
            yes_arrays.append(image_to_array(image_path, image_size, mode))

    for image in no_images:
        if image != '.ipynb_checkpoints':
            image_path = brain_tumor_no + r'/' + image
            no_arrays.append(image_to_array(image_path, image_size, mode))

    yes_arrays = asarray(yes_arrays)
    no_arrays = asarray(no_arrays)
    
    return yes_arrays, no_arrays