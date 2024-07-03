from os import getcwd, listdir
from pathlib import Path
from PIL import Image
from numpy import asarray

current_path = getcwd()
brain_tumor_yes = Path(current_path + r'\data\brain_tumor_dataset\yes')
brain_tumor_no = Path(current_path + r'\data\brain_tumor_dataset\no')

yes_images = listdir(brain_tumor_yes)
no_images = listdir(brain_tumor_no)


def image_to_array(path:Path, image_size:tuple):
    image = Image.open(path)
    image = image.resize(image_size)
    image = image.convert('L')
    image_array = asarray(image)
    return image_array
    
def get_images(image_size=tuple):
    yes_arrays = []
    no_arrays = []
    
    for image in yes_images:
        if image != '.ipynb_checkpoints':
            image_path = Path(str(brain_tumor_yes) + r'\\' + image)
            yes_arrays.append(image_to_array(image_path, image_size))
        
    for image in no_images:
        if image != '.ipynb_checkpoints':
            image_path = Path(str(brain_tumor_no) + r'\\' + image)
            no_arrays.append(image_to_array(image_path, image_size))
    
    yes_arrays = asarray(yes_arrays)
    no_arrays = asarray(no_arrays)
    
    return yes_arrays, no_arrays