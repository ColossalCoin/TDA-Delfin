from os import getcwd, listdir
from pathlib import Path
from PIL import Image
from numpy import asarray, array
from pandas import Series

current_path = getcwd()
brain_tumor_yes = Path(current_path + r'\data\brain_tumor_dataset\yes')
brain_tumor_no = Path(current_path + r'\data\brain_tumor_dataset\no')

yes_images = listdir(brain_tumor_yes)
no_images = listdir(brain_tumor_no)


def image_to_array(path:Path):
    image = Image.open(path)
    image = image.convert('L')
    image_array = asarray(image)
    return image_array
    
def get_images():
    yes_arrays = []
    no_arrays = []
    
    for image in yes_images:
        if image != '.ipynb_checkpoints':
            image_path = Path(str(brain_tumor_yes) + r'\\' + image)
            yes_arrays.append(image_to_array(image_path))
        
    for image in no_images:
        if image != '.ipynb_checkpoints':
            image_path = Path(str(brain_tumor_no) + r'\\' + image)
            no_arrays.append(image_to_array(image_path))
    
    yes_arrays = Series(yes_arrays)
    no_arrays = Series(no_arrays)
    
    return yes_arrays, no_arrays

yes, no = get_images()