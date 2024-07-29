from tda_pipeline import pipeline
from data.image_converter import get_images
import pickle

def train_and_save(size=(224, 224), original=False):
    yes, no = get_images(size)
    yes_piped = pipeline(yes, original)
    no_piped = pipeline(no, original)
    
    if not original:
        yes_path = r'models/yes_piped.pickle'
        no_path = r'models/no_piped.pickle'
    else:
        yes_path = r'models/yes_piped_original.pickle'
        no_path = r'models/no_piped_original.pickle'
        
    with open(yes_path, 'wb') as file:
        pickle.dump(yes_piped, file)
        file.close()
    with open(no_path, 'wb') as file:
        pickle.dump(no_piped, file)
        file.close()
        
train_and_save(size=(224, 224), original=True)