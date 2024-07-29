from tda_pipeline import pipeline
from data.image_converter import get_images
import pickle

yes, no = get_images((224, 224))

yes_piped = pipeline(yes)
no_piped = pipeline(no)

with open('models/yes_piped.pickle', 'wb') as file:
    pickle.dump(yes_piped, file)
    file.close()
with open('models/no_piped.pickle', 'wb') as file:
    pickle.dump(no_piped, file)
    file.close()