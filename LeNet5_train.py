import matplotlib.pyplot as plt
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from data.image_converter import get_images

images_yes, images_no = get_images((224, 224), mode='L')
X = np.concatenate((images_yes, images_no))
y = np.concatenate((np.ones(images_yes.shape[0]), np.zeros(images_no.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

# Restore the model
model = tf.keras.models.load_model('lenet5_model.h5')

# Make prediction.
predictions = model.predict(X_test)

#Retrieve predictions indexes.
y_pred = np.argmax(predictions, axis=1)

# Print test set accuracy
print('Test set error rate: {}'.format(np.mean(y_pred == y_test)))

# Plot training error.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.show()