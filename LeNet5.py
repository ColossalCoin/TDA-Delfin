import numpy as np
from data.image_converter import get_images
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

images_yes, images_no = get_images((224, 224), mode='L')
X = np.concatenate((images_yes, images_no))
y = np.concatenate((np.ones(images_yes.shape[0]), np.zeros(images_no.shape[0])))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=123)

def LeNet_5():
    model = Sequential()
    
    # C1: (None, 224, 224, 1) -> (None, 220, 220, 6).
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                     input_shape=(224, 224, 1), padding='valid'))
    
    # P1: (None, 220, 220, 6) -> (None, 110, 110, 6).
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    # C2: (None, 110, 110, 6) -> (None, 106, 106, 16).
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh',
                     padding='valid'))
    
    # P2: (None, 106, 106, 16) -> (None, 53, 53, 16).
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    # Flatten: (None, 53, 53, 16) -> (None, 179 776).
    model.add(Flatten())
    
    # FC1: (None, 179 776) -> (None, 120).
    model.add(Dense(120, activation='tanh'))
    
    # FC2: (None, 120) -> (None, 84).
    model.add(Dense(84, activation='tanh'))
    
    # FC3: (None, 84) -> (None, 10).
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

model = LeNet_5()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
# Save the model
model.save('lenet5_model.h5')