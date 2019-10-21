from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initializing CNN

classifier=Sequential()

#Convolution
classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,3),activation='relu'))
#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding another CNN LAYER
classifier.add(Convolution2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening
classifier.add(Flatten())
# full connection

classifier.add(Dense(activation='relu',units=128))
classifier.add(Dense(activation='softmax',units=6))

#compile
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/content/gdrive/My Drive/waste/train',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='categorical')

test_set = test_datagen.flow_from_directory('/content/gdrive/My Drive/waste/test',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch=2370,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=217)

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('/content/gdrive/My Drive/waste/plastic single test.jpg', target_size=(64, 64))
test_image=image.img_to_array(test_image)


test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image)
print(result)
training_set.class_indices