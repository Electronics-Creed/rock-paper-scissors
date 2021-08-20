'''!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip \
    -O /tmp/rps.zip
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip \
    -O /tmp/rps-test-set.zip'''

import zipfile

Localzip = '/tmp/rps.zip'
Zipref = zipfile.ZipFile(Localzip, 'r')
Zipref.extractall('/tmp/')
Zipref.close()

Localzip = '/tmp/rps-test-set.zip'
Zipref = zipfile.ZipFile(Localzip, 'r')
Zipref.extractall('/tmp/')
Zipref.close()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
                                   tf.keras.layers.MaxPooling2D(2, 2),
                                   tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                                   tf.keras.layers.MaxPooling2D(2, 2),
                                   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                   tf.keras.layers.MaxPooling2D(2, 2),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                   tf.keras.layers.Dropout(0.5),
                                   tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                   tf.keras.layers.Dropout(0.5),
                                   tf.keras.layers.Dense(3, activation=tf.nn.softmax)])

model.summary()

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = "/tmp/rps/"
validation_dir = "/tmp/rps-test-set/"

train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size = (300, 300), class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size = (300, 300), class_mode = 'categorical')

history = model.fit_generator(train_generator, validation_data = validation_generator, epochs=5, verbose=1)

model.save_weights('my_model_weights1.h5')
model.save('my_model1.h5')

