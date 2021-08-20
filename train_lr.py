import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='valid', input_shape=(300, 300, 3)),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid', strides=(3, 3)),
     tf.keras.layers.MaxPooling2D(2, 2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(1024, activation=tf.nn.relu),

     tf.keras.layers.Dense(256, activation=tf.nn.relu),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(3, activation=tf.nn.softmax)])

model.summary()

optimizer = tf.keras.optimizers.Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = "C:\\Users\\jeeva\\PycharmProjects\\Rock_paper_sicsir\\dataset-rps\\rps-train-set"
validation_dir = "C:\\Users\\jeeva\\PycharmProjects\\Rock_paper_sicsir\\dataset-rps\\rps-test-set"

train_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=20)
test_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=20)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), batch_size=64, shuffle=True,
                                                    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(300, 300), batch_size=64,
                                                        shuffle=True, class_mode='categorical')

history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=30, verbose=1)

model.save_weights('my_model_weights1.h5')
model.save('my_model1.h5')
