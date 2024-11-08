# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries

import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    cifar100 = tf.keras.datasets.cifar100

    start_time = time.time()

    (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(100, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=50)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    prediction = model.predict(test_images)

    print("%s seconds" % (time.time()-start_time))
