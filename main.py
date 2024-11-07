# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries

import numpy as np
import matplotlib.pyplot as plt

import time as time


if __name__ == '__main__':
    start_time = time.time()

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print("%s seconds" % (time.time()-start_time))
