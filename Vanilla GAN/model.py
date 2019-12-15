import os,sys
import pandas as pd
import tensorflow as tf  


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc_1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc_3 = tf.keras.layers.Dense(512, activation='relu')
        self.fc_4 = tf.keras.layers.Dense(28 * 28 * 1, activation='tanh')

        self.reshape = tf.keras.layers.Reshape((28, 28, 1))

    def call(self, inputs):
        x = self.fc_1(inputs)
        x = self.fc_2(inputs)
        x = self.fc_3(x)
        x = self.fc_4(x)
        return self.reshape(x)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.flatten = tf.keras.layers.Flatten()

        self.fc_1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc_3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc_4 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return self.fc_4(x)


if __name__ == "__main__":
    pass
