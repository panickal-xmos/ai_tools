# %%
from operator import xor
import tensorflow as tf

inputs = tf.keras.Input(shape=(160,160, 3),)
x0 = tf.keras.layers.Conv2D(16,(3, 3), strides=(2,2),activation='relu',padding='same')(inputs)
x_skip = x0
x1 = tf.keras.layers.DepthwiseConv2D(3,padding='same')(x0)
x2 = tf.keras.layers.Conv2D(16,1)(x1)
x3 = tf.keras.layers.Add()([x2,x2])
x4 = tf.keras.layers.Flatten()(x3)
outputs = tf.keras.layers.Dense(10, activation='relu')(x4)
model = tf.keras.Model(inputs=inputs,outputs=outputs)

model.summary()

# %%
