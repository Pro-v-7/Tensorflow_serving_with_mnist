import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
loaded = tf.saved_model.load("/maaru_model2/1538687457")
print(loaded)
infer = loaded.signatures["serving_default"]
print(infer.function_def)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(loaded.predict([x_test[0]]))

infer = loaded.signatures["predict"]
print(infer.structured_outputs)
