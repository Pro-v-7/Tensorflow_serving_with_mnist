from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf
import os
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

hostport="localhost:8500"

mnist = tf.keras.datasets.mnist
(x_train, train_y), (x_test, test_y) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
train_x = x_train[..., tf.newaxis]
test_x = x_test[..., tf.newaxis]


channel = grpc.insecure_channel(hostport)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# create the request object and set the name and signature_name params
request = predict_pb2.PredictRequest()
request.model_spec.name = 'm3'
request.model_spec.signature_name = 'sevring_default'

# fill in the request object with the necessary data
request.inputs['x'].CopyFrom(
  tf.make_tensor_proto(test_x[0].astype(dtype=np.float32), shape=[1,28,28,1]))
p=stub.Predict(request, 5.0)
sess=tf.Session()

print(p.outputs['f'])
print( tf.make_ndarray(p.outputs['f']))
