# Tensorflow_serving_with_mnist
Creating ML model is a task that requires most attantion while building the AI related business solution. But it is equally necessary to oversee hoe the model will be used after production for usage by user. Tensorflow provides this servicd called tensorflow serving for packing the model and necessary data into a docker container and then deploye it over any server for predictions. 

The mootive for creating this repository is to help the newbies of data science to understand the importance of tensorflow serving with practical example since no other content is available.

Steps:

1. First create the model using the mnist_cnn.py

2. move the created pb file and variables into a new folder called "your model name"

3. install docker if not installed

4. run "sudo docker container kill $(sudo docker ps -q)" to kill running containers

5. run "sudo docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/path/to/model/model_name,target=/models/model_name -e MODEL_NAME=m3 -t tensorflow/serving &" to deploy model

6. change the model name and host port variables into client program

7. run client program



tensorflow serving:
https://www.tensorflow.org/tfx/guide/serving
