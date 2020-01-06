# Tensorflow_serving_with_mnist
Creating ML model is a task that requires most attantion while building the AI related business solution. But it is equally necessary to oversee hoe the model will be used after production for usage by user. Tensorflow provides this servicd called tensorflow serving for packing the model and necessary data into a docker container and then deploye it over any server for predictions. 


Steps:

First create the model using the mnist_cnn.py
move the created pb file and variables into a folder 
install docker if not installed
run "sudo docker container kill $(sudo docker ps -q)" to kill running containers
run "sudo docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/path/to/model/model_name,target=/models/model_name -e MODEL_NAME=m3 -t tensorflow/serving &" to deploy model
change the model name and host port variables into client program
run client program

tensorflow serving:
https://www.tensorflow.org/tfx/guide/serving
