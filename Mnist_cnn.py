import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


training_iters = 1
learning_rate = 0.001
batch_size = 128


# data=input_data.read_data_sets('mnist/fashion',one_hot=True)
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}


n_input = 28

# MNIST total classes (0-9 digits)
n_classes = 10

mnist = tf.keras.datasets.mnist
(x_train, train_y), (x_test, test_y) = mnist.load_data()


def one_hot(t):
    l = []
    for i in t:
        p = np.zeros(10)
        p[i] = 1
        l.append(p)
    return l


train_y = one_hot(train_y)
test_y = one_hot(test_y)
print(train_y)
#train_y = data.train.labels
#test_y = data.test.labels
train_x = x_train[..., tf.newaxis]
test_x = x_test[..., tf.newaxis]

# both placeholders are of type float
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


weights = {
    'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32)),
    'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64)),
    'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128)),
    'wd1': tf.get_variable('W3', shape=(4*4*128, 128)),
    'out': tf.get_variable('W6', shape=(128, n_classes)),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32)),
    'bc2': tf.get_variable('B1', shape=(64)),
    'bc3': tf.get_variable('B2', shape=(128)),
    'bd1': tf.get_variable('B3', shape=(128)),
    'out': tf.get_variable('B4', shape=(10)),
}


def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


pred = conv_net(x, weights, biases)
print(x,pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_x)//batch_size):
            batch_x = train_x[batch*batch_size:min((batch+1)*batch_size, len(train_x))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size, len(train_y))]
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                 y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " +
              "{:.6f}".format(loss) + ", Training Accuracy= " +
              "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_x, y: test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    #summary_writer.close()






    saver = tf.saved_model.builder.SavedModelBuilder("./example_model/2")
    info_input = tf.saved_model.utils.build_tensor_info(x)
    info_output = tf.saved_model.utils.build_tensor_info(pred)
    signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x':info_input}
            ,outputs={'f':info_output}
            ,method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
    saver.add_meta_graph_and_variables(
            sess
            ,[tf.saved_model.tag_constants.SERVING]
            ,signature_def_map={'sevring_default':signature}
            )
    saver.save()

# Close and clean up
sess.close()
