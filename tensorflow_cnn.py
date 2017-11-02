import tensorflow as tf
import numpy as np
import os

from tensorflow_image_tensor import get_tensor_for_image
from tensorflow_image_tensor import convert_image_to_tensor

def one_hot_convert(y):
    onehot_y = np.zeros(number_of_classes)
    onehot_y[y]=1  
    return onehot_y  

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def cnn_graph(x):
    
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First Convolution Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolution Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connnected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - Regularization Term for training larger neural network
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # Final Readout Layer
    W_fc2 = weight_variable([1024, number_of_classes])
    b_fc2 = bias_variable([number_of_classes])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return y_conv


def cnn_computation(y_conv):
    
    cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver() 
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            sess.run(train_step,feed_dict={x:train, y_:labels})
            train_accuracy = accuracy.eval(feed_dict={x:train, y_:labels})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        y_pred = tf.nn.softmax(y_conv,name="y_pred")
        saver.save(sess,'cats-dogs-model')
        return y_pred
            

# path_name = '/Users/ashu/Desktop/mlai/Pictures'
path_name = ""
path_name = raw_input("Image Directory Name : ")
current_path = os.getcwd()
path_name = os.path.join(current_path,path_name)
number_of_classes = 0
listing = os.listdir(path_name)
for file in listing:
        if(os.path.isdir(os.path.join(path_name,file))):
            number_of_classes +=1
train,labels,label_map = get_tensor_for_image(path_name)
for i in range(len(labels)):
    labels[i] = one_hot_convert(labels[i])
train = train.astype('float32')
labels = np.array(labels)
labels = labels.astype('float32')

# print(labels)
# labels=[]
# for i in range(20):
#     if i<10:
#         labels.append([1,0])
#     else:
#         labels.append([0,1])
# labels = np.array(labels)
# labels = labels.astype('float32')

x = tf.placeholder(tf.float32,[None,28,28,1],name="x")
y_= tf.placeholder(tf.float32,[None,number_of_classes],name="y_")
y_conv = cnn_graph(x)     
# y_pred = tf.nn.softmax(y_conv,name="y_pred")
y_pred = cnn_computation(y_conv)



def get_prediction(target_image):

    # target_image = '/Users/ashu/Desktop/mlai/cat.jpg'
    image_tensor = convert_image_to_tensor(target_image)
    images=[]
    images.append(image_tensor)
    images = np.array(images,dtype=np.uint8)
    images = images.astype('float32')
    # image_tensor = image_tensor.astype('float32')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result=sess.run(y_pred, feed_dict={x:images})
    # print(result)
    return result

result1 = get_prediction(os.path.join(os.getcwd(),"cat.jpg"))
result2 = get_prediction(os.path.join(os.getcwd(),"dog.jpg"))
# result3 = get_prediction(os.path.join(os.getcwd(),"tree.jpeg"))
# result4 = get_prediction(os.path.join(os.getcwd(),"house.jpeg"))
# print(result)
print("Cat Prediction :")
for i in range(len(result1[0])):
    print(label_map[i],result1[0][i])
print("Dog Prediction :")
for i in range(len(result1[0])):
    print(label_map[i],result2[0][i])
# print("Tree Prediction :")
# for i in range(len(result1[0])):
#     print(label_map[i],result3[0][i])
# print("House Prediction :")
# for i in range(len(result1[0])):
#     print(label_map[i],result4[0][i])



def cnn_model_computation(train,labels):
    logits = conn_graph(train,labels)
    predictions = {
        "classes" : tf.argmax(input=logits,axis=1),
        "probabilities" : tf.nn.softmax(logits,name="softmax_tensor")
    }
    
    
    # defining loss function
    onehot_labels = tf.one_hot(indices=labels,depth=2)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)
    
    # optimizing the loss/error function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)
    
    total_epoch=10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(total_epoch):
            epoch_loss,_ = sess.run([loss,train_op])
            print(epoch,epoch_loss)
#         accuracy =tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
#         print(accuracy)
#         print('Accuracy = ', sess.run(accuracy))

