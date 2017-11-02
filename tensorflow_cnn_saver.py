import tensorflow as tf
import numpy as np
import os

from tensorflow_image_tensor import convert_image_to_tensor

def get_prediction(target_image = '/home/iiita/Documents/HashtagGen-master/cat.jpg'):
    # target_image = '/home/iiita/Documents/HashtagGen-master/cat.jpg'
    image_tensor = convert_image_to_tensor(target_image)
    images=[]
    images.append(image_tensor)
    images = np.array(images,dtype=np.uint8)
    images = images.astype('float32')

    # Start Session
    sess = tf.Session()
    saver = tf.train.import_meta_graph('cats-dogs-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()

    # Get prediction model and feature placeholder
    y_pred = graph.get_tensor_by_name('y_pred:0')
    x= graph.get_tensor_by_name("x:0")
    # y_ = graph.get_tensor_by_name("y_:0") 
    # y_test_images = np.zeros((1, 2)) 

    sess.run(tf.global_variables_initializer())
    result=sess.run(y_pred, feed_dict={x:images}) # Getting prediction for the image
    # print(result)
    return result


path_name = raw_input("Image Directory Name : ")
current_path = os.getcwd()
path_name = os.path.join(current_path,path_name)
label_map = {}
listing = os.listdir(path_name)
class_number = 0
for file in listing:
        current_path = os.path.join(path_name,file)
        if(os.path.isdir(current_path)):
            label_map[class_number] = os.path.basename(current_path)
            class_number += 1

result = get_prediction(os.path.join(os.getcwd(),"cat.jpg")) 
# print(result)    
print("Cat Prediction :")
for i in range(len(result[0])):
    print(label_map[i],result[0][i])
