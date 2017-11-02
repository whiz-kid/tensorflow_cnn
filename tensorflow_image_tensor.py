import tensorflow as tf
import numpy as np
import os

# from tensorflow_cnn import number_of_classes

def convert_image_to_tensor(file_name):
    input_height=28
    input_width=28
    input_mean=128
    input_std=128
    
    file_reader=tf.read_file(file_name)
    image_reader = tf.image.decode_jpeg(file_reader, channels = 1,name='jpeg_reader')
    float_caster = tf.cast(image_reader,tf.float32)
    resized = tf.image.resize_images(float_caster,[input_height,input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    
    with tf.Session() as sess:
        # print(sess.run(tf.shape(normalized)))
        return sess.run(normalized)


def get_tensor_for_image(path1):

    listing1 = os.listdir(path1)
    train = []
    label = []
    label_map = {}
    class_number = -1
    ext = ['jpeg','jpg','png']
    # sess=tf.Session()
    for file in listing1:
        if(os.path.isdir(os.path.join(path1,file))):
            class_number +=1
            path2 = os.path.join(path1,file)
            listing2 = os.listdir(path2)
            for image in listing2:
                if image.lower().endswith(tuple(ext)):
                    label_map[class_number]=os.path.basename(path2)
                    file_name = os.path.join(path2,image)
                    tensor = convert_image_to_tensor(file_name)
                    train.append(tensor)
                    # onehot_label = one_hot_convert(class_number)
                    label.append(class_number)
                    # print(sess.run(tf.shape(tensor)))
    # print(train) 
    return (np.array(train),label,label_map)
