import tensorflow as tf
import numpy as np
import cv2
import os

channels=3
n_class=2

class UNet(object):
    def __init__(self, x, y, img_rows = 256, img_cols = 256):
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.x=x
        self.y=y
        self.n_class=2
        self.model=self.model()

    def model(self):
        conv1_1=self.conv(self.x, 3, 64)
        conv1_2=self.conv(conv1_1, 3, 64)
        pool1=self.max_pool(conv1_2, 2, 2)

        conv2_1=self.conv(pool1, 3, 128)
        conv2_2=self.conv(conv2_1, 3, 128)
        pool2=self.max_pool(conv2_2, 2, 2)

        conv3_1=self.conv(pool2, 3, 256)
        conv3_2=self.conv(conv3_1, 3, 256)
        pool3=self.max_pool(conv3_2, 2, 2)

        conv4_1=self.conv(pool3, 3, 512)
        conv4_2=self.conv(conv4_1, 3, 512)
        pool4=self.max_pool(conv4_2, 2, 2)

        conv5_1=self.conv(pool4, 3, 1024)
        conv5_2=self.conv(conv5_1, 3, 1024)

        deconv1_1=self.deconv(conv5_2, 2)
        deconv1_2=self.concatenate(deconv1_1,conv4_2)
        deconv1_3=self.conv(deconv1_2, 3, 512)
        deconv1_4=self.conv(deconv1_3, 3, 512)

        deconv2_1=self.deconv(deconv1_4, 2)
        deconv2_2=self.concatenate(deconv2_1,conv3_2)
        deconv2_3=self.conv(deconv2_2, 3, 256)
        deconv2_4=self.conv(deconv2_3, 3, 256)

        deconv3_1=self.deconv(deconv2_4, 2)
        deconv3_2=self.concatenate(deconv3_1,conv2_2)
        deconv3_3=self.conv(deconv3_2, 3, 128)
        deconv3_4=self.conv(deconv3_3, 3, 128)

        deconv4_1=self.deconv(deconv3_4, 2)
        deconv4_2=self.concatenate(deconv4_1,conv1_2)
        deconv4_3=self.conv(deconv4_2, 3, 64)
        deconv4_4=self.conv(deconv4_3, 3, 64)

        deconv_shape=deconv4_4.get_shape().as_list()
        weights=tf.Variable(tf.random_normal((1,1,deconv_shape[3],self.n_class),stddev=0.05))
        biases=tf.Variable(tf.constant(0.05),1)
        self.output=(tf.nn.conv2d(input=deconv4_4,filter=weights,strides=[1,1,1,1], padding='SAME')+biases)



    def conv(self, input, kernel_size ,num_kernels):

        ip_shape=input.get_shape().as_list()

        shape=[ip_shape[0], ip_shape[1]*2, ip_shape[2]*2, ip_shape[3]//2]
        #Initialize the weights
        weights=tf.Variable(tf.random_normal((kernel_size,kernel_size,ip_shape[3],num_kernels),stddev=0.05))
        #Initialize the biases
        biases=tf.Variable(tf.constant(0.05),num_kernels)
        layer=tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
        layer+=biases
        #Use ReLU activation
        return tf.nn.relu(layer)

    def max_pool(self, input, kernel_size, stride):
        return tf.nn.max_pool(value=input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME')

    def deconv(self, input, kernel_size):
        ip_shape=input.get_shape().as_list()
        weights = tf.Variable(tf.truncated_normal((kernel_size,kernel_size,ip_shape[3]//2,ip_shape[3]),stddev=0.05))
        biases=tf.Variable(tf.constant(0.05),ip_shape[3]//2)
        output_shape=[ip_shape[0], ip_shape[1]*2, ip_shape[2]*2, ip_shape[3]//2]
        layer = tf.nn.conv2d_transpose(input, weights, output_shape, strides=[1,2,2,1], padding='SAME')
        layer+=biases
        return tf.nn.relu(layer)

    def concatenate(self, layer1, layer2):
        return tf.concat([layer1,layer2], axis=3)

    def predict(self, model_path, x_test):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction
