import tensorflow as tf
import prepare_data as pp
import model
import numpy as np
from sklearn.cross_validation import train_test_split

batch_size=20
epochs=100
learning_rate=0.000000001
img_rows=32
img_cols=32
channels=3



def preprocess(train_images):
    std=np.std(train_images)
    mean=np.mean(train_images)
    train_images -= mean
    train_images /= std
    return train_images

def split_data(train_images,train_masks):
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.15, random_state=42)
    return X_train, X_val, y_train, y_val

def train():
    batch_size=20
    images,masks=pp.get_train_data()

    images=images.astype('float32')
    images=preprocess(images)
    X_train, X_val, y_train, y_val=split_data(images,masks)
    x=tf.placeholder("float", shape=[8, img_rows, img_cols, 3])
    y=tf.placeholder("float", shape=[8, img_rows, img_cols, 2])
    #with tf.Session() as sess:
    #if write_graph:
    #    tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

    nn_model=model.UNet(x,y)

    output_map=nn_model.output

    var=tf.trainable_variables()
    flat_logits = tf.reshape(output_map, [-1, 2])

    flat_labels = tf.reshape(y, [-1, 2])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    #with tf.name_scope("cross_ent"):
    #    loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_map,labels=y))

    #with tf.name_scope("train"):
    #    gradients=tf.gradients(loss,var)
    #    gradients=list(zip(gradients,var))

    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    #for gradient, var in gradients:
    #tf.summary.histogram(var.name + '/gradient', gradient)

    #for var in var_list:
    #tf.summary.histogram(var.name, var)

    #tf.summary.scalar('cross_entropy', loss)
    X_train=X_train[0:len(X_train)-1]
    y_train=y_train[0:len(y_train)-1]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0,(len(X_train)/batch_size)-1):
                print i
                if i==0:
                    xtrain=X_train[i:i+8]
                    ytrain=y_train[i:i+8]
                else:
                    xtrain=X_train[i*8:(i*8)+8]
                    ytrain=y_train[i*8:(i*8)+8]

                print "___________________________________"
                c,_ = sess.run([loss, optimizer], feed_dict={nn_model.x: xtrain, nn_model.y:ytrain})
                print c
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(output_map, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

#def predict(path):
#    img=cv2.imread(path)
#    x=tf.placeholder("float", shape=[1, img_rows, img_cols, 3])
#    nn_model=model.UNet(x)



train()
