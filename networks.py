import tensorflow as tf

def DenseNet(inputs, depth, width, bn_do, output_count, dropout=0.5):
    activations = []
    zs = []
    output = tf.reshape(inputs, (inputs.shape[0], -1))
    for i in range(depth):
        output = tf.layers.dense(output, width, name="dense_{}".format(i))
        zs.append(output)
        if bn_do == "BN":
            output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        activations.append(output)

    if bn_do == "DO":
        output = tf.nn.dropout(output, dropout)

    output = tf.layers.dense(output, output_count, name="dense_{}".format(depth))
    return output, activations, zs


def LeNet(inputs, bn_do, output_count, dropout=0.5):
    activations = []
    zs = []
    output = inputs
    output = tf.layers.conv2d(output, 6, (5,5), padding="same", name="conv_1")
    zs.append(output)
    output = tf.nn.relu(output)
    activations.append(tf.contrib.layers.flatten(output))
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    output =  tf.layers.conv2d(output, 16, (5,5), padding="valid", name="conv_2")
    zs.append(output)
    output = tf.nn.relu(output)
    activations.append(tf.contrib.layers.flatten(output))
    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    output =  tf.layers.conv2d(output, 120, (5,5), padding="valid", name="conv_3")
    zs.append(output)
    output = tf.nn.relu(output)
    activations.append(tf.contrib.layers.flatten(output))

    output = tf.reshape(output ,[-1,120])

    output = tf.layers.dense(output, 84, name="dense_1")
    output = tf.nn.relu(output)

    if bn_do == "DO":
        output = tf.nn.dropout(output, dropout)
        
    output = tf.layers.dense(output, output_count, name="dense_2")
    return output, activations, zs
