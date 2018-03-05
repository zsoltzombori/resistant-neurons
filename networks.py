import tensorflow as tf

def DenseNet(inputs, depth, width, bn, output_count):
    activations = []
    output = tf.reshape(inputs, (inputs.shape[0], -1))
    for i in range(depth):
        output = tf.layers.dense(output, width)
        if bn:
            output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        activations.append(output)

    output = tf.layers.dense(output, output_count)
    return output, activations

