import tensorflow as tf
import numpy as np
import time
import os

import networks
import data

DATASET="mnist"
SEED=None
BN=False
BATCH_SIZE=50
DEPTH=3
WIDTH=20
OUTPUT_COUNT = 10
LR=0.001
MEMORY_SHARE=0.25
ITERS=10000
AUGMENTATION=False
SESSION_NAME="tmp"
COV_WEIGHT = 0.01


(X_train, y_train), (X_devel, y_devel), (X_test, y_test) = data.load_data(DATASET, SEED)
INPUT_SHAPE = X_train.shape[1:]
train_gen = data.classifier_generator((X_train, y_train), BATCH_SIZE, augment=AUGMENTATION)


inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE] + list(INPUT_SHAPE))
output, activations = networks.DenseNet(inputs, DEPTH, WIDTH, BN, OUTPUT_COUNT)

labels = tf.placeholder(tf.uint8, shape=[BATCH_SIZE])
labels_onehot = tf.one_hot(labels, OUTPUT_COUNT)

loss_list = []

xent_loss = tf.nn.softmax_cross_entropy_with_logits(
    logits=output,
    labels=labels_onehot
    )
xent_loss = tf.reduce_mean(xent_loss)
loss_list.append(('xent_loss', xent_loss))
total_loss = xent_loss

# push the off diagonal elements of the featurewise correlation matrix to zero
activations = activations[:1] # TODO
cov_loss = tf.constant(0.0)
for act in activations:
    feature_count = int(act.shape[1])
    act_centered = act - tf.reduce_mean(act, axis=0)
    for i in range(feature_count):
        for j in range(i, feature_count):
            covariance = tf.reduce_mean(act_centered[i] * act_centered[j])
            cov_loss += tf.square(covariance)
cov_loss = COV_WEIGHT * cov_loss
total_loss += cov_loss
loss_list.append(('cov_loss', cov_loss))

# TODO other losses

loss_list.append(('total_loss', total_loss))

optimizer = tf.train.AdamOptimizer(
    learning_rate=LR
).minimize(total_loss)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = MEMORY_SHARE
session = tf.Session(config=config)

session.run(tf.global_variables_initializer())

LOG_DIR = "logs/%s" % SESSION_NAME
os.system("rm -rf {}".format(LOG_DIR))
log_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
loss_summaries = []
for (name, loss) in loss_list:
    loss_summaries.append(tf.summary.scalar(name, loss))
merged_loss_summary_op = tf.summary.merge(loss_summaries)


def evaluate(Xs, ys, BATCH_SIZE):
    eval_gen = data.classifier_generator((Xs, ys), BATCH_SIZE, infinity=False)
    _total_losses = []
    _total_acc = []
    for X_batch, y_batch in eval_gen:
        
        (_total_loss,predicted) = session.run([total_loss, output], feed_dict={inputs:X_batch, labels:y_batch})
        _total_acc.append(accuracy(predicted, y_batch))
        _total_losses.append(_total_loss)
    eval_loss = np.mean(_total_losses)
    eval_acc = np.mean(_total_acc)
    return eval_loss, eval_acc

def accuracy(predicted, expected):
    return float(np.sum(np.argmax(predicted, axis=1) == expected)) / len(predicted)


start_time = time.time()

for iteration in xrange(ITERS+1):
    train_data = train_gen.next()

    # training step
    _, _total_loss, predicted, loss_summary = session.run(
        [optimizer, total_loss, output, merged_loss_summary_op],
        feed_dict={inputs: train_data[0], labels: train_data[1]}
    )
    log_writer.add_summary(loss_summary, iteration)

    # eval step
    if iteration % 200 == 0:
        train_acc = accuracy(predicted, train_data[1])
        eval_loss, eval_acc = evaluate(X_devel, y_devel, BATCH_SIZE)
        print("{}:\t train acc {},\t dev acc {}").format(iteration, train_acc, eval_acc)
    

print "Total time: {}".format (time.time() - start_time)

