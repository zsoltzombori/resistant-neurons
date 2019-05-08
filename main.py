import tensorflow as tf
import numpy as np
import time
import os
import sys
import json

import networks
import data

DATASET = "mnist"
TRAINSIZE = 20000
SEED = None
BN_DO = 'DO'  # "BN" (batchnorm), "DO" (dropout), None
BATCH_SIZE = 100
DEPTH = 4
WIDTH = 20
OUTPUT_COUNT = 10
LR = 0.001
MEMORY_SHARE = 0.25
ITERS = 10000
AUGMENTATION = False
SESSION_NAME = "tmp_{}".format(time.strftime('%Y%m%d-%H%M%S'))
BN_WEIGHT = 0
COV_WEIGHT = 0
CLASSIFIER_TYPE = "dense"  # "conv" / "dense"
LOG_DIR = "logs/%s" % SESSION_NAME
os.system("rm -rf {}".format(LOG_DIR))


def heuristic_cast(s):
    s = s.strip()  # Don't let some stupid whitespace fool you.
    if s == "None":
        return None
    elif s == "True":
        return True
    elif s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


for k, v in [arg.split('=', 1) for arg in sys.argv[1:]]:
    assert v != '', "Malformed command line"
    assert k.startswith('--'), "Malformed arg %s" % k
    k = k[2:]
    assert k in locals(), "Unknown arg %s" % k
    v = heuristic_cast(v)
    print("Changing argument %s from default %s to %s" % (k, locals()[k], v))
    locals()[k] = v


##########################################

(X_train, y_train), (X_devel, y_devel), (X_test,
                                         y_test) = data.load_data(DATASET, SEED)

X_train = X_train[:TRAINSIZE]
y_train = y_train[:TRAINSIZE]
INPUT_SHAPE = X_train.shape[1:]
train_gen = data.classifier_generator((X_train, y_train), BATCH_SIZE,
                                      augment=AUGMENTATION)
inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE] + list(INPUT_SHAPE))
if CLASSIFIER_TYPE == "dense":
    output, activations, zs = networks.DenseNet(inputs, DEPTH, WIDTH, BN_DO,
                                                OUTPUT_COUNT, dropout=1)
elif CLASSIFIER_TYPE == "conv":
    output, activations, zs = networks.LeNet(
        inputs, BN_DO, OUTPUT_COUNT, dropout=0.8)


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

# Regularization

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01
total_loss += reg_constant * tf.math.reduce_sum(reg_losses)
# activations = activations[:1] # TODO

cov_ops = []
# push the off diagonal elements of the featurewise correlation matrix to zero
if COV_WEIGHT > 0:
    cov_loss = tf.constant(0.0)
    cov_ops = []
    for act in activations:
        feature_count = int(act.shape[1])
        act_centered = act - tf.reduce_mean(act, axis=0)
        for i in range(feature_count):
            for j in range(i, feature_count):
                covariance, cov_op = tf.contrib.metrics.streaming_covariance(
                    act_centered[i], act_centered[j])
                cov_ops.append(cov_op)
                cov_loss += COV_WEIGHT * tf.square(covariance)
                total_loss += cov_loss
                loss_list.append(('cov_loss', cov_loss))

# push each neuron to have zero mean and unit variance output
if BN_WEIGHT > 0:
    bn_loss = tf.constant(0.0)
    for z in zs:
        z_mean = tf.reduce_mean(tf.square(z), axis=0)
        z_variance = tf.reduce_sum(tf.square(z - z_mean), axis=0)
        #        bn_loss += tf.reduce_sum(z_mean)
        bn_loss += tf.reduce_sum(tf.square(z_variance - 1))
        bn_loss = BN_WEIGHT * bn_loss
        total_loss += bn_loss
        loss_list.append(('bn_loss', bn_loss))


# TODO other losses

loss_list.append(('total_loss', total_loss))

log_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
loss_summaries = []
for (name, loss) in loss_list:
    loss_summaries.append(tf.summary.scalar(name, loss))

# keep track of how significant activations are as reflected by the magnitude of outgoing weights
kernels = [v for v in tf.trainable_variables() if (
    v.name.find("dense_") == 0) and v.name.find("kernel") > 0]
for kernel in kernels:
    k = tf.reduce_sum(tf.abs(kernel), axis=1)
    k = k / tf.reduce_sum(k)
    loss_summaries.append(tf.summary.histogram(kernel.name, k))

merged_loss_summary_op = tf.summary.merge(loss_summaries)


optimizer = tf.train.AdamOptimizer(
    learning_rate=LR
).minimize(total_loss)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = MEMORY_SHARE
session = tf.Session(config=config)
print("NETWORK PARAMETER COUNT",
      np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
activations_dict = {}


def evaluate(Xs, ys, BATCH_SIZE):
    nonzeros = []
    for a in activations:
        assert len(a.shape) == 2
        nonzeros.append(np.zeros(int(a.shape[1])))

    eval_gen = data.classifier_generator((Xs, ys), BATCH_SIZE, infinity=False)
    _total_losses = []
    _total_acc = []
    for X_batch, y_batch in eval_gen:
        value_list = session.run([total_loss, output, activations] + list(cov_ops),
                                 feed_dict={inputs: X_batch, labels: y_batch})
        (_total_loss, predicted, _activations) = value_list[:3]
        _total_acc.append(accuracy(predicted, y_batch))
        _total_losses.append(_total_loss)
        for i, a in enumerate(_activations):
            nonzeros[i] += np.count_nonzero(a, axis=0)

    eval_loss = np.mean(_total_losses)
    eval_acc = np.mean(_total_acc)
    for i, _ in enumerate(nonzeros):
        nonzeros[i] = nonzeros[i] * 1.0 / (len(Xs))
        nonzeros[i] = np.histogram(nonzeros[i], bins=10, range=(0.0, 1.0))[0]
    return eval_loss, eval_acc, nonzeros, activations


def accuracy(predicted, expected):
    return float(np.sum(np.argmax(predicted, axis=1) == expected)) / len(predicted)


start_time = time.time()

for iteration in range(ITERS+1):
    train_data = next(train_gen)
    # training step
    _, _total_loss, predicted, loss_summary = session.run(
        [optimizer, total_loss, output, merged_loss_summary_op],
        feed_dict={inputs: train_data[0], labels: train_data[1]}
    )
    log_writer.add_summary(loss_summary, iteration)

    # eval step
    if iteration % 500 == 0:
        train_acc = accuracy(predicted, train_data[1])
        eval_loss, eval_acc, nonzeros, current_activations = evaluate(
            X_devel, y_devel, BATCH_SIZE)
        print(_total_loss)
        print('Total loss:{:.3f}, L reg:{}'.format(_total_loss,
                                                   session.run(reg_losses)))
        # print(session.run([reg_losses]))
        # print(tf.math.reduce_sum(reg_losses))
        print("{:>5}:    train acc {:.2f}    dev acc {:.2f}".format(
            iteration, train_acc, eval_acc))
        # print(zs)
        for line in nonzeros:
            print(' '.join(['{: <2}'.format(x) for x in line]))
        print()
        activations_dict[iteration] = session.run([current_activations],
                                                  feed_dict={inputs:
                                                             X_devel[:BATCH_SIZE],
                                                             labels:
                                                             y_devel[:BATCH_SIZE]})
print("Total time: {}".format(time.time() - start_time))
for it in activations_dict:
    activations_dict[it] = [[nda.tolist() for nda in lst]
                            for lst in activations_dict[it]]
with open('neuron_logs/output_{}'.format(time.strftime('%Y%m%d-%H%M%S')), 'w') as f:
    json.dump(activations_dict, f)
