import tensorflow as tf
import numpy as np
import time
import os
import sys
import json
import itertools
import networks
import data
import joblib
#needed for prediction
import gzip, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reviving

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

DATASET = "fashion_mnist"
TRAINSIZE = 60000
SEED = None
BN_DO = None  # "BN" (batchnorm), "DO" (dropout), None
BATCH_SIZE = 500
DEPTH = 5
WIDTH = 100
OUTPUT_COUNT = 10
LR = 0.001
MEMORY_SHARE = 0.25
ITERS = 100
EVALUATION_CHECKPOINT = 20
AUGMENTATION = False
SESSION_NAME = "tmp_{}".format(time.strftime('%Y%m%d-%H%M%S'))
BN_WEIGHT = 0
COV_WEIGHT = 0
CLASSIFIER_TYPE = "dense"  # "conv" / "dense"
LOG_DIR = "logs/%s" % SESSION_NAME
EVALUATE_USEFULNESS = False
USEFULNESS_EVAL_SET_SIZE = 1000

os.system("rm -rf {}".format(LOG_DIR))
# os.nice(20)


##########################################

dummy_mask = np.ones((DEPTH, WIDTH))
# dummy_mask[0, :] = np.zeros(WIDTH)
# dummy_mask[0, 2:5] = np.ones(3)

(X_train, y_train), (X_devel, y_devel), (X_test,
                                         y_test) = data.load_data(DATASET, SEED, USEFULNESS_EVAL_SET_SIZE)

X_train = X_train[:TRAINSIZE]
y_train = y_train[:TRAINSIZE]
INPUT_SHAPE = X_train.shape[1:]
train_gen = data.classifier_generator((X_train, y_train), BATCH_SIZE,
                                      augment=AUGMENTATION)
inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE] + list(INPUT_SHAPE))
mask = tf.placeholder(tf.float32, shape=[DEPTH, WIDTH])
if CLASSIFIER_TYPE == "dense":
    output, activations, zs = networks.DenseNet(inputs, DEPTH, WIDTH, BN_DO,
                                                OUTPUT_COUNT, dropout=0.5, mask=mask)
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

config = tf.ConfigProto(device_count={'GPU': 2})
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = MEMORY_SHARE
session = tf.Session(config=config)
print("NETWORK PARAMETER COUNT",
      np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
activations_dict = {}
zs_dict = {}

def evaluate(Xs, ys, BATCH_SIZE, net_mask = dummy_mask):
    nonzeros = []
    for a in activations:
        assert len(a.shape) == 2
        nonzeros.append(np.zeros(int(a.shape[1])))

    eval_gen = data.classifier_generator((Xs, ys), BATCH_SIZE, infinity=False)
    _total_losses = []
    _total_acc = []
    labels_to_return = [[], []]
    # print('we are before entering eval_gen')
    for X_batch, y_batch in eval_gen:
        value_list = session.run([total_loss, output, activations] + list(cov_ops),
                                 feed_dict={inputs: X_batch, labels: y_batch, mask: net_mask})
        (_total_loss, predicted, _activations) = value_list[:3]
        # labels_to_return += [(y_batch, np.argmax(predicted, axis=1))]
        labels_to_return[0] += [y_batch.tolist()]
        labels_to_return[1] += [np.argmax(predicted, axis=1).tolist()]
        _total_acc.append(accuracy(predicted, y_batch))
        _total_losses.append(_total_loss)
        for i, a in enumerate(_activations):
            nonzeros[i] += np.count_nonzero(a, axis=0)

    eval_loss = np.mean(_total_losses)
    eval_acc = np.mean(_total_acc)
    for i, _ in enumerate(nonzeros):
        nonzeros[i] = nonzeros[i] * 1.0 / (len(Xs))
        nonzeros[i] = np.histogram(nonzeros[i], bins=10, range=(0.0, 1.0))[0]

    return eval_loss, eval_acc, nonzeros, activations, zs, labels_to_return


def evaluate_usefulness(Xs, ys, usefulness_mask):
    for a in activations:
        assert len(a.shape) == 2

        _total_losses = []
        _total_acc = []
        # VERY QUCIK AND REAL DIRTY
        # reinitializing eval_gen between calls is very costly, so we need to make an infinite one
        # however, we lose deterministric results this way
        # also, we manually have to specify when to stop in this cycle below

        i = 0
        for X_batch, y_batch in EVAL_GEN:
            if i >= np.ceil(USEFULNESS_EVAL_SET_SIZE/BATCH_SIZE):
                # print(f'breaking at epoch i: {i}')
                break
            value_list = session.run([total_loss, output, activations] + list(cov_ops),
                                     feed_dict={inputs: X_batch, labels: y_batch, mask: usefulness_mask})
            (_total_loss, predicted, _activations) = value_list[:3]
            _total_acc.append(accuracy(predicted, y_batch))
            _total_losses.append(_total_loss)
            i += 1

    eval_loss = np.mean(_total_losses)
    eval_acc = np.mean(_total_acc)

    return eval_loss, eval_acc


def accuracy(predicted, expected):
    return float(np.sum(np.argmax(predicted, axis=1) == expected)) / len(predicted)


def create_0_mask(dep, wid):
    new_mask = np.ones((DEPTH, WIDTH))
    new_mask[dep, wid] = 0
    return(new_mask)


def find_weights(d, w):
    trainables = dict([(v.name, v) for v in tf.trainable_variables()])
    input_weights = session.run(trainables[f"dense_{d}/kernel:0"])[:, w]
    output_weights = session.run(trainables[f"dense_{d+1}/kernel:0"])[w, :]
    return(input_weights, output_weights)
   

cumulative_dictionary = {}

start_time = time.time()
TRAIN_ITER = 0
iteration_no = 0
saver = tf.train.Saver(max_to_keep=20)

for iteration in range(ITERS+1):

    TRAIN_ITER = iteration
    # print(f'iteration number {iteration}')
    train_data = next(train_gen)
    # training step
    _, _total_loss, predicted, loss_summary = session.run(
        [optimizer, total_loss, output, merged_loss_summary_op],
        feed_dict={inputs: train_data[0], labels: train_data[1], mask: dummy_mask} 
    )
    log_writer.add_summary(loss_summary, iteration)
    usefulness_dict = dict([(f"{d} {w}", []) for d in range(DEPTH) for w in range(WIDTH)])

    # eval step
    if iteration % EVALUATION_CHECKPOINT == 0:
        train_acc = accuracy(predicted, train_data[1])
        eval_loss, eval_acc, nonzeros, current_activations, current_zs, labels_evaluated =\
            evaluate(X_devel, y_devel, BATCH_SIZE)

        saver.save(session, f'saved_models/{SESSION_NAME}', global_step=iteration)
        # print(_total_loss)
        # print('Total loss:{:.3f}, L reg:{}'.format(_total_loss,
        # session.run(reg_losses)))
        # print(session.run([reg_losses]))
        # print(tf.math.reduce_sum(reg_losses))
        print("{:>5}:    train acc {:.3f}    dev acc {:.3f}".format(
            iteration, train_acc, eval_acc))
        # print(zs)
        # print(f"HEREWEGO:{nonzeros}")

        # for line in nonzeros:
        #     print(' '.join(['{: <2}'.format(x) for x in line]))

        # print(current_zs)
        # print(len(X_devel))
        zs_evaluated = np.empty((DEPTH, 1, WIDTH))
        for i in range(0, len(X_devel), BATCH_SIZE):
            current = [session.run([current_zs],
                                   feed_dict={inputs: X_devel[i:i+BATCH_SIZE],
                                              labels: y_devel[i:i+BATCH_SIZE],
                                              mask: dummy_mask})]
            current = np.squeeze(np.array(current))
            zs_evaluated = np.concatenate((zs_evaluated, current), axis=1)

        # zs_dict[iteration][depth][no_of_images][width]
        # zs_dict[iteration_no] = zs_evaluated[:, 1:, :].tolist()
        zs_dict[iteration_no] = zs_evaluated[:, 1:, :]

        if EVALUATE_USEFULNESS:

            usefulness_starttime = time.time()
            EVAL_GEN = data.classifier_generator((X_devel, y_devel), BATCH_SIZE, infinity=True)
            cumulative_dictionary[iteration_no] = {}

            # print(f'length of devel: {len(X_devel)}')

            for d in range(DEPTH):
                for w in range(WIDTH):
                    current_mask = create_0_mask(d, w)
                    u_eval_loss, u_eval_acc = evaluate_usefulness(X_devel, y_devel, current_mask)
                    if u_eval_acc == 0:
                        usefulness_a = 0
                    else:
                        usefulness_a = eval_acc/u_eval_acc

                    if eval_loss == 0:
                        usefulness_l = 0
                    else:
                        usefulness_l = u_eval_loss/eval_loss

                    current_neuron = f"{d} {w}"
                    usefulness_dict[current_neuron] += [(usefulness_l, usefulness_a)]

                    # print(f"Loc: ({d}, {w}), loss and acc: {u_eval_loss:.3f}, {u_eval_acc:.3f}")
                    # print(f"Loc: ({d}, {w}), acc and loss usefulness: {usefulness_a:.3f}, {usefulness_l:.3f}")
                    # print(f"Usefulness_in_a: {usefulness_a:.3f}")
                    # print(f"Usefulness_in_l: {usefulness_l:.3f}")
                    # if loss gets higher without neuron -> not useful = ratio < 1
                    # if accuracy gets lower without neuron -> not useful (reciprocated) = ratio < 1
                    usefulness_endtime = time.time()
                    usefulness_elapsed = usefulness_endtime-usefulness_starttime
                    in_w, ou_w = find_weights(d, w)
                    # CUMULATIVE_DICT SECTION

                    temp_cum = {'activations': zs_dict[iteration_no][d, :, w].tolist(),
                                'usefulness_loss': usefulness_l.tolist(),
                                'usefulness_acc': usefulness_a.tolist(),
                                'depth': d,
                                'inverse_depth': DEPTH-1-d,
                                'width': w,
                                'input_weights': in_w.tolist(),
                                'output_weights': ou_w.tolist(),
                                'reg_loss_in_layer': session.run(reg_losses)[d].tolist()}

                    cumulative_dictionary[iteration_no][current_neuron] = temp_cum

            print(f"""Usefulness loop time: {usefulness_elapsed:.2f} seconds, with
                  {usefulness_elapsed/(DEPTH*WIDTH):.2f} seconds per
                  subloop.""")
            cumulative_dictionary[iteration_no]['original_labels'] = list(
                itertools.chain.from_iterable(labels_evaluated[0]))
            cumulative_dictionary[iteration_no]['predicted_labels'] = list(
                itertools.chain.from_iterable(labels_evaluated[1]))

        iteration_no += 1




print("Total time: {}".format(time.time() - start_time))

def make_1_iteration(target='valid', logging = 1, net_mask = dummy_mask):

    # makes one train iteration with logging;
    # useful for watching the effects of neuron changing
    # target can be 'train' or 'valid'


    if target == 'train':
        train_data = next(train_gen)
        # training step
        _, _total_loss, predicted, loss_summary = session.run(
            [optimizer, total_loss, output, merged_loss_summary_op],
            feed_dict={inputs: train_data[0], labels: train_data[1], mask: net_mask}
        )
        train_acc = accuracy(predicted, train_data[1])
        global TRAIN_ITER
        TRAIN_ITER += 1
        

    # eval step

    eval_loss, eval_acc, nonzeros, current_activations, current_zs, labels_evaluated =\
        evaluate(X_devel, y_devel, BATCH_SIZE, net_mask = net_mask)
    # print()
    if TRAIN_ITER % logging == 0:
        if target == 'train':
            print("{:>5}:    train acc {:.3f}    dev acc {:.3f}".format(
                TRAIN_ITER, train_acc, eval_acc))
        else:
            print("{:>5}:  dev acc {:.3f}".format(
                TRAIN_ITER, eval_acc))



make_1_iteration()
usefulness_per_neuron = reviving.get_data_and_predict(session, X_devel, y_devel, BATCH_SIZE, DEPTH, WIDTH, 
                inputs, labels, mask, dummy_mask, reg_losses, evaluate)
classifications = reviving.classificate_neurons(usefulness_per_neuron, 50, WIDTH, DEPTH)
trainables = dict([(v.name, v) for v in tf.trainable_variables()])
reviving.revive_genetic_algorithm_good_plus_bad(session, DEPTH, WIDTH, trainables, classifications)
make_1_iteration()

for i in range(200):
    
    make_1_iteration('train', logging = 20)
    



if EVALUATE_USEFULNESS:
    with open('neuron_logs/train_data/output_{}.json'.format(SESSION_NAME), 'w') as f:
        json.dump(cumulative_dictionary, f)

