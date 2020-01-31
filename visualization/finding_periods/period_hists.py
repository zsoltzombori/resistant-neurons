import gzip, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from joblib import dump, load
import os, json
import collections
import numpy.fft as fft
import glob

def find_period(data):
    a=np.abs(fft.rfft(data))
    #Not sure if this is a good idea but seems to help with choppy data..
    a[0] = 0
    freqs = fft.rfftfreq(n=data.size, d=1)
    freqs = np.divide(1,freqs)
    max_freq = freqs[np.argmax(a)]
    return(max_freq)

plt.style.use('ggplot')

basepath = 'neuron_logs/shuffled/'
save_path = 'visualization/finding_periods/period_hists/'
files = [x.split('/')[-1] for x in glob.glob(basepath + '*.json.gz')]
print(f'files are: \n{files}')

activations_no = 1000
usefulness_per_neuron = collections.defaultdict(dict)
target = 'usefulness_loss'

for fname in files:
    with gzip.open(os.path.join(basepath, fname), 'rt') as f:
        neuron_data = json.load(f)
        
    for e in neuron_data.keys():
        for neuron in neuron_data[e]:
            if ' ' not in neuron:
                continue
            current_data = neuron_data[e][neuron]
            important_features = []
            important_features += [current_data['depth']]
            important_features += [current_data['inverse_depth']]
            important_features += [current_data['width']]
            # important_features += [current_data['input_weights']]
            # important_features += [current_data['output_weights']]
            important_features += [current_data['reg_loss_in_layer']]
            important_features += current_data['activations'][:activations_no]
            usefulness_gold = current_data[target]
            line_of_data = np.array(important_features, dtype = np.float32).reshape(1, -1)
            usefulness_per_neuron[e][neuron] = usefulness_gold

    WIDTH = len([x for x in neuron_data['0'] if x[0] == '0'])
    # print()
    DEPTH = len(np.unique([int(x.split(' ')[0]) for x in neuron_data['0'] if ' ' in x]))

    all_periods = []
    layers = [str(x) for x in range(DEPTH)]
    for layer in layers:
        layer = str(layer)
        periods = []
        for pos in range(WIDTH):
            pos = str(pos)
            data = np.array([usefulness_per_neuron[e][f'{layer} {pos}'] for e in usefulness_per_neuron])
            period_time = find_period(data)
            periods += [period_time]



        all_periods += [periods]
        # fig, ax = plt.subplots()
    all_periods = np.array(all_periods, dtype = float).T


    plt.figure(figsize = (12, 9))
    plt.suptitle(f'Distribution of period times of usefulness in network\n{fname}')
    plt.subplot(211)
    bins = np.arange(0, 111, 10)
    plt.ylim(0, 100)
    plt.xlim(0, 110)
    plt.hist(all_periods, bins, label = [f'layer {l}' for l in layers])
    plt.xticks(bins)
    plt.xlabel('period time')
    plt.ylabel('number of neurons')
    plt.grid(b=True)
    plt.subplot(212)
    # plt.hist(periods[:, 0].astype(float), bins = bins, alpha=1, linewidth = 5, histtype = 'stepfilled', label = f'layer {layer}')
    plt.grid(b=True)
    bins = np.arange(0, 20, 1)
    plt.xticks(bins)
    plt.yticks(range(0, 31, 2))
    plt.ylim(0, 30)
    plt.xlim(0, 20)
    plt.xlabel('period time')
    plt.ylabel('number of neurons')
    plt.hist(all_periods, histtype='bar', bins = bins, label = [f'layer {l}' for l in layers])


    plt.legend()
    plt.savefig(f'{save_path}{fname.split(".json")[0]}.png', dpi=300)
