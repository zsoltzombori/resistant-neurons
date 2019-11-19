#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import sklearn
import pandas as pd
import scipy.stats as stats
import gzip

neuron_data = {}
files = sorted(os.listdir('../neuron_logs/train_data'))

def reduce_to_statistics(activations, labels, debug=False):
    sorted_data = []
    for i in range(10): #hardcoded MOFO
        sorted_data += [[]]
    for i, a in zip(labels, activations):
        sorted_data[i] += [a]
    if debug:
        return(sorted_data)
    statistics = []
    for ar in sorted_data:
        curr_stats = stats.describe(ar)
        statistics += [curr_stats.mean, curr_stats.variance, curr_stats.skewness, curr_stats.kurtosis, curr_stats.minmax[0],
                       curr_stats.minmax[1], curr_stats.nobs]
        #print(statistics)
    return(statistics)

def extract_data(filename, fin = 10, activations_no = 1000, target = 'usefulness_loss', shuffle = True):
    
    features, labels = [], []
    with open(os.path.join('..', 'neuron_logs', 'train_data', filename), 'r') as f:
        neuron_data = json.load(f)
        for e in neuron_data.keys():
            if e == '0' or int(e) > fin:
                continue
            for neuron in neuron_data[e]:
                if ' ' not in neuron:
                    continue
                current_data = neuron_data[e][neuron]
                important_features = []
                important_features += [filename]
                important_features += [current_data['depth']]
                important_features += [current_data['inverse_depth']]
                important_features += [current_data['width']]
                # important_features += [current_data['input_weights']]
                # important_features += [current_data['output_weights']]
                important_features += [current_data['reg_loss_in_layer']]
                important_features += current_data['activations'][:activations_no]
                important_features += reduce_to_statistics(current_data['activations'], neuron_data[e]['original_labels'])
                important_features += [e]
                important_features += [current_data[target]]
                features += [important_features]
                #labels += [current_data[target]]
    
    return(features)


import csv 

with gzip.open('test_data_2.gz', 'wt', compresslevel=5) as f:
    writer = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i, filename in enumerate(files[1:]):
        print(f'Opening file {filename}')
        data = extract_data(filename)
        for line in data:
            writer.writerow(line)
    



