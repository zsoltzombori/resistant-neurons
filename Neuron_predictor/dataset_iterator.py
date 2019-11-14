#!/usr/bin/env python
# coding: utf-8

# In[20]:


import gzip
import numpy as np
from torch.utils.data import IterableDataset

class CustomIterableDataset(IterableDataset):

    def __init__(self, filename):

        #Store the filename in object's memory
        self.filename = filename

        #And that's it, we no longer need to store the contents in the memory

    def preprocess(self, feature):

        ### Do something with text here
        feature = np.array(feature, dtype=np.float32)
        ###

        return feature

    def line_mapper(self, line):
        
        #Splits the line into text and label and applies preprocessing to the text
        line = line.strip().split('\t')
        fname, feature, label = line[0], line[1:-1], line[-1]
        feature = self.preprocess(feature)
        label = np.array(label, dtype = np.float32)
        

        return feature, label.reshape(-1, 1)


    def __iter__(self):

        #Create an iterator
        file_itr = gzip.open(self.filename, 'rt')

        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)
        
        return mapped_itr


# In[ ]:




