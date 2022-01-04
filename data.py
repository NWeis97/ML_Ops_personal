#Import for data injection
from numpy.lib.npyio import load
import torch
import numpy as np
from torchvision import datasets, transforms
import pdb

def mnist(batch_size):
    #Load data
    Test = dict(np.load('corruptmnist/test.npz'))

    #Concatenate training datasets
    #Inspiration from https://coderedirect.com/questions/615101/how-to-merge-very-large-numpy-arrays

    #Define train datasets to load and dicts
    data_files = ['corruptmnist/train_0.npz','corruptmnist/train_1.npz','corruptmnist/train_2.npz','corruptmnist/train_3.npz','corruptmnist/train_4.npz']
    n_items = {'images':0,'labels':0,'allow_pickle':0}
    rows = {'images':None,'labels':None,'allow_pickle':None}
    cols = {'images':None,'labels':None,'allow_pickle':None}
    dtype = {'images':None,'labels':None,'allow_pickle':None}

    #Load all files and check for size
    for data_file in data_files:
        with np.load(data_file) as data:
            keys = list(rows.keys())
            for i in keys:
                chunk = data[i]
                #pdb.set_trace()
                try:
                    n_items[i] += chunk.shape[0]
                except:
                    n_items[i] = 1
                    #Set to 1
                try:
                    rows[i] = chunk.shape[1]
                except:
                    rows[i] = 1
                    #Do nothing
                try:
                    cols[i] = chunk.shape[2]
                except:
                    cols[i] = 1
                    #Do nothing
                dtype[i] = chunk.dtype

    #Initialize training dataset
    Train = {}

    # Once the size is know create concatenated version of data
    keys_new = keys[0:2]
    for i in keys_new:
        #pdb.set_trace()
        merged = np.empty(shape=(n_items[i],rows[i], cols[i]),dtype=dtype[i])
        merged = np.squeeze(merged)   

        idx = 0
        for data_file in data_files:
            with np.load(data_file) as data:
                chunk = data[i]
                merged[idx:idx + len(chunk)] = chunk
                idx += len(chunk)
        Train[i] = merged


    #Convert to dataloader object
    #pdb.set_trace()
    Train = torch.utils.data.TensorDataset(torch.Tensor(Train['images']), torch.Tensor(Train['labels']).type(torch.LongTensor))
    trainloader = torch.utils.data.DataLoader(Train, batch_size=batch_size, shuffle=True)
    Test = torch.utils.data.TensorDataset(torch.Tensor(Test['images']), torch.Tensor(Test['labels']).type(torch.LongTensor))
    testloader = torch.utils.data.DataLoader(Test, batch_size=batch_size, shuffle=True)


    #Return datasets
    return [trainloader,testloader]



#From images it seems like noise has been added for single pixels (one pixel in the perimeter is a different color than the rest)
#image, label = next(iter(trainloader))
#import matplotlib.pyplot as plt
#plt.imshow(image[0,:]);
#plt.show()
